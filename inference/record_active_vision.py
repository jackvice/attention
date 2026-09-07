#!/usr/bin/env python3
"""
Standalone recorder for active-vision videos.

Subscribes to the raw camera topic and reads the window index that Dreamer
writes into the `camera_latest` shared memory block, then writes an MP4 of the
full camera view with the policy-selected window outlined.

Nothing in the training pipeline is modified: this attaches read-only to shared
memory and adds one extra subscriber to the camera topic. Start and stop it
whenever you want a take.

Run it in the same environment used for fisheye_ros2_mem_share.py:

    cd ~/src/attention/inference
    python record_active_vision.py --output ~/videos/active_vision.mp4

Ctrl-C to stop; the buffer is flushed and the file finalized on exit.
"""
import argparse
import mmap
import os
import signal
import struct
import sys
import time
from multiprocessing import resource_tracker, shared_memory
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
try:
    from rclpy._rclpy_pybind11 import RCLError
except ImportError:  # older rclpy builds
    RCLError = RuntimeError
from sensor_msgs.msg import Image

# Must match fisheye_ros2_mem_share.py. Duplicated rather than imported so this
# script has no import side effects on the producer's shared memory.
SHM_NAME = "camera_latest"
SHM_H, SHM_W = 320, 320
NUM_IMAGES = 6
CTRL_OFFSET = 8 + NUM_IMAGES * SHM_H * SHM_W * 3

SRC_W, SRC_H = 1600, 600
HFOV_SRC_RAD = 2.8
LUT_HW = 320

# Keep these in step with WINDOW_YAWS_DEG / WINDOW_HFOV_DEG in
# fisheye_ros2_mem_share.py, or the outlines will not match what the agent
# sees. Use --window-yaws / --window-fov to draw a different configuration,
# for instance to annotate footage from an older run.
PRODUCER_YAWS_DEG = (-60.0, -30.0, 0.0, 30.0, 60.0)
PRODUCER_HFOV_WIN_DEG = 40.0


def make_pinhole_to_rect_lut(
    src_w: int,
    src_h: int,
    dst_hw: int,
    hfov_dst_rad: float,
    yaw_deg: float,
    hfov_src_rad: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Copy of the producer's LUT builder, plus the ray-depth term.

    Returns map_x, map_y (source pixel coords) and zr, the ray z-component in
    camera frame. Samples with zr <= 0 are behind the image plane and their
    projected coordinates are meaningless.
    """
    xx, yy = np.meshgrid(
        np.linspace(-1.0, 1.0, dst_hw),
        np.linspace(-1.0, 1.0, dst_hw),
    )

    t_dst = np.tan(hfov_dst_rad / 2.0)
    x = xx * t_dst
    y = yy * t_dst
    z = np.ones_like(x)

    n = np.sqrt(x * x + y * y + z * z)
    x /= n
    y /= n
    z /= n

    yaw = np.deg2rad(yaw_deg)
    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)

    xr = cos_y * x + sin_y * z
    yr = y
    zr = -sin_y * x + cos_y * z

    fx = (src_w / 2.0) / np.tan(hfov_src_rad / 2.0)
    fy = fx

    cx = src_w / 2.0
    cy = src_h / 2.0

    zr_safe = np.where(zr == 0.0, 1e-6, zr)

    u = fx * (xr / zr_safe) + cx
    v = fy * (yr / zr_safe) + cy

    return u.astype(np.float32), v.astype(np.float32), zr.astype(np.float32)


def window_footprints(win_deg: float, yaws_deg: Sequence[float]) -> Dict[int, List[np.ndarray]]:
    """Outline, in source-image pixels, of the region each window samples.

    A window whose edge passes 90 deg off axis is past the singularity of the
    rectilinear source model, so part of its grid projects to garbage; those
    samples are dropped and the outline traces only the region actually
    covered. The current 40 deg windows reach +/-80 deg at most, so every
    sample is valid, but the older 60 deg setting reached +/-94 deg and did
    lose about a quarter of the outer two windows.
    """
    footprints: Dict[int, List[np.ndarray]] = {}
    for idx, yaw in enumerate(yaws_deg):
        u, v, zr = make_pinhole_to_rect_lut(
            src_w=SRC_W,
            src_h=SRC_H,
            dst_hw=LUT_HW,
            hfov_dst_rad=np.deg2rad(win_deg),
            yaw_deg=yaw,
            hfov_src_rad=HFOV_SRC_RAD,
        )
        valid = (
            (zr > 0)
            & np.isfinite(u)
            & np.isfinite(v)
            & (u >= 0)
            & (u < SRC_W)
            & (v >= 0)
            & (v < SRC_H)
        )
        mask = np.zeros((SRC_H, SRC_W), np.uint8)
        mask[v[valid].astype(np.int32), u[valid].astype(np.int32)] = 255
        # The mapping stretches badly near the edges, leaving pinholes in the
        # rasterized footprint; close them before tracing the boundary.
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        )
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > 50.0]
        footprints[idx] = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
    return footprints


class ControlBlock:
    """Read-only view of the producer's shared memory, however we got at it."""

    def __init__(self, buf, size: int, source: str, closers: List) -> None:
        self.buf = buf
        self.size = size
        self.source = source
        self._closers = closers

    def close(self) -> None:
        self.buf = None
        for closer in self._closers:
            try:
                closer()
            except Exception:
                pass
        self._closers = []


def _attach_by_name(shm_name: str) -> ControlBlock:
    """Attach by name, without letting our resource tracker unlink the segment.

    The segment belongs to fisheye_ros2_mem_share.py. Python 3.13 has
    track=False; before that the registration has to be undone by hand, or
    exiting this process destroys the producer's segment.
    """
    try:
        shm = shared_memory.SharedMemory(name=shm_name, track=False)
    except TypeError:  # track= is Python 3.13+
        shm = shared_memory.SharedMemory(name=shm_name)
        try:
            resource_tracker.unregister(shm._name, "shared_memory")
        except Exception as e:
            print(f"Warning: could not detach the resource tracker from "
                  f"{shm_name}; exiting may unlink it: {e}")
    return ControlBlock(shm.buf, shm.size, f"/dev/shm/{shm_name}", [shm.close])


def _attach_via_proc(shm_name: str) -> Optional[ControlBlock]:
    """Fall back to another process's open file descriptor.

    If the segment was unlinked while its users kept running, the name is gone
    from /dev/shm but the memory is still live and reachable through
    /proc/<pid>/fd of any process that still has it mapped.
    """
    target = f"/dev/shm/{shm_name}"
    for pid in sorted(os.listdir("/proc")):
        if not pid.isdigit():
            continue
        fd_dir = f"/proc/{pid}/fd"
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue
        for fd in fds:
            fd_path = f"{fd_dir}/{fd}"
            try:
                link = os.readlink(fd_path)
            except OSError:
                continue
            if link not in (target, f"{target} (deleted)"):
                continue
            try:
                raw_fd = os.open(fd_path, os.O_RDONLY)
                size = os.fstat(raw_fd).st_size
                mm = mmap.mmap(raw_fd, size, prot=mmap.PROT_READ)
            except OSError:
                continue
            return ControlBlock(
                mm, size, f"{fd_path} (pid {pid})",
                [mm.close, lambda f=raw_fd: os.close(f)],
            )
    return None


def attach_control(shm_name: str) -> Optional[ControlBlock]:
    try:
        block = _attach_by_name(shm_name)
    except FileNotFoundError:
        block = _attach_via_proc(shm_name)
        if block is None:
            print(f"Warning: shared memory '{shm_name}' not found and no process "
                  "has it open; recording without a window index. Is "
                  "fisheye_ros2_mem_share.py running?")
            return None
        print(f"Note: '{shm_name}' is not in /dev/shm but is still live; reading "
              f"it through {block.source}")

    if block.size < CTRL_OFFSET + 8:
        print(f"Warning: '{shm_name}' is {block.size} B, expected at least "
              f"{CTRL_OFFSET + 8} B; ignoring it.")
        block.close()
        return None
    return block


def read_window_index(block: Optional[ControlBlock]) -> Tuple[int, int]:
    if block is None:
        return -1, -1
    idx, step = struct.unpack_from("<ii", block.buf, CTRL_OFFSET)
    return idx, step


class ActiveVisionRecorder(Node):

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("active_vision_recorder")
        self.args = args
        self.shm = attach_control(args.shm_name)
        self.win_deg = args.window_fov
        self.yaws_deg = args.window_yaws
        self.footprints = window_footprints(self.win_deg, self.yaws_deg)
        self.hypothetical = (
            abs(self.win_deg - PRODUCER_HFOV_WIN_DEG) > 1e-6
            or tuple(self.yaws_deg) != tuple(PRODUCER_YAWS_DEG)
        )

        self.scale = args.scale
        self.out_w = int(round(SRC_W * self.scale))
        self.out_h = int(round(SRC_H * self.scale))

        self.buffer: List[np.ndarray] = []
        self.writer: Optional[cv2.VideoWriter] = None
        self.frames_written = 0
        self.frames_seen = 0
        self.first_stamp: Optional[float] = None
        self.last_stamp: Optional[float] = None

        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

        per_frame_mb = self.out_w * self.out_h * 3 / 1e6
        print(f"Recording {self.out_w}x{self.out_h} to {args.output}")
        print(f"Buffering {args.flush_every} frames "
              f"({per_frame_mb * args.flush_every:.0f} MB) between flushes")

        self.sub = self.create_subscription(Image, args.topic, self.on_image, 10)

    def on_image(self, msg: Image) -> None:
        if msg.width != SRC_W or msg.height != SRC_H:
            self.get_logger().warn(
                f"Frame is {msg.width}x{msg.height}, expected {SRC_W}x{SRC_H}; "
                "window outlines will not line up.",
                once=True,
            )

        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        if msg.encoding == "rgb8":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif msg.encoding == "bgr8":
            frame = frame.copy()
        else:
            self.get_logger().error(f"Unsupported encoding '{msg.encoding}'")
            return

        window_idx, step = read_window_index(self.shm)
        self.annotate(frame, window_idx, step)

        if self.scale != 1.0:
            frame = cv2.resize(frame, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)

        self.buffer.append(frame)
        self.frames_seen += 1

        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        if self.first_stamp is None:
            self.first_stamp = stamp
        self.last_stamp = stamp

        if len(self.buffer) >= self.args.flush_every:
            self.flush()

        if self.args.max_frames and self.frames_seen >= self.args.max_frames:
            raise KeyboardInterrupt

    def annotate(self, frame: np.ndarray, window_idx: int, step: int) -> None:
        for idx, contours in self.footprints.items():
            selected = idx == window_idx
            if selected:
                color, thickness = (0, 255, 0), 4
            elif self.args.show_all:
                color, thickness = (120, 120, 120), 1
            else:
                continue
            cv2.drawContours(frame, contours, -1, color, thickness)

        if window_idx < 0 or window_idx >= len(self.yaws_deg):
            label = "window: n/a"
        else:
            yaw = self.yaws_deg[window_idx]
            half = self.win_deg / 2.0
            label = (f"window {window_idx}  {yaw:+.0f} deg  "
                     f"[{yaw - half:+.0f}, {yaw + half:+.0f}]")
        if step >= 0:
            label += f"   step {step}"

        self.caption(frame, label, 46, (0, 255, 0))

        if False: #self.hypothetical:
            yaws = ",".join(f"{y:+.0f}" for y in self.yaws_deg)
            self.caption(
                frame,
                f"illustrative: {self.win_deg:g} deg windows at {yaws} "
                f"(pipeline renders {PRODUCER_HFOV_WIN_DEG:g} deg at "
                f"{','.join(f'{y:+.0f}' for y in PRODUCER_YAWS_DEG)})",
                88, (0, 200, 255), scale=0.7)

    @staticmethod
    def caption(frame: np.ndarray, text: str, y: int, color, scale: float = 1.1) -> None:
        cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color, 2, cv2.LINE_AA)

    def flush(self) -> None:
        if not self.buffer:
            return
        if self.writer is None:
            fourcc = cv2.VideoWriter_fourcc(*self.args.fourcc)
            self.writer = cv2.VideoWriter(
                self.args.output, fourcc, self.args.fps, (self.out_w, self.out_h)
            )
            if not self.writer.isOpened():
                raise RuntimeError(
                    f"Could not open {self.args.output} with fourcc "
                    f"'{self.args.fourcc}'"
                )
        for frame in self.buffer:
            self.writer.write(frame)
        self.frames_written += len(self.buffer)
        self.buffer.clear()
        print(f"\rWrote {self.frames_written} frames", end="", flush=True)

    def close(self) -> None:
        self.flush()
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.shm is not None:
            self.shm.close()
            self.shm = None
        print()
        if self.frames_written and self.first_stamp is not None:
            span = self.last_stamp - self.first_stamp
            measured = (self.frames_written - 1) / span if span > 0 else float("nan")
            print(f"Finished: {self.frames_written} frames, {span:.1f} s of sim time "
                  f"({measured:.1f} fps measured, written at {self.args.fps:g} fps)")
            print(f"Output: {self.args.output}")


def _yaw_list(text: str) -> Tuple[float, ...]:
    try:
        yaws = tuple(float(part) for part in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a comma-separated list of "
                                         f"numbers: {text!r}")
    if len(yaws) != 5:
        raise argparse.ArgumentTypeError(
            f"expected 5 yaws to match the 5 windows in shared memory, got "
            f"{len(yaws)}")
    return yaws


def _shutdown() -> None:
    """Shut down rclpy, tolerating its SIGINT handler having done it already."""
    try:
        if rclpy.ok():
            rclpy.shutdown()
    except RCLError:
        pass


def preview(args: argparse.Namespace) -> int:
    """Annotate a single live frame and save it as a PNG, for checking geometry."""
    rclpy.init()
    node = ActiveVisionRecorder(args)
    holder: Dict[str, np.ndarray] = {}

    def grab(msg: Image) -> None:
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        if msg.encoding == "rgb8":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame = frame.copy()
        window_idx, step = read_window_index(node.shm)
        node.annotate(frame, window_idx, step)
        holder["frame"] = frame

    node.destroy_subscription(node.sub)
    node.create_subscription(Image, args.topic, grab, 10)

    deadline = time.time() + 10.0
    while "frame" not in holder and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)

    if node.shm is not None:
        node.shm.close()
    node.destroy_node()
    _shutdown()

    if "frame" not in holder:
        print(f"No frame received on {args.topic} within 10 s")
        return 1
    cv2.imwrite(args.preview, holder["frame"])
    print(f"Wrote {args.preview}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="active_vision.mp4", help="output MP4 path")
    parser.add_argument("--topic", default="/camera/image_raw", help="camera topic")
    parser.add_argument("--shm-name", default=SHM_NAME, help="producer shared memory block")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="playback frame rate written into the file; the "
                             "camera publishes at 30 Hz, so this is real time")
    parser.add_argument("--scale", type=float, default=0.5,
                        help="resize factor applied before encoding")
    parser.add_argument("--flush-every", type=int, default=100,
                        help="frames buffered in memory between disk writes")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="stop after this many frames (0 = until Ctrl-C)")
    parser.add_argument("--fourcc", default="mp4v", help="OpenCV fourcc code")
    parser.add_argument("--show-all", action="store_true",
                        help="also outline the four unselected windows")
    parser.add_argument("--window-fov", type=float, default=PRODUCER_HFOV_WIN_DEG,
                        metavar="DEG",
                        help=f"window width used for the outlines; default "
                             f"{PRODUCER_HFOV_WIN_DEG:g} matches the producer")
    parser.add_argument("--window-yaws", type=_yaw_list,
                        default=PRODUCER_YAWS_DEG, metavar="D,D,...",
                        help="comma-separated window yaws; default "
                             + ",".join(f"{y:+.0f}" for y in PRODUCER_YAWS_DEG)
                             + ". Use with --window-fov to annotate footage "
                               "from a run with different settings, e.g. "
                               "--window-yaws -64,-32,0,32,64 --window-fov 60")
    parser.add_argument("--preview", metavar="PNG",
                        help="save one annotated frame to PNG and exit")
    args = parser.parse_args()

    if args.preview:
        return preview(args)

    rclpy.init()
    node = ActiveVisionRecorder(args)

    def on_signal(signum, frame) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_signal)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        _shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
