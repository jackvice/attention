
#!/usr/bin/env python3
"""
ROS2 node for writing camera frames to a single shared memory slot.
Consumer (inference.py) reads a 320x320 RGB frame and treats it as
a ~60° pinhole view: we take the center 600x600 crop from a 1600x600
camera image, then downsample to 320x320.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from multiprocessing import shared_memory
import numpy as np
import cv2
import struct
import atexit
import signal
import sys
from typing import Optional
import numpy.typing as npt
from typing import List, Sequence, Mapping, Tuple



# Configuration: inference expects 320x320 RGB
H, W = 320, 320
NUM_IMAGES = 6
SHM_NAME = "camera_latest"

# Global for cleanup
g_shm: Optional[shared_memory.SharedMemory] = None

ImageRGB = npt.NDArray[np.uint8]


def make_pinhole_to_rect_lut(
    src_w: int,
    src_h: int,
    dst_hw: int,
    hfov_dst_rad: float,
    yaw_deg: float,
    hfov_src_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map from a virtual pinhole camera (dst_hw x dst_hw, HFOV = hfov_dst_rad,
    yaw-rotated by yaw_deg) into the *real* pinhole camera image
    (src_w x src_h, HFOV = hfov_src_rad).

    Returns:
        map_x, map_y: float32 arrays of shape (dst_hw, dst_hw) with source coordinates.
    """
    # 1) Pixel grid in the *output* (virtual) image, normalized to [-1, 1]
    xx, yy = np.meshgrid(
        np.linspace(-1.0, 1.0, dst_hw),
        np.linspace(-1.0, 1.0, dst_hw),
    )

    # 2) Convert to directions in the *virtual* camera frame
    #    HFOV_dst: x spans [-tan(FOV/2), +tan(FOV/2)] at z=1
    t_dst = np.tan(hfov_dst_rad / 2.0)
    x = xx * t_dst
    y = yy * t_dst
    z = np.ones_like(x)

    # Normalize to unit sphere
    n = np.sqrt(x * x + y * y + z * z)
    x /= n
    y /= n
    z /= n

    # 3) Rotate these rays by yaw around the vertical axis
    yaw = np.deg2rad(yaw_deg)
    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)

    xr = cos_y * x + sin_y * z
    yr = y
    zr = -sin_y * x + cos_y * z

    # 4) Project into the *real* pinhole camera with HFOV_src
    #    f = (W/2) / tan(HFOV_src/2)
    fx = (src_w / 2.0) / np.tan(hfov_src_rad / 2.0)
    fy = fx  # assume square pixels, no skew

    cx = src_w / 2.0
    cy = src_h / 2.0

    # Avoid division by zero
    zr_safe = np.where(zr == 0.0, 1e-6, zr)

    u = fx * (xr / zr_safe) + cx
    v = fy * (yr / zr_safe) + cy

    return u.astype(np.float32), v.astype(np.float32)


def build_lut_bank(
    src_w: int,
    src_h: int,
    yaws_deg: tuple[float, ...],
    hfov_win_deg: float = 60.0,
    dst_hw: int = 96,
    hfov_src_rad: float = 2.8,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    Build LUTs mapping each output window (60° virtual pinhole view at a given yaw)
    into the real pinhole camera image (src_w x src_h, HFOV = hfov_src_rad).
    """
    hfov_dst_rad = np.deg2rad(hfov_win_deg)

    bank: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for idx, yaw in enumerate(yaws_deg):
        mx, my = make_pinhole_to_rect_lut(
            src_w=src_w,
            src_h=src_h,
            dst_hw=dst_hw,
            hfov_dst_rad=hfov_dst_rad,
            yaw_deg=yaw,
            hfov_src_rad=hfov_src_rad,
        )
        bank[idx] = (mx, my)
    return bank


def cleanup_shared_memory() -> None:
    """Clean up shared memory on exit."""
    global g_shm
    try:
        if g_shm is not None:
            g_shm.close()
            g_shm.unlink()
            print("Unlinked shared memory")
    except Exception as e:
        print(f"Error cleaning up: {e}")


atexit.register(cleanup_shared_memory)


def signal_handler(sig, frame) -> None:
    cleanup_shared_memory()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def crop_center_square(image: np.ndarray) -> np.ndarray:
    """
    From a 600x1600 RGB image, extract the central 600x600 square:

    - Input:  image shape [H_in, W_in, 3], e.g. [600, 1600, 3]
    - Output: image[:, 500:1100, :] -> [600, 600, 3]

    This approximates reducing HFOV from ~160° to ~60° by keeping
    the central portion of a rectilinear camera.
    """
    h, w, _ = image.shape
    # Ensure height <= width so we can take a central square of size h
    side = min(h, w)
    start_x = (w - side) // 2
    end_x = start_x + side
    return image[:, start_x:end_x, :]


class CameraSingleSlot(Node):
    """ROS2 node that writes latest camera frame to single shared memory slot."""

    def __init__(self, camera_topic: str = "/camera/image_raw") -> None:
        super().__init__("camera_single_slot")


        self.bridge = CvBridge()

        # Parameters for LUT-based rectified windows
        self.crop_top_px = 0
        src_width = 1600
        src_height_full = 600

        # Precompute LUTs once: 5 yaw bins across ~160°
        self.out_hw_lut = 320
        try:
            yaws_deg = (-64.0, -32.0, 0.0, 32.0, 64.0)
            self._lut_bank = build_lut_bank(
                src_w=src_width,
                src_h=src_height_full,
                yaws_deg=yaws_deg,
                hfov_win_deg=60.0,
                dst_hw=self.out_hw_lut,
            )
            self.get_logger().info("Precomputed LUTs for 5 rectified windows.")
        except Exception as e:
            self.get_logger().warning(
                f"Failed to precompute LUTs, using center view only: {e}"
            )
            self._lut_bank = {}

        
        # Clean up any existing shared memory with same name
        try:
            shared_memory.SharedMemory(name=SHM_NAME).unlink()
        except FileNotFoundError:
            pass
        except Exception as e:
            self.get_logger().warning(f"Error cleaning existing memory: {e}")

        # Create shared memory
        frame_size = H * W * 3  # uint8 RGB
        total_size = 8 + frame_size * NUM_IMAGES + 8 # timestamp + images + action/step

        self.shm = shared_memory.SharedMemory(create=True, size=total_size, name=SHM_NAME)
        global g_shm
        g_shm = self.shm
        self.frame_size = frame_size

        # Initialize control fields for RL agent
        action_offset = 8 + NUM_IMAGES * frame_size
        struct.pack_into('<i', self.shm.buf, action_offset, 0)      # action = 0
        struct.pack_into('<i', self.shm.buf, action_offset + 4, 0)  # step_count = 0
        
        # ROS2 subscriber
        self.sub = self.create_subscription(
            Image,
            camera_topic,
            self.camera_callback,
            10,
        )

        self.get_logger().info(
            f"Camera single slot ready on {camera_topic}, SHM={SHM_NAME}"
        )


    def camera_callback(self, msg: Image) -> None:
        """Convert, crop, generate rectified windows, and write frames to shared memory."""
        try:
            # 1. ROS Image -> RGB NumPy (H_in, W_in, 3) = (600,1600,3)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

            # Image 0: center 600x600 -> 320x320 (60° central view)
            center_square = crop_center_square(cv_image)
            center_320 = cv2.resize(
                center_square,
                (W, H),
                interpolation=cv2.INTER_AREA,
            )

            windows: List[ImageRGB] = []
            if self._lut_bank:
                for idx in range(5):
                    map_x, map_y = self._lut_bank[idx]
                    rect = cv2.remap(
                        cv_image,
                        map_x,
                        map_y,
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                    )
                    # LUT grid is 320x320, so rect should be 320x320
                    assert rect.shape == (H, W, 3), f"Rectified window {idx} has shape {rect.shape}"
                    windows.append(rect)
            else:
                # Fallback: just reuse the center view for all 5 windows
                windows = [center_320] * 5

            # Collect all 6 images
            images = [center_320] + list(windows)
            assert len(images) == NUM_IMAGES

            # Write timestamp (double, little-endian)
            timestamp = self.get_clock().now().nanoseconds / 1e9
            struct.pack_into("<d", self.shm.buf, 0, timestamp)

            # Write all 6 images sequentially
            for idx, img in enumerate(images):
                if img.dtype != np.uint8:
                    img = np.clip(img, 0, 255).astype(np.uint8)
                offset = 8 + idx * self.frame_size
                self.shm.buf[offset : offset + self.frame_size] = img.tobytes()

        except Exception as e:
            self.get_logger().error(f"Error writing frame: {e}")


    def destroy_node(self) -> None:
        """Clean up on destruction."""
        try:
            if hasattr(self, "shm"):
                self.shm.close()
                self.shm.unlink()
            global g_shm
            g_shm = None
        except Exception as e:
            self.get_logger().error(f"Error in cleanup: {e}")
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CameraSingleSlot()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
