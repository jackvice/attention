#!/usr/bin/env python3
# inference.py
"""
Process frames from shared memory buffer with YOLO to detect pedestrians
and prepare data for pedestrian trajectory prediction.
"""
import numpy as np
import time
import os
from multiprocessing import shared_memory
import argparse
from typing import List, Tuple, Optional, Dict, Any, NamedTuple
import onnxruntime as ort
import time, torch, nvtx, jax
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import struct
import jax.numpy as jnp

import struct
# Import existing utilities
from trajectory_utils import (
    Pedestrian,
    create_target_heatmap_from_pedestrians,
    detect_pedestrians_yolo_onnx,
    create_masks_from_pedestrians,
    create_fused_observation_jax,
    write_observation_to_shm,
    #estimate_depth_pytorch,
    save_debug_observation,
    wait_for_depth_result,
    submit_depth_estimation,
    create_target_heatmap_from_pedestrians,
)



# Shared memory names - must match producer
SHM_NAME = "camera_latest"



# Configuration - must match new single-slot approach
H, W = 320, 320
BUFFER_SIZE = 60
SAMPLE_OFFSETS = [0, 15, 30, 45, 59]
NUM_IMAGES = 6                       # center + 5 windows
FRAME_BYTES = H * W * 3              # uint8 RGB
SHM_ACTIVE_CTRL = "active_window_ctrl"  # control block from Dreamer


# Default YOLO model path
DEFAULT_YOLO_PATH = "/home/jack/src/attention/models/yolo11n.onnx"


def read_camera_frame(shm: shared_memory.SharedMemory, 
                     img_index: int,
                     h: int = 320, w: int = 320) -> np.ndarray:
    """Read single image from shared memory with correct offset."""
    frame_size = h * w * 3
    offset = 8 + img_index * frame_size
    frame_bytes = bytes(shm.buf[offset:offset + frame_size])
    return np.frombuffer(frame_bytes, dtype=np.uint8).reshape(h, w, 3)


def read_rl_control(shm: shared_memory.SharedMemory, 
                    num_images: int = 6) -> Tuple[int, int]:
    """Read active_vision_action and step_count from shared memory."""
    frame_size = 320 * 320 * 3
    action_offset = 8 + num_images * frame_size
    action = struct.unpack_from('<i', shm.buf, action_offset)[0]
    step = struct.unpack_from('<i', shm.buf, action_offset + 4)[0]
    return action, step


def write_rl_observation(shm: shared_memory.SharedMemory,
                        attention: np.ndarray,
                        fused: np.ndarray,
                        step_count: int) -> None:
    """Write 96x96x4 observation + step_count to rl_observation shared memory."""
    output = np.concatenate([attention, fused], axis=-1)  # 96x96x4
    struct.pack_into('<i', shm.buf, 0, step_count)
    shm.buf[4:4 + output.nbytes] = output.tobytes()

def read_six_images_if_new(
    shm: shared_memory.SharedMemory,
    last_timestamp: float,
    num_images: int = NUM_IMAGES,
    height: int = H,
    width: int = W,
) -> Tuple[Optional[np.ndarray], Optional[List[np.ndarray]], float]:
    """
    Read all 6 images (center + 5 windows) from shared memory
    if the timestamp is newer than last_timestamp.

    Layout in SHM:
        [0:8]   -> float64 timestamp (seconds)
        [8:...] -> num_images * (H*W*3) bytes of uint8 RGB images.

    Returns:
        (center_rgb_float, window_rgb_list_float, current_timestamp)
        or (None, None, current_timestamp) if no new frame.
    """
    # Read timestamp
    current_timestamp = struct.unpack_from("<d", shm.buf, 0)[0]
    if current_timestamp <= last_timestamp:
        return None, None, current_timestamp

    # Read all image bytes as a flat view
    total_frame_bytes = num_images * FRAME_BYTES
    raw = memoryview(shm.buf)[8 : 8 + total_frame_bytes]
    arr = np.frombuffer(raw, dtype=np.uint8)

    # Reshape to [num_images, H, W, 3]
    arr = arr.reshape(num_images, height, width, 3)

    # Convert to float32 [0,1]
    arr_f = arr.astype(np.float32) / 255.0

    # image_0 is center; image_1..5 are windows
    center_rgb = arr_f[0]
    window_rgbs = [arr_f[i] for i in range(1, num_images)]

    return center_rgb, window_rgbs, current_timestamp



def read_active_window_ctrl(
    shm_ctrl: Optional[shared_memory.SharedMemory],
    default_idx: int = 0,
    default_step: int = 0,
) -> Tuple[int, int]:
    """
    Read (window_idx, step_or_ts) from the control SHM.

    Layout:
        [0:4]  -> int32 window_idx (0..4)
        [4:8]  -> int32 version    (unused here but reserved)
        [8:16] -> int64 step_or_ts

    If shm_ctrl is None, returns (default_idx, default_step).
    """
    if shm_ctrl is None:
        return default_idx, default_step

    buf = shm_ctrl.buf

    window_idx = struct.unpack_from("<i", buf, 0)[0]
    step_or_ts = struct.unpack_from("<q", buf, 8)[0]

    return int(window_idx), int(step_or_ts)


def attach_active_window_memory(
    shm_name: str = SHM_ACTIVE_CTRL,
) -> shared_memory.SharedMemory:
    """
    Attach to the small control shared memory block that carries:
        int32 window_idx  (0..4)
        int32 version     (monotonic counter or just write count)
        int64 step_or_ts  (Dreamer step or sim timestamp)
    """
    try:
        return shared_memory.SharedMemory(name=shm_name, track=False)
    except FileNotFoundError:
        print(f"Warning: control shared memory '{shm_name}' not found.")
        print("Defaulting to window_idx=0, step=0.")
        return None  # caller must handle None



def attach_single_frame_memory(shm_name: str = SHM_NAME) -> shared_memory.SharedMemory:
    """
    Attach to single-slot shared memory.
    
    Returns:
        SharedMemory object
        
    Raises:
        SystemExit: If shared memory not found
    """
    try:
        return shared_memory.SharedMemory(name=shm_name, track=False)
    except FileNotFoundError:
        print(f"Error: Shared memory '{shm_name}' not found. Is producer running?")
        sys.exit(1)



def sample_frames_from_buffer(
    frame_buffer: deque,
    target_time_span: float = 2.0,
    num_samples: int = 5
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Sample frames covering approximately target_time_span seconds of simulation time.
    
    Args:
        frame_buffer: Buffer of (frame, timestamp) tuples
        target_time_span: Target time span in seconds (default: 2.0)
        num_samples: Number of frames to sample (default: 5)
        
    Returns:
        Tuple of (sampled_frames, sampled_timestamps) or None if insufficient data
    """
    if len(frame_buffer) < num_samples:
        return None
    
    # Convert deque to list for easier indexing
    buffer_list = list(frame_buffer)
    
    # Get newest timestamp (first in buffer since we use appendleft)
    newest_time = buffer_list[0][1]
    
    # Find the oldest frame within target_time_span
    target_oldest_time = newest_time - target_time_span
    
    # Find the furthest back index that's still within our time window
    max_index = 0
    for i, (frame, timestamp) in enumerate(buffer_list):
        if timestamp >= target_oldest_time:
            max_index = i
        else:
            break  # Timestamps get older as we go further in the buffer
    
    # If we don't have enough time span, use what we have
    if max_index < num_samples - 1:
        max_index = min(len(buffer_list) - 1, BUFFER_SIZE - 1)
    
    # Sample frames evenly across the available range
    if max_index == 0:
        # Only one frame available, duplicate it
        indices = [0] * num_samples
    else:
        # Create evenly spaced indices
        indices = [int(i * max_index / (num_samples - 1)) for i in range(num_samples)]
    
    sampled_frames = []
    sampled_timestamps = []
    
    for idx in indices:
        frame, timestamp = buffer_list[idx]
        sampled_frames.append(frame)
        sampled_timestamps.append(timestamp)
    
    return np.array(sampled_frames), np.array(sampled_timestamps)



def load_attention_model(
    checkpoint_path: str,
) -> Tuple[callable, Dict[str, Any]]:
    """
    Load the attention model from a checkpoint file.
    
    Args:
        checkpoint_path: Path to the model checkpoint file
        
    Returns:
        Tuple of (prediction_function, model_state)
    """
    import pickle
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    import jax
    import flax.linen as nn
    from trajectory_model import SpatiotemporalAttention, ModelConfig
    
    # Load checkpoint file
    print(f"Loading attention model from {checkpoint_path}")
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    # Extract model parameters and config
    params = checkpoint['params']
    config_dict = checkpoint.get('config', {})
    
    # Create model config
    if isinstance(config_dict, dict):
        config = ModelConfig(**config_dict)
    else:
        # Assume it's already a ModelConfig object
        config = config_dict
    
    # Create model instance
    model = SpatiotemporalAttention(config=config)
    
    # Create optimized prediction function
    @jax.jit
    def predict_fn(rgb_frames, mask_frames):
        # Add batch dimension if not present
        if rgb_frames.ndim == 4:
            rgb_frames = rgb_frames[None, ...]  # [1, T, H, W, 3]
        if mask_frames.ndim == 4:
            mask_frames = mask_frames[None, ...]  # [1, T, H, W, 1]
            
        # Run model in inference mode
        return model.apply({'params': params}, rgb_frames, mask_frames, training=False)
    
    return predict_fn, {'model': model, 'params': params, 'config': config}


class ProcessedBatch(NamedTuple):
    """Processed frames with detections, ready for trajectory prediction."""
    rgb_frames: np.ndarray  # [T, H, W, 3]
    mask_frames: np.ndarray  # [T, H, W, 1]
    timestamps: np.ndarray  # [T]


    
def process_with_attention(
    batch: ProcessedBatch,
    all_pedestrians: List[List[Pedestrian]],
    predict_fn: callable
) -> np.ndarray:
    """
    Process batch with attention model - only predict if last frame has pedestrians.
    
    Args:
        batch: ProcessedBatch containing RGB and mask frames
        all_pedestrians: Pedestrian detections for each frame in sequence
        predict_fn: Jitted prediction function from loaded model
        
    Returns:
        Combined heatmap [H, W, 1] with current positions + predicted trajectories
    """
    import jax.numpy as jnp
    from config_temporal import SIGMA_PX
    
    # Get spatial dimensions
    h, w = batch.rgb_frames.shape[1:3]
    
    # Check if last frame has pedestrians
    if not all_pedestrians or len(all_pedestrians[0]) == 0:
        return np.zeros((h, w, 1), dtype=np.float32)
    
    # Get predicted heatmap
    rgb_frames = jnp.array(batch.rgb_frames)
    mask_frames = jnp.array(batch.mask_frames)
    predictions = predict_fn(rgb_frames, mask_frames)
    predicted_heatmap = np.array(predictions[0])
    
    # Get current positions heatmap using Gaussian blobs
    current_yolo_heatmap = create_target_heatmap_from_pedestrians(
        all_pedestrians[0], h, w, sigma=SIGMA_PX
    )
    
    # Combine: emphasize current YOLO detections over attention predictions
    combined_heatmap = 0.5 * predicted_heatmap + 0.7 * current_yolo_heatmap
    
    # Normalize to [0,1] range
    combined_heatmap = np.clip(combined_heatmap, 0.0, 1.0)
    
    #return np.zeros((h, w, 1), dtype=np.float32)  # no heatmap
    return combined_heatmap # with heatmap




import numpy as np
from typing import Tuple

def fuse_gray_alpha_depth(
    rgb: np.ndarray,
    boxes: np.ndarray,
    depth: np.ndarray,
    depth_min: float = 0.0,
    depth_max: float = 10.0,
) -> np.ndarray:
    """
    Convert RGB to grayscale + alpha (YOLO boxes) and fuse with depth
    into a 3-channel float32 image.

    Args:
        rgb:   H x W x 3 uint8 or float32 image (RGB).
        boxes: N x 5 array of YOLO detections in pixel coords:
               [x_min, y_min, x_max, y_max, score] per row.
               Coordinates must be in the same H,W frame as `rgb`.
        depth: H x W or H x W x 1 float32 depth image (meters or arbitrary units).
        depth_min: Minimum depth value for clipping (default 0.0).
        depth_max: Maximum depth value for clipping/normalization (default 10.0).

    Returns:
        fused: H x W x 3 float32 array in [0,1], where:
               channel 0: grayscale luminance
               channel 1: alpha / person-confidence mask
               channel 2: normalized depth
    """
    # --- Basic shape checks ---
    assert rgb.ndim == 3 and rgb.shape[2] == 3, "rgb must be HxWx3"
    h, w, _ = rgb.shape

    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[..., 0]
    assert depth.shape == (h, w), "depth must match rgb spatial size"

    if boxes.size == 0:
        boxes = boxes.reshape(0, 5)
    assert boxes.ndim == 2 and boxes.shape[1] >= 5, "boxes must be Nx5+"

    # --- Convert RGB to grayscale luminance in [0,1] ---
    if rgb.dtype == np.uint8:
        rgb_f = rgb.astype(np.float32) / 255.0
    else:
        rgb_f = rgb.astype(np.float32)

    # Standard luminance transform
    L = 0.299 * rgb_f[..., 0] + 0.587 * rgb_f[..., 1] + 0.114 * rgb_f[..., 2]  # HxW

    # --- Initialize alpha mask ---
    A = np.zeros((h, w), dtype=np.float32)

    # --- Compute border thickness based on resolution ---
    # Aim for ≥1px after downsampling to 96x96
    scale = max(h, w) / 96.0

    t = max(4, int(round(max(H,W) / 48))) # border thickness pixels to survive downsampling
    
    # --- Overlay boxes: interior alpha + thick border ---
    for box in boxes:
        x_min, y_min, x_max, y_max, score = box[:5]

        # Clamp to valid integer pixel indices
        x0 = max(0, min(w - 1, int(np.floor(x_min))))
        y0 = max(0, min(h - 1, int(np.floor(y_min))))
        x1 = max(0, min(w - 1, int(np.ceil(x_max))))
        y1 = max(0, min(h - 1, int(np.ceil(y_max))))

        if x1 <= x0 or y1 <= y0:
            continue  # degenerate box

        # Interior region (may be empty if box is very thin)
        xi0 = x0 + t
        yi0 = y0 + t
        xi1 = x1 - t
        yi1 = y1 - t

        # Interior: encode confidence in alpha, keep grayscale as-is
        if xi1 > xi0 and yi1 > yi0:
            A[yi0:yi1, xi0:xi1] = np.maximum(A[yi0:yi1, xi0:xi1], float(score))

        # Border region: overwrite to ensure visibility after downsampling
        # Border brightness: white (1.0) for maximum contrast.
        # Alpha on border: 1.0 to make it a strong signal.
        # Top border
        yt0, yt1 = y0, min(y0 + t, y1)
        L[yt0:yt1, x0:x1] = 1.0
        A[yt0:yt1, x0:x1] = 1.0

        # Bottom border
        yb0, yb1 = max(y1 - t, y0), y1
        L[yb0:yb1, x0:x1] = 1.0
        A[yb0:yb1, x0:x1] = 1.0

        # Left border
        xl0, xl1 = x0, min(x0 + t, x1)
        L[y0:y1, xl0:xl1] = 1.0
        A[y0:y1, xl0:xl1] = 1.0

        # Right border
        xr0, xr1 = max(x1 - t, x0), x1
        L[y0:y1, xr0:xr1] = 1.0
        A[y0:y1, xr0:xr1] = 1.0

    # --- Normalize depth to [0,1] ---
    depth_f = depth.astype(np.float32)
    # Replace NaNs/Infs with far depth
    invalid = ~np.isfinite(depth_f)
    if np.any(invalid):
        depth_f[invalid] = depth_max

    depth_f = np.clip(depth_f, depth_min, depth_max)
    if depth_max > depth_min:
        D_norm = (depth_f - depth_min) / (depth_max - depth_min)
    else:
        # avoid divide-by-zero; fall back to zeros
        D_norm = np.zeros_like(depth_f, dtype=np.float32)

    # --- Stack into fused 3-channel output ---
    fused = np.stack([L, A, D_norm], axis=-1).astype(np.float32)  # HxWx3

    return fused




def main():
    """Main function with multi-image shared memory and active window control."""
    parser = argparse.ArgumentParser(description="Process frames with YOLO and predict trajectories")
    parser.add_argument(
        "--yolo_model",
        type=str,
        default=DEFAULT_YOLO_PATH,
        help=f"Path to YOLO ONNX model (default: {DEFAULT_YOLO_PATH})",
    )
    parser.add_argument(
        "--attention_model",
        type=str,
        required=True,
        help="Path to attention model checkpoint file",
    )
    parser.add_argument(
        "--rl-obs-name",
        type=str,
        default="rl_observation",
        help="Name of RL observation shared memory (default: rl_observation)",
    )

    args = parser.parse_args()

    print("Trajectory prediction pipeline starting...")
    print(f"Using YOLO model: {args.yolo_model}")
    print(f"Using attention model: {args.attention_model}")
    print(f"Sampling frames at offsets: {SAMPLE_OFFSETS}")

    # RL observation parameters (unchanged: 96x96x3)
    rl_obs_height, rl_obs_width = 96, 96
    rl_obs_channels = 4

    shm = None
    shm_ctrl = None
    rl_obs_shm = None
    depth_executor = None

    try:
        # Load YOLO model
        yolo_session = ort.InferenceSession(
            args.yolo_model,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        print("Successfully loaded YOLO model")

        # Load attention model
        predict_fn, model_info = load_attention_model(args.attention_model)
        print("Successfully loaded attention model")

        # Attach to camera shared memory (6 images)
        shm = attach_single_frame_memory(SHM_NAME)
        print("Successfully attached to camera shared memory")

        # Attach to active-window control shared memory
        shm_ctrl = attach_active_window_memory()
        if shm_ctrl is None:
            print(
                f"Warning: control SHM '{SHM_ACTIVE_CTRL}' not found; "
                "defaulting to window_idx=0, step=0."
            )
        else:
            print(f"Successfully attached to control SHM '{SHM_ACTIVE_CTRL}'")

        # Setup RL observation shared memory
        try:
            shared_memory.SharedMemory(name=args.rl_obs_name).unlink()
        except FileNotFoundError:
            pass

        header_size = 8 + 4  # timestamp + valid flag
        obs_data_size = rl_obs_height * rl_obs_width * rl_obs_channels * 4  # float32
        rl_obs_shm_size = header_size + obs_data_size

        rl_obs_shm = shared_memory.SharedMemory(
            create=True,
            size=rl_obs_shm_size,
            name=args.rl_obs_name,
        )
        print(f"Created RL observation SHM '{args.rl_obs_name}'")

        # Create thread pool for depth estimation
        depth_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="depth_worker"
        )
        print("Created ThreadPoolExecutor for depth estimation")

        # Initialize state
        frame_buffer: deque = deque(maxlen=BUFFER_SIZE)  # center-view temporal buffer
        last_timestamp = 0.0
        depth_session = None
        last_depth_image: Optional[np.ndarray] = None
        loop_idx = 0

        print("Waiting for frames...")


        last_step = -1
        # Main processing loop
        while True:
            loop_idx += 1

            # 1) Wait for a new 6-image frame
            while True:
                center_rgb, window_rgbs, current_timestamp = read_six_images_if_new(
                    shm, last_timestamp
                )
                if center_rgb is not None:
                    break
                time.sleep(0.0001)

            # center_rgb, window_rgbs are float32 [0,1]
            frame_buffer.appendleft((center_rgb, current_timestamp))
            last_timestamp = current_timestamp

            # 2) Run attention pipeline (Pipeline 1) on center-view buffer
            heatmap, yolo_session = run_attention_pipeline_from_buffer(
                frame_buffer, yolo_session, predict_fn
            )
            if heatmap is None:
                # Buffer not full yet / not enough temporal span
                if loop_idx % 100 == 0:
                    print(f"Buffer filling: {len(frame_buffer)}/{BUFFER_SIZE}")
                continue

            # 3) Read active window selection (for image_2)
            window_idx, step_or_ts = read_active_window_ctrl(shm_ctrl, default_idx=0, default_step=0)
            # Clamp index to [0, 4]
            if window_idx < 0 or window_idx > 4:
                window_idx = 0
            # image_2 is one of the 5 windows
            image_2_rgb = window_rgbs[window_idx]  # float32 [0,1], shape [H,W,3]

            # 4) Start depth estimation on image_2 (Pipeline 3)
            depth_future = submit_depth_estimation(depth_executor, image_2_rgb, depth_session)

            # 5) Run YOLO on image_2 (Pipeline 2)
            pedestrians_win, yolo_session = run_yolo_on_window(
                image_2_rgb, yolo_session
            )
            # (We don't yet use pedestrians_win in the RL obs; it's ready for future fusion.)

            # 6) Wait for depth result, with cached fallback like before
            depth_image, new_depth_session = wait_for_depth_result(
                depth_future, timeout_seconds=0.1
            )
            if new_depth_session is not None:
                depth_session = new_depth_session

            if depth_image is None:
                if last_depth_image is not None:
                    depth_image = last_depth_image
                else:
                    depth_image = np.zeros(image_2_rgb.shape[:2], dtype=np.float32)
            else:
                last_depth_image = depth_image

            # 7) Create fused observation (still 96x96x3 via existing JAX helper)
            try:
                heatmap_jax = jnp.array(heatmap)

                # 1) Fused 3-channel image: gray + alpha (YOLO boxes) + depth
                fused_img = fuse_gray_alpha_depth(
                    rgb=image_2_rgb,
                    boxes=pedestrians_win,
                    depth=depth_image,
                    target_height=rl_obs_height,
                    target_width=rl_obs_width,
                )  # shape (H, W, 3)

                # 2) Resize heatmap to RL resolution; ensure it is (H, W, 1)
                heatmap_resized = cv2.resize(
                    heatmap, (rl_obs_width, rl_obs_height), interpolation=cv2.INTER_LINEAR
                )
                if heatmap_resized.ndim == 2:
                    heatmap_resized = heatmap_resized[..., None]

                # 3) Stack into 4-channel observation: [heatmap, gray, alpha, depth]
                rl_obs = np.concatenate([heatmap_resized, fused_img], axis=-1)  # (H, W, 4)
                rl_obs = rl_obs.astype(np.float32)

                write_observation_to_shm(rl_obs, rl_obs_shm)


            except Exception as e:
                print(f"Error creating RL observation: {e}")

    except KeyboardInterrupt:
        print("Interrupted by user. Shutting down.")
    except Exception as e:
        print(f"Fatal error in main(): {e}")
    finally:
        print("Cleaning up resources...")
        try:
            if depth_executor is not None:
                depth_executor.shutdown(wait=True)
        except Exception:
            pass
        try:
            if shm is not None:
                shm.close()
        except Exception:
            pass
        try:
            if shm_ctrl is not None:
                shm_ctrl.close()
        except Exception:
            pass
        try:
            if rl_obs_shm is not None:
                rl_obs_shm.close()
                rl_obs_shm.unlink()
        except Exception:
            pass
        

if __name__ == "__main__":
    main()
