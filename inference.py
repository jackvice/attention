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
import time, nvtx, jax
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import struct
import jax.numpy as jnp
import cv2
import sys

import struct
# Import existing utilities
from trajectory_utils import (
    Pedestrian,
    create_target_heatmap_from_pedestrians,
    wait_for_depth_result,
    submit_depth_estimation,
    create_target_heatmap_from_pedestrians,
    detect_pedestrians_yolo_onnx,
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


def write_observation_to_shm(obs: np.ndarray, shm, step_count: int):
    struct.pack_into('<i', shm.buf, 0, step_count)
    shm.buf[4:4 + obs.nbytes] = obs.tobytes()

def read_camera_frame(shm: shared_memory.SharedMemory, 
                     img_index: int,
                     h: int = 320, w: int = 320) -> np.ndarray:
    """Read single image from shared memory with correct offset."""
    frame_size = h * w * 3
    offset = 8 + img_index * frame_size
    frame_bytes = bytes(shm.buf[offset:offset + frame_size])
    return np.frombuffer(frame_bytes, dtype=np.uint8).reshape(h, w, 3)


def read_rl_control(shm: shared_memory.SharedMemory) -> Tuple[int, int]:
    frame_size = 320 * 320 * 3
    action_offset = 8 + 6 * frame_size
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


def read_six_images_if_new(shm, last_ts):
    ts = struct.unpack_from('<d', shm.buf, 0)[0]
    if ts <= last_ts:
        return None, [], ts
    
    frame_size = 320 * 320 * 3
    center = np.frombuffer(shm.buf[8:8+frame_size], dtype=np.uint8).reshape(320,320,3)
    windows = [np.frombuffer(shm.buf[8+(i+1)*frame_size:8+(i+2)*frame_size], 
               dtype=np.uint8).reshape(320,320,3) for i in range(5)]
    
    return center / 255.0, [w / 255.0 for w in windows], ts


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


def fuse_edges_boxes_depth(
    rgb: np.ndarray,
    boxes: np.ndarray,
    depth: np.ndarray,
    depth_min: float = 0.0,
    depth_max: float = 1.0,
) -> np.ndarray:
    """
    Channel 0: Grayscale luminance
    Channel 1: Edge magnitude + pedestrian boxes
    Channel 2: Depth
    """
    h, w, _ = rgb.shape
    
    # Convert to grayscale
    if rgb.dtype == np.uint8:
        rgb_f = rgb.astype(np.float32) / 255.0
    else:
        rgb_f = rgb.astype(np.float32)
    
    Y = 0.299*rgb_f[:,:,0] + 0.587*rgb_f[:,:,1] + 0.114*rgb_f[:,:,2]
    
    # Compute edge magnitude using Sobel
    gray_uint8 = (Y * 255).astype(np.uint8)
    sobelx = cv2.Sobel(gray_uint8, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_uint8, cv2.CV_32F, 0, 1, ksize=3)
    edges = np.sqrt(sobelx**2 + sobely**2)
    edges = np.clip(edges / 255.0, 0, 1)  # Normalize to [0,1]
    
    # Overlay pedestrian boxes on edge channel
    for box in boxes:
        x_min, y_min, x_max, y_max, score = box[:5]
        x0 = max(0, min(w-1, int(x_min)))
        y0 = max(0, min(h-1, int(y_min)))
        x1 = max(0, min(w-1, int(x_max)))
        y1 = max(0, min(h-1, int(y_max)))
        
        if x1 <= x0 or y1 <= y0:
            continue
        
        edges[y0:y1, x0:x1] = 1.0
    
    # Depth
    D = depth.astype(np.float32)
    if D.ndim == 3:
        D = D[:,:,0]
    
    return np.stack([Y, edges, D], axis=-1).astype(np.float32)



def run_attention_pipeline_from_buffer(
    frame_buffer: deque,
    yolo_session: ort.InferenceSession,
    predict_fn: callable
) -> Tuple[Optional[np.ndarray], ort.InferenceSession]:
    """
    Run attention pipeline on buffered frames.
    
    Returns:
        heatmap: (320, 320) float32 in [0,1], or None if buffer not full
        yolo_session: Updated session (or same if no reload)
    """
    if len(frame_buffer) < BUFFER_SIZE:
        return None, yolo_session
    
    # Get temporal sequence from buffer
    indices = [0, 15, 30, 45, 59]
    frames = [frame_buffer[i][0] for i in indices]
    
    # Stack into batch: (1, 5, 320, 320, 3)
    rgb_batch = np.stack(frames)[np.newaxis]
    
    # Run YOLO on each frame to create masks
    from trajectory_utils import detect_pedestrians_yolo_onnx
    
    mask_batch = []
    for frame in frames:
        # detect_pedestrians_yolo_onnx expects [0,1] float
        pedestrians, yolo_session = detect_pedestrians_yolo_onnx(
            frame,
            session=yolo_session
        )
        
        # Create mask from pedestrian bboxes
        mask = np.zeros((320, 320, 1), dtype=np.float32)
        for ped in pedestrians:
            x1, y1, x2, y2 = ped.bbox
            mask[y1:y2, x1:x2] = 1.0
        
        mask_batch.append(mask)
    
    mask_batch = np.stack(mask_batch)[np.newaxis]  # (1, 5, 320, 320, 1)
    
    # Run attention model
    import jax.numpy as jnp
    rgb_jax = jnp.array(rgb_batch)
    mask_jax = jnp.array(mask_batch)
    
    heatmap_jax = predict_fn(rgb_jax, mask_jax)
    heatmap = np.array(heatmap_jax[0, :, :, 0])  # (320, 320)
    
    return heatmap, yolo_session

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
            window_idx, step_count = read_rl_control(shm)  # shm is camera_latest
            
            # Clamp index to [0, 4]
            if window_idx < 0 or window_idx > 4:
                window_idx = 0
            # image_2 is one of the 5 windows
            image_2_rgb = window_rgbs[window_idx]  # float32 [0,1], shape [H,W,3]

            # 4) Start depth estimation on image_2 (Pipeline 3)
            #print(f"DEBUG: Submitting depth estimation, session is None: {depth_session is None}")
            depth_future = submit_depth_estimation(depth_executor, image_2_rgb, depth_session)

            # 5) Run YOLO on image_2 (Pipeline 2)
            pedestrians_win, yolo_session = detect_pedestrians_yolo_onnx(
                image_2_rgb, session=yolo_session
            )
            
            # (We don't yet use pedestrians_win in the RL obs; it's ready for future fusion.)

            # 6) Wait for depth result, with cached fallback like before
            #print(f"DEBUG: Waiting for depth result...")
            depth_image, new_depth_session = wait_for_depth_result(
                depth_future, timeout_seconds=5.0  # Allow time for first model load
            )
            #print(f"DEBUG: Depth result - image is None: {depth_image is None}, session is None: {new_depth_session is None}")
            if new_depth_session is not None:
                depth_session = new_depth_session
                #print(f"DEBUG: Updated depth_session cache")

            if depth_image is None:
                if last_depth_image is not None:
                    depth_image = last_depth_image
                    #print(f"DEBUG: Using cached depth")
                else:
                    depth_image = np.zeros(image_2_rgb.shape[:2], dtype=np.float32)
                    #print(f"DEBUG: Using zero depth fallback")
            else:
                last_depth_image = depth_image
                #print(f"DEBUG: Got new depth, min={np.min(depth_image):.3f}, max={np.max(depth_image):.3f}")

            # 7) Create fused observation (still 96x96x3 via existing JAX helper)
            try:

                heatmap_jax = jnp.array(heatmap)

                # Resize image_2_rgb to RL resolution first
                image_2_resized = cv2.resize(
                    (image_2_rgb * 255).astype(np.uint8),
                    (rl_obs_width, rl_obs_height),
                    interpolation=cv2.INTER_AREA
                )
                
                # Resize depth to match
                depth_resized = cv2.resize(
                    depth_image,
                    (rl_obs_width, rl_obs_height),
                    interpolation=cv2.INTER_LINEAR
                )
                
                # Convert pedestrians to boxes array and scale to resized coordinates
                scale_x = rl_obs_width / 320
                scale_y = rl_obs_height / 320
                boxes = np.array([
                    [p.bbox[0]*scale_x, p.bbox[1]*scale_y, 
                     p.bbox[2]*scale_x, p.bbox[3]*scale_y, p.confidence]
                    for p in pedestrians_win
                ]) if pedestrians_win else np.zeros((0, 5))

                # Just before fuse_gray_alpha_depth call (around line 687)
                #print(f"DEBUG: depth_resized shape={depth_resized.shape}, min={np.min(depth_resized):.3f}, max={np.max(depth_resized):.3f}")
                #print(f"DEBUG: image_2_resized shape={image_2_resized.shape}")
                #print(f"DEBUG: boxes shape={boxes.shape}, count={len(pedestrians_win)}")

                
                # 1) Fused 3-channel image: gray + alpha (YOLO boxes) + depth
                fused_img = fuse_edges_boxes_depth(
                    rgb=image_2_resized,
                    boxes=boxes,
                    depth=depth_resized,
                    depth_min=0.0,
                    depth_max=1.0,  # ADD THIS - depth is already normalized
                )  # shape (96, 96, 3)

                #print(f"DEBUG: fused_img shape={fused_img.shape}")
                #print(f"DEBUG: fused_img channel 0 (gray) min={np.min(fused_img[:,:,0]):.3f}, max={np.max(fused_img[:,:,0]):.3f}")
                #print(f"DEBUG: fused_img channel 1 (alpha) min={np.min(fused_img[:,:,1]):.3f}, max={np.max(fused_img[:,:,1]):.3f}")
                #print(f"DEBUG: fused_img channel 2 (depth) min={np.min(fused_img[:,:,2]):.3f}, max={np.max(fused_img[:,:,2]):.3f}")
                
                # 2) Resize heatmap to RL resolution; ensure it is (H, W, 1)
                heatmap_resized = cv2.resize(
                    heatmap, (rl_obs_width, rl_obs_height), interpolation=cv2.INTER_LINEAR
                )
                if heatmap_resized.ndim == 2:
                    heatmap_resized = heatmap_resized[..., None]

                # 3) Stack into 4-channel observation: [heatmap, gray, alpha, depth]
                rl_obs = np.concatenate([heatmap_resized, fused_img], axis=-1)  # (H, W, 4)
                rl_obs = rl_obs.astype(np.float32)

                write_observation_to_shm(rl_obs, rl_obs_shm, step_count)


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
