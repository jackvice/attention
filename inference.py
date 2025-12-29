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
from typing import Tuple, Optional, Dict, Any
import onnxruntime as ort
import time, jax
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import struct
import jax.numpy as jnp
import cv2
import sys
import pickle
from trajectory_model import SpatiotemporalAttention, ModelConfig
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
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    
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


# inference.py
def fuse_edges_boxes_depth(
    rgb: np.ndarray,
    boxes: np.ndarray,
    depth: np.ndarray,
    depth_min: float = 0.0,
    depth_max: float = 1.0,
) -> np.ndarray:
    """
    Channel 0: Grayscale luminance
    Channel 1: Edge magnitude + Super Thick White Bounding Box Borders
    Channel 2: Depth
    """
    h, w, _ = rgb.shape
    
    # --- 1. Define rgb_f ---
    if rgb.dtype == np.uint8:
        rgb_f = rgb.astype(np.float32) / 255.0
    else:
        rgb_f = rgb.astype(np.float32)
    
    # --- 2. Calculate Grayscale (Y) ---
    Y = 0.299*rgb_f[:,:,0] + 0.587*rgb_f[:,:,1] + 0.114*rgb_f[:,:,2]
    
    # --- 3. Compute Edges ---
    gray_uint8 = (Y * 255).astype(np.uint8)
    sobelx = cv2.Sobel(gray_uint8, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_uint8, cv2.CV_32F, 0, 1, ksize=3)
    edges = np.sqrt(sobelx**2 + sobely**2)
    edges = np.clip(edges / 255.0, 0, 1)  # Normalize to [0,1]
    
    # --- 4. Overlay Super Thick WHITE Borders ---
    # Thickness in pixels (relative to the 96x96 RL image)
    # 3px is very thick for this resolution, creating a distinct frame.
    thickness = 8
    
    for box in boxes:
        x_min, y_min, x_max, y_max, score = box[:5]
        
        # Clamp coordinates to image bounds
        x0 = max(0, min(w, int(x_min)))
        y0 = max(0, min(h, int(y_min)))
        x1 = max(0, min(w, int(x_max)))
        y1 = max(0, min(h, int(y_max)))
        
        if x1 <= x0 or y1 <= y0:
            continue
            
        # We explicitly set pixels to 1.0 (Maximum Brightness/White)
        
        # Top Border
        edges[y0 : min(y0 + thickness, y1), x0:x1] = 1.0
        
        # Bottom Border
        edges[max(y0, y1 - thickness) : y1, x0:x1] = 1.0
        
        # Left Border
        edges[y0:y1, x0 : min(x0 + thickness, x1)] = 1.0
        
        # Right Border
        edges[y0:y1, max(x0, x1 - thickness) : x1] = 1.0
        
        # Note: We do NOT touch the pixels in the center. 
        # They retain their natural Edge detection values (the person's shape).

    # --- 5. Process Depth ---
    D = depth.astype(np.float32)
    if D.ndim == 3:
        D = D[:,:,0]
    
    # Stack channels: Y, Edges+Box, Depth
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

                # 1) Fused 3-channel image: gray + alpha (YOLO boxes) + depth
                fused_img = fuse_edges_boxes_depth(
                    rgb=image_2_resized,
                    boxes=boxes,
                    depth=depth_resized,
                    depth_min=0.0,
                    depth_max=1.0,  # ADD THIS - depth is already normalized
                )  # shape (96, 96, 3)

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
