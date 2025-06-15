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

import struct
# Import existing utilities
from trajectory_utils import (
    Pedestrian,
    detect_pedestrians_yolo_onnx,
    create_masks_from_pedestrians,
    create_fused_observation_jax,
    write_observation_to_shm,
    estimate_depth_pytorch,
    save_debug_observation
)

# Configuration - must match producer
H, W = 320, 320            # Frame dimensions
FPS_HINT = 30              # Expected camera frame rate (approximate)
SPAN_SEC = 2.0             # Time span to maintain in the buffer
CAPACITY = 1 << (int(np.ceil(np.log2(FPS_HINT * SPAN_SEC))))  # e.g., 64

# Shared memory names - must match producer
SHM_IMG = "fifo_frames"
SHM_META = "fifo_meta"

# Default YOLO model path
DEFAULT_YOLO_PATH = "/home/jack/src/attention/models/yolo11n.onnx"


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
        Predicted trajectory heatmap [H, W, 1] or zeros if no person in last frame
    """
    import jax.numpy as jnp
    
    # Check if last frame has pedestrians
    if not all_pedestrians or len(all_pedestrians[-1]) == 0:
        # Return empty heatmap with same shape as expected output
        h, w = batch.rgb_frames.shape[1:3]  # Get spatial dimensions
        return np.zeros((h, w, 1), dtype=np.float32)
    
    # Convert to JAX arrays and run prediction
    rgb_frames = jnp.array(batch.rgb_frames)
    mask_frames = jnp.array(batch.mask_frames)
    predictions = predict_fn(rgb_frames, mask_frames)
    
    return np.array(predictions[0])


def process_buffer(
    yolo_session: ort.InferenceSession,
    frames: np.ndarray,
    timestamps: np.ndarray,
    cursor: np.ndarray,
    num_frames: int = 5,
    span: float = 2.0
) -> Tuple[Optional[ProcessedBatch], List[List[Pedestrian]]]:
    """
    Process buffer - UPDATED to return pedestrian data for detection gating.
    
    Returns:
        Tuple of (ProcessedBatch, all_pedestrians_list) or (None, [])
    """
    # Sample frames evenly across the buffer
    sampled_frames, sampled_timestamps = sample_frames_evenly(
        frames=frames,
        stamps=timestamps,
        cursor=cursor,
        num_frames=num_frames,
        span=span
    )
    
    if sampled_frames is None:
        return None, []
    
    # Detect pedestrians in each frame
    rgb_frames, all_pedestrians = process_frames_with_yolo(
        frames=sampled_frames,
        yolo_session=yolo_session
    )
    
    # Create binary mask frames (now with center boxes)
    mask_frames = create_mask_frames(
        frames=rgb_frames,
        pedestrians_list=all_pedestrians
    )
    
    batch = ProcessedBatch(
        rgb_frames=rgb_frames,
        mask_frames=mask_frames,
        timestamps=sampled_timestamps
    )
    
    return batch, all_pedestrians


def attach_blocks(
    frames_name: str = SHM_IMG,
    meta_name: str = SHM_META
) -> Tuple[shared_memory.SharedMemory, shared_memory.SharedMemory, np.ndarray, np.ndarray, np.ndarray]:
    """
    Attach to shared memory blocks and create numpy views.
    Keep these open for the lifetime of the program.

    Args:
        frames_name: Name of frames shared memory block
        meta_name: Name of metadata shared memory block

    Returns:
        Tuple of (frames_shm, meta_shm, frames_array, timestamps_array, cursor_array)
    """
    try:
        # Prevent Python's resource_tracker from unlinking on crash
        shm_frames = shared_memory.SharedMemory(name=frames_name, track=False)
        shm_meta   = shared_memory.SharedMemory(name=meta_name,   track=False)

        frames_array = np.ndarray(
            (CAPACITY, H, W, 3), dtype=np.uint8, buffer=shm_frames.buf
        )

        timestamps = np.ndarray(
            (CAPACITY,), dtype=np.float64, buffer=shm_meta.buf[:CAPACITY * 8]
        )

        cursor = np.ndarray(
            (1,), dtype=np.uint32, buffer=shm_meta.buf[CAPACITY * 8:]
        )

        return shm_frames, shm_meta, frames_array, timestamps, cursor

    except FileNotFoundError as e:
        print(f"Error: Could not find shared memory block: {e}, SHM_IMG: {frames_name}, SHM_META: {meta_name}")
        try:
            import os
            if os.path.exists("/dev/shm"):
                print("Available shared memory segments:")
                for f in os.listdir("/dev/shm"):
                    print(f"  - {f}")
        except:
            pass
        raise


def sample_frames_evenly(
    frames: np.ndarray,
    stamps: np.ndarray,
    cursor: np.ndarray,
    num_frames: int = 5,
    span: float = 2.0
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Sample num_frames evenly spaced frames from the most recent frames within time span.
    
    Args:
        frames: Numpy array of all frames (from shared memory)
        stamps: Numpy array of timestamps (from shared memory)
        cursor: Numpy array with current write position (from shared memory)
        num_frames: Number of frames to retrieve
        span: Time span to sample from (seconds)
        
    Returns:
        Tuple of (sampled frames, sampled timestamps),
        or (None, None) if not enough frames are available
    """
    write_pos = int(cursor[0])
    now = time.time()
    
    # Collect indices of frames within time span
    indices = []
    for off in range(1, CAPACITY + 1):
        i = (write_pos - off) & (CAPACITY - 1)
        if now - stamps[i] > span:  # Too old
            break
        indices.append(i)
    
    # Check if we have enough frames
    if len(indices) < num_frames:
        return None, None
    
    # Select evenly spaced indices
    sel = np.linspace(0, len(indices) - 1, num=num_frames, dtype=int)
    
    # Extract and normalize frames
    selected_indices = np.array(indices)[sel]
    sampled_frames = frames[selected_indices].astype(np.float32) / 255.0  # (K, H, W, 3)
    sampled_timestamps = stamps[selected_indices]
    
    newest_time = stamps[indices[0]]
    oldest_time = stamps[indices[sel[-1]]]
    #print(f"Retrieved {num_frames} frames spanning {newest_time - oldest_time:.2f}s")
    
    return sampled_frames, sampled_timestamps


def process_frames_with_yolo(
    frames: np.ndarray,
    yolo_session: ort.InferenceSession
) -> Tuple[np.ndarray, List[List[Pedestrian]]]:
    """
    Process frames with YOLO to detect pedestrians.
    
    Args:
        frames: Numpy array of frames [T, H, W, 3] with values in [0,1]
        yolo_session: ONNX session for YOLO model
        
    Returns:
        Tuple of (frames, list of pedestrian detections for each frame)
    """
    all_pedestrians = []
    
    for frame in frames:
        # Detect pedestrians in frame
        pedestrians, yolo_session = detect_pedestrians_yolo_onnx(
            frame,
            session=yolo_session
        )
        
        all_pedestrians.append(pedestrians)
    
    return frames, all_pedestrians


def create_mask_frames(
    frames: np.ndarray,
    pedestrians_list: List[List[Pedestrian]]
) -> np.ndarray:
    """
    Create binary mask frames for detected pedestrians.
    
    Args:
        frames: Numpy array of frames [T, H, W, 3]
        pedestrians_list: List of pedestrian detections for each frame
        
    Returns:
        Numpy array of mask frames [T, H, W, 1]
    """
    T, H, W, _ = frames.shape
    mask_frames = np.zeros((T, H, W, 1), dtype=np.float32)
    
    for i, pedestrians in enumerate(pedestrians_list):
        if pedestrians:
            mask_frames[i] = create_masks_from_pedestrians(
                pedestrians,
                height=H,
                width=W
            )
    
    return mask_frames


def wait_for_blocks(frames_name="fifo_frames",
                    meta_name="fifo_meta",
                    timeout=5.0,  # total seconds to wait
                    delay=0.1):   # seconds between tries
    import time, os
    t0 = time.time()
    while True:
        try:
            return attach_blocks(frames_name, meta_name)
        except FileNotFoundError:
            if time.time() - t0 > timeout:
                raise RuntimeError(
                    f"Timed out after {timeout}s – segments never appeared")
            time.sleep(delay)



def main():
    """Main function for YOLO frame processor with trajectory prediction."""
    parser = argparse.ArgumentParser(description="Process frames with YOLO and predict trajectories")
    parser.add_argument("--yolo_model", type=str, default=DEFAULT_YOLO_PATH, 
                      help=f"Path to YOLO ONNX model (default: {DEFAULT_YOLO_PATH})")
    parser.add_argument("--attention_model", type=str, required=True,
                      help="Path to attention model checkpoint file")
    parser.add_argument("--frames", type=int, default=5,
                      help="Number of frames to sample (default: 5)")
    parser.add_argument("--span", type=float, default=2.0,
                      help="Time span to sample from in seconds (default: 2.0)")
    parser.add_argument("--interval", type=float, default=0.1,
                      help="Processing interval in seconds (default: 0.1)")
    parser.add_argument("--frames-name", type=str, default=SHM_IMG,
                      help=f"Name of frames shared memory (default: {SHM_IMG})")
    parser.add_argument("--meta-name", type=str, default=SHM_META,
                      help=f"Name of metadata shared memory (default: {SHM_META})")
    parser.add_argument("--rl-obs-name", type=str, default="rl_observation",
                      help="Name of RL observation shared memory (default: rl_observation)")
    
    args = parser.parse_args()
    
    print(f"Trajectory prediction pipeline starting...")
    print(f"Using YOLO model: {args.yolo_model}")
    print(f"Using attention model: {args.attention_model}")
    print(f"Sampling {args.frames} frames over {args.span}s span")
    print(f"Processing interval: {args.interval}s")
    print(f"SHM_IMG: {args.frames_name}, SHM_META: {args.meta_name} ")    
    # RL observation parameters
    rl_obs_height, rl_obs_width = 96, 96
    rl_obs_channels = 3

    try:
        # Load YOLO model
        yolo_session = ort.InferenceSession(
            args.yolo_model,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        print("Successfully loaded YOLO model")
        
        # Load attention model
        predict_fn, model_info = load_attention_model(args.attention_model)
        print(f"Successfully loaded attention model with embedding_dim={model_info['config'].embedding_dim}")


        shm_frames, shm_meta, frames_array, timestamps, cursor = wait_for_blocks(
            frames_name=args.frames_name,
            meta_name=args.meta_name )
        print("Successfully attached to shared memory blocks")
        # Attach to shared memory blocks
        #shm_frames, shm_meta, frames_array, timestamps, cursor = attach_blocks(
        #    frames_name=args.frames_name,
        #    meta_name=args.meta_name
        #)
        print("Successfully attached to shared memory blocks")
        
        # Setup RL observation shared memory
        #from rl_observation_functions import write_observation_to_shm
        from trajectory_utils import write_observation_to_shm
        
        # Clean up any existing RL observation shared memory
        try:
            shared_memory.SharedMemory(name=args.rl_obs_name).unlink()
            print(f"Cleaned up existing RL observation shared memory: {args.rl_obs_name}")
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Warning: Could not clean up existing RL observation shared memory: {e}")
        
        # Create RL observation shared memory
        header_size = 8 + 4  # timestamp + valid flag
        obs_data_size = rl_obs_height * rl_obs_width * rl_obs_channels * 4  # float32
        rl_obs_shm_size = header_size + obs_data_size
        
        rl_obs_shm = shared_memory.SharedMemory(
            create=True,
            size=rl_obs_shm_size,
            name=args.rl_obs_name
        )
        print(f"Created RL observation shared memory: {args.rl_obs_name} ({rl_obs_shm_size} bytes)")
        
        # Main processing loop
        depth_session = None
        
        loop_idx = 0        
        while True:
            loop_idx += 1
            loop_start_time = time.time()
            
            # Process buffer and get batch for attention model
            """
            batch = process_buffer(
                yolo_session=yolo_session,
                frames=frames_array,
                timestamps=timestamps,
                cursor=cursor,
                num_frames=args.frames,
                span=args.span
            )
            
            if batch is not None:
                # Process with attention model
                start_time = time.time()
                heatmap = process_with_attention(batch, predict_fn)
            """
            batch, all_pedestrians = process_buffer(
                yolo_session=yolo_session,
                frames=frames_array,
                timestamps=timestamps,
                cursor=cursor,
                num_frames=args.frames,
                span=args.span
            )
            
            if batch is not None:
                # Process with attention model (now with detection gating)
                start_time = time.time()
                heatmap = process_with_attention(batch, all_pedestrians, predict_fn)
           
                inference_time = time.time() - start_time
                
                # Calculate some statistics for debugging
                heat_max = np.max(heatmap)
                heat_mean = np.mean(heatmap)
                heat_nonzero = np.mean(heatmap > 0.1)
                
                # Create fused observation for RL agent
                try:
                    import jax.numpy as jnp
                    #from rl_observation_functions import create_fused_observation_jax
                    
                    # Get latest RGB frame and mask frame
                    latest_rgb = jnp.array(batch.rgb_frames[-1])  # [H, W, 3]
                    latest_mask = jnp.array(batch.mask_frames[-1])  # [H, W, 1]
                    
                    # Create zero depth frame (placeholder)
                    #depth_frame = jnp.zeros(latest_rgb.shape[:2], dtype=jnp.float32)  # [H, W]
                    latest_frame_idx = (cursor[0] - 1) & (CAPACITY - 1)
                    latest_rgb = frames_array[latest_frame_idx].astype(np.float32) / 255.0
                    # Get depth for latest frame
                    depth_image, depth_session = estimate_depth_pytorch(
                        latest_rgb, 
                        session=depth_session
                    )
                    # Convert heatmap to JAX array
                    heatmap_jax = jnp.array(heatmap)  # [H, W, 1]
                    
                    # Create fused observation
                    fused_obs = create_fused_observation_jax(
                        rgb=latest_rgb,
                        depth=jnp.array(depth_image), #depth_frame,
                        heatmap=heatmap_jax,
                        pedestrian_masks=latest_mask,
                        target_height=rl_obs_height,
                        target_width=rl_obs_width
                    )
                    
                    # Convert to numpy for shared memory
                    fused_obs_np = np.array(fused_obs).astype(np.float32)
                    
                    # Write to shared memory
                    write_observation_to_shm(fused_obs_np, rl_obs_shm)
                    
                    #print(f"Created and wrote RL observation {fused_obs_np.shape} to shared memory")
                    loop_end_time = time.time()
                    total_loop_time = loop_end_time - loop_start_time
                    
                    if loop_idx % 1000 == 0:
                        save_debug_observation( loop_idx, fused_obs, './debug_png' )
                        print(f"Complete loop cycle: {total_loop_time*1000:.1f}ms at {time.time():.3f}")
                        print(f"Prediction: max={heat_max:.3f}, mean={heat_mean:.3f}, "
                              f"coverage={heat_nonzero:.1%}, time={inference_time*1000:.1f}ms")
                
                    
        
                except Exception as e:
                    print(f"Error creating RL observation: {e}")
                    import traceback
                    traceback.print_exc()
                
            else:
                print("Not enough frames available yet")
            
            # Wait before next processing
            last_cursor_pos = cursor[0]
            while cursor[0] == last_cursor_pos:
                time.sleep(0.001)  # 1ms polling
    
    except KeyboardInterrupt:
        print("Processor stopped by user")
    #except FileNotFoundError:
    #    print("Could not attach to shared memory - is the producer running?")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up resources
        try:
            shm_frames.close()
            shm_meta.close()
            print("Closed shared memory blocks")
        except:
            pass
        
        try:
            rl_obs_shm.close()
            rl_obs_shm.unlink()
            print("Closed and unlinked RL observation shared memory")
        except:
            pass


if __name__ == "__main__":
    main()
