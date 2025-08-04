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


import struct
# Import existing utilities
from trajectory_utils import (
    Pedestrian,
    detect_pedestrians_yolo_onnx,
    create_masks_from_pedestrians,
    create_fused_observation_jax,
    write_observation_to_shm,
    estimate_depth_pytorch,
    save_debug_observation,
    wait_for_depth_result,
    submit_depth_estimation
)



# Shared memory names - must match producer
SHM_IMG = "fifo_frames"
SHM_META = "fifo_meta"
SHM_NAME = "camera_latest"



# Configuration - must match new single-slot approach
H, W = 320, 320
BUFFER_SIZE = 60
SAMPLE_OFFSETS = [0, 15, 30, 45, 59]


# Default YOLO model path
DEFAULT_YOLO_PATH = "/home/jack/src/attention/models/yolo11n.onnx"


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

def read_latest_frame_if_new(
    shm: shared_memory.SharedMemory,
    last_timestamp: float
) -> Tuple[Optional[np.ndarray], float]:
    """
    Read frame from shared memory if timestamp is newer.
    
    Args:
        shm: Shared memory object
        last_timestamp: Last seen timestamp
        
    Returns:
        Tuple of (frame_array or None, current_timestamp)
    """
    # Read timestamp
    current_timestamp = struct.unpack_from('<d', shm.buf, 0)[0]
    
    # Check if frame is new
    if current_timestamp <= last_timestamp:
        return None, current_timestamp
    
    # Read frame data
    frame_bytes = bytes(shm.buf[8:8 + H*W*3])
    frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((H, W, 3))
    frame_float = frame.astype(np.float32) / 255.0
    
    return frame_float, current_timestamp


def sample_frames_from_buffer(
    frame_buffer: deque,
    sample_offsets: List[int] = SAMPLE_OFFSETS
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Sample frames at fixed offsets from newest.
    
    Args:
        frame_buffer: Buffer of (frame, timestamp) tuples
        sample_offsets: Indices to sample [0, 15, 30, 45, 59]
        
    Returns:
        Tuple of (sampled_frames, sampled_timestamps) or None if buffer not full
    """
    if len(frame_buffer) < max(sample_offsets) + 1:
        return None
    
    # Convert deque to list for indexing
    buffer_list = list(frame_buffer)
    
    sampled_frames = []
    sampled_timestamps = []
    
    for offset in sample_offsets:
        frame, timestamp = buffer_list[offset]
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
    if not all_pedestrians or len(all_pedestrians[-1]) == 0:
        return np.zeros((h, w, 1), dtype=np.float32)
    
    # Get predicted heatmap
    rgb_frames = jnp.array(batch.rgb_frames)
    mask_frames = jnp.array(batch.mask_frames)
    predictions = predict_fn(rgb_frames, mask_frames)
    predicted_heatmap = np.array(predictions[0])
    
    # Get current positions heatmap
    #current_heatmap = create_target_heatmap_from_pedestrians(
    #    all_pedestrians[-1], h, w, sigma=SIGMA_PX
    #)
    
    # Combine: current + predicted
    return predicted_heatmap #+ current_heatmap don't combine
    
def process_with_attention_old(
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


def main():
    """Main function with single-slot shared memory approach."""
    parser = argparse.ArgumentParser(description="Process frames with YOLO and predict trajectories")
    parser.add_argument("--yolo_model", type=str, default=DEFAULT_YOLO_PATH, 
                      help=f"Path to YOLO ONNX model (default: {DEFAULT_YOLO_PATH})")
    parser.add_argument("--attention_model", type=str, required=True,
                      help="Path to attention model checkpoint file")
    parser.add_argument("--rl-obs-name", type=str, default="rl_observation",
                      help="Name of RL observation shared memory (default: rl_observation)")
    
    args = parser.parse_args()
    
    print(f"Trajectory prediction pipeline starting...")
    print(f"Using YOLO model: {args.yolo_model}")
    print(f"Using attention model: {args.attention_model}")
    print(f"Sampling frames at offsets: {SAMPLE_OFFSETS}")
    
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
        print(f"Successfully loaded attention model")

        # Attach to shared memory
        shm = attach_single_frame_memory()
        print("Successfully attached to shared memory")
        
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
            name=args.rl_obs_name
        )

        # Create thread pool for depth estimation
        depth_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="depth_worker")
        print("Created ThreadPoolExecutor for depth estimation")
        
        # Initialize state
        frame_buffer = deque(maxlen=BUFFER_SIZE)
        last_timestamp = 0.0
        depth_session = None
        loop_idx = 0
        
        print("Waiting for frames...")
        
        # Main processing loop
        while True:
            loop_idx += 1
            t0 = time.perf_counter_ns()
            # Wait for a new frame to arrive
            while True:
                new_frame, current_timestamp = read_latest_frame_if_new(shm, last_timestamp)
                if new_frame is not None:
                    break
                time.sleep(0.0001)  # Short sleep while waiting for new frame
            t1 = time.perf_counter_ns()
            # Add to buffer (deque modifies in-place)
            frame_buffer.appendleft((new_frame, current_timestamp))
            last_timestamp = current_timestamp
            
            # Try to sample frames
            sample_result = sample_frames_from_buffer(frame_buffer)
            t2 = time.perf_counter_ns()
            if sample_result is not None:
                sampled_frames, sampled_timestamps = sample_result
                
                # Start depth estimation on latest frame
                latest_rgb = sampled_frames[0]  # First frame is newest
                depth_future = submit_depth_estimation(depth_executor, latest_rgb, depth_session)

                t3 = time.perf_counter_ns()
                
                # Process with YOLO
                all_pedestrians = []
                for frame in sampled_frames:
                    pedestrians, yolo_session = detect_pedestrians_yolo_onnx(frame, session=yolo_session)
                    all_pedestrians.append(pedestrians)

                t4 = time.perf_counter_ns()
                    
                # Create mask frames
                mask_frames = np.zeros((len(sampled_frames), H, W, 1), dtype=np.float32)
                for i, pedestrians in enumerate(all_pedestrians):
                    if pedestrians:
                        mask_frames[i] = create_masks_from_pedestrians(pedestrians, H, W)
                t5 = time.perf_counter_ns()
                
                # Process with attention
                batch = ProcessedBatch(
                    rgb_frames=sampled_frames,
                    mask_frames=mask_frames,
                    timestamps=sampled_timestamps
                )
                heatmap = process_with_attention(batch, all_pedestrians, predict_fn)

                t6 = time.perf_counter_ns()

                # Wait for depth
                depth_image, new_depth_session = wait_for_depth_result(depth_future, timeout_seconds=0.1)
                if new_depth_session is not None:
                    depth_session = new_depth_session
                if depth_image is None:
                    depth_image = np.zeros(latest_rgb.shape[:2], dtype=np.float32)

                t7 = time.perf_counter_ns()

                # Create fused observation
                try:
                    import jax.numpy as jnp
                    latest_mask = jnp.array(mask_frames[0])  # Newest frame mask
                    heatmap_jax = jnp.array(heatmap)
                    
                    fused_obs = create_fused_observation_jax(
                        rgb=latest_rgb,
                        depth=jnp.array(depth_image),
                        heatmap=heatmap_jax,
                        pedestrian_masks=latest_mask,
                        target_height=rl_obs_height,
                        target_width=rl_obs_width
                    )
                    
                    fused_obs_np = np.array(fused_obs).astype(np.float32)
                    write_observation_to_shm(fused_obs_np, rl_obs_shm)

                    t8 = time.perf_counter_ns()

                        
                except Exception as e:
                    print(f"Error creating RL observation: {e}")
            
            else:
                # Buffer not full yet
                if loop_idx % 100 == 0:
                    print(f"Buffer filling: {len(frame_buffer)}/{BUFFER_SIZE}")

            t9 = time.perf_counter_ns()

            if loop_idx % 500 == 0:
                #save_debug_observation(loop_idx, fused_obs, './debug_png')
                #print(f"in_ms | get_frames={(t1-t0)//1_000_000} | t2={(t2-t1)//1_000_000} | t3={(t3-t2)//1_000_000}")
                #print(f"yolo={(t4-t3)//1_000_000} | t5={(t5-t4)//1_000_000} | t6={(t6-t5)//1_000_000}")
                #print(f"depth={(t7-t6)//1_000_000} | t8={(t8-t7)//1_000_000} | total={(t8-t0)//1_000_000}")
                #print(" ")
                print(f"yolo={(t4-t3)//1_000_000} | mem_write={(t8-t7)//1_000_000} | total={(t9-t0)//1_000_000}")

  

            
    except KeyboardInterrupt:
        print("Processor stopped by user")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        try:
            depth_executor.shutdown(wait=True, timeout=2.0)
        except:
            pass
        try:
            shm.close()
        except:
            pass
        try:
            rl_obs_shm.close()
            rl_obs_shm.unlink()
        except:
            pass


def main_old():
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

        # CREATE THREAD POOL EXECUTOR FOR DEPTH ESTIMATION
        depth_executor = ThreadPoolExecutor(
            max_workers=1, 
            thread_name_prefix="depth_worker"
        )
        print("Created ThreadPoolExecutor for depth estimation")
        
        # Main processing loop
        depth_session = None
        loop_idx = 0        
        
        while True:
            loop_idx += 1
            loop_start_time = time.time()
            t0 = time.perf_counter_ns()            
            # STEP 1: Extract RGB frame and start depth estimation IMMEDIATELY
            latest_rgb = get_latest_rgb_frame(frames_array, cursor, CAPACITY)
            depth_future = submit_depth_estimation(depth_executor, latest_rgb, depth_session)
            
            t01 = time.perf_counter_ns()
            
            
            # STEP 2: Process YOLO (runs concurrently with depth estimation)
            batch, all_pedestrians = process_buffer(
                yolo_session=yolo_session,
                frames=frames_array,
                timestamps=timestamps,
                cursor=cursor,
                num_frames=args.frames,
                span=args.span
            )
            t1 = time.perf_counter_ns()
            
            if batch is not None:
                # STEP 3: Process attention (very fast, still concurrent with depth)
                heatmap = process_with_attention(batch, all_pedestrians, predict_fn)
                t2 = time.perf_counter_ns()
                
                # STEP 4: Wait for depth to complete (should be done or nearly done)
                t3 = time.perf_counter_ns()
                depth_image, new_depth_session = wait_for_depth_result(
                    depth_future, 
                    timeout_seconds=0.1  # Short timeout since it should be nearly done
                )
                
                # Update session only if depth succeeded
                if new_depth_session is not None:
                    depth_session = new_depth_session
                
                if depth_image is None:
                    depth_image = np.zeros(latest_rgb.shape[:2], dtype=np.float32)
                    print("Warning: Using fallback zero depth image")
                
                t4 = time.perf_counter_ns()
                
                # STEP 5: Create fused observation (both dependencies ready)
                try:
                    import jax.numpy as jnp
                    
                    latest_mask = jnp.array(batch.mask_frames[-1])
                    heatmap_jax = jnp.array(heatmap)
                    
                    
                    
                    fused_obs = create_fused_observation_jax(
                        rgb=latest_rgb,
                        depth=jnp.array(depth_image),
                        heatmap=heatmap_jax,
                        pedestrian_masks=latest_mask,
                        target_height=rl_obs_height,
                        target_width=rl_obs_width
                    )
                    
                    fused_obs_np = np.array(fused_obs).astype(np.float32)
                    write_observation_to_shm(fused_obs_np, rl_obs_shm)

                    t5 = time.perf_counter_ns()
                 
                except Exception as e:
                    print(f"Error creating RL observation: {e}")
                    import traceback
                    traceback.print_exc()
                
            else:
                print("Not enough frames available yet")
                depth_future.cancel()  # Cancel depth if no batch to process
            
            # Wait before next processing
            last_cursor_pos = cursor[0]
            while cursor[0] == last_cursor_pos:
                time.sleep(0.0001)
            t6 = time.perf_counter_ns()
            if loop_idx % 500 == 0:
                #save_debug_observation(loop_idx, fused_obs, './debug_png')
                print(f"timings_ms | depth_future={(t01-t0)//1_000_000} | yolo={(t1-t01)//1_000_000} | attn={(t2-t1)//1_000_000} | depth_wait={(t4-t3)//1_000_000} | fuse={(t5-t4)//1_000_000} | new_frame={(t6-t5)//1_000_000} | total={(t6-t0)//1_000_000}")

        
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
            depth_executor.shutdown(wait=True, timeout=2.0)
            print("Shut down ThreadPoolExecutor")
        except:
            pass
        
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
