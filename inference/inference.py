#!/usr/bin/env python3
# yolo_frame_processor.py
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

# Import existing utilities
from trajectory_utils import (
    Pedestrian,
    detect_pedestrians_yolo_onnx,
    create_masks_from_pedestrians
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


class ProcessedBatch(NamedTuple):
    """Processed frames with detections, ready for trajectory prediction."""
    rgb_frames: np.ndarray  # [T, H, W, 3]
    mask_frames: np.ndarray  # [T, H, W, 1]
    timestamps: np.ndarray  # [T]


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
        # Connect to shared memory blocks - will raise FileNotFoundError if not found
        shm_frames = shared_memory.SharedMemory(name=frames_name)
        shm_meta = shared_memory.SharedMemory(name=meta_name)
        
        # Create views into shared memory
        frames_array = np.ndarray(
            (CAPACITY, H, W, 3),
            dtype=np.uint8,
            buffer=shm_frames.buf
        )
        
        meta_buf = shm_meta.buf
        timestamps = np.ndarray(
            (CAPACITY,),
            dtype=np.float64,
            buffer=meta_buf[:CAPACITY * 8]
        )
        cursor = np.ndarray(
            (1,),
            dtype=np.uint32, 
            buffer=meta_buf[CAPACITY * 8:]
        )
        
        return shm_frames, shm_meta, frames_array, timestamps, cursor
        
    except FileNotFoundError as e:
        print(f"Error: Could not find shared memory block: {e}")
        # Just print available segments without creating new SharedMemory objects
        try:
            import os
            if os.path.exists("/dev/shm"):
                print("Available shared memory segments:")
                for f in os.listdir("/dev/shm"):
                    if not f.startswith('psm_'):
                        continue
                    print(f"  - {f[4:]}")  # Strip 'psm_' prefix
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
    
    # For debugging
    newest_time = stamps[indices[0]]
    oldest_time = stamps[indices[sel[-1]]]
    print(f"Retrieved {num_frames} frames spanning {newest_time - oldest_time:.2f}s")
    
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


def process_buffer(
    yolo_session: ort.InferenceSession,
    frames: np.ndarray,
    timestamps: np.ndarray,
    cursor: np.ndarray,
    num_frames: int = 5,
    span: float = 2.0
) -> Optional[ProcessedBatch]:
    """
    Process the frame buffer to produce input for the attention model.
    
    Args:
        yolo_session: ONNX session for YOLO model
        frames: Numpy array of all frames (from shared memory)
        timestamps: Numpy array of timestamps (from shared memory)
        cursor: Numpy array with current write position (from shared memory)
        num_frames: Number of frames to sample
        span: Time span to sample from (seconds)
        
    Returns:
        ProcessedBatch object with RGB frames, mask frames, and timestamps,
        or None if not enough frames are available
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
        return None
    
    # Detect pedestrians in each frame
    rgb_frames, all_pedestrians = process_frames_with_yolo(
        frames=sampled_frames,
        yolo_session=yolo_session
    )
    
    # Create binary mask frames
    mask_frames = create_mask_frames(
        frames=rgb_frames,
        pedestrians_list=all_pedestrians
    )
    
    return ProcessedBatch(
        rgb_frames=rgb_frames,
        mask_frames=mask_frames,
        timestamps=sampled_timestamps
    )


def main():
    """Main function for YOLO frame processor."""
    parser = argparse.ArgumentParser(description="Process frames with YOLO for trajectory prediction")
    parser.add_argument("--yolo_model", type=str, default=DEFAULT_YOLO_PATH, 
                      help=f"Path to YOLO ONNX model (default: {DEFAULT_YOLO_PATH})")
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
    
    args = parser.parse_args()
    
    print(f"YOLO frame processor starting...")
    print(f"Using YOLO model: {args.yolo_model}")
    print(f"Sampling {args.frames} frames over {args.span}s span")
    print(f"Processing interval: {args.interval}s")
    
    try:
        # Load YOLO model
        yolo_session = ort.InferenceSession(
            args.yolo_model,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        print("Successfully loaded YOLO model")
        
        # Attach to shared memory blocks
        shm_frames, shm_meta, frames_array, timestamps, cursor = attach_blocks(
            frames_name=args.frames_name,
            meta_name=args.meta_name
        )
        print("Successfully attached to shared memory blocks")
        
        # Main processing loop
        while True:
            # Process buffer and get batch for attention model
            batch = process_buffer(
                yolo_session=yolo_session,
                frames=frames_array,
                timestamps=timestamps,
                cursor=cursor,
                num_frames=args.frames,
                span=args.span
            )
            
            if batch is not None:
                # Here you would pass batch to the attention model
                # For now, just print some statistics
                rgb_mean = np.mean(batch.rgb_frames)
                mask_mean = np.mean(batch.mask_frames)
                print(f"Processed batch: RGB mean={rgb_mean:.4f}, Mask mean={mask_mean:.4f}")
                
                # Number of pedestrians detected (non-zero mask pixels)
                num_pedestrians = np.sum(batch.mask_frames > 0.5)
                print(f"Detected {num_pedestrians} pedestrian pixels in masks")
                
                # TODO: Pass to attention model for trajectory prediction
            else:
                print("Not enough frames available yet")
            
            # Wait before next processing
            time.sleep(args.interval)
    
    except KeyboardInterrupt:
        print("Processor stopped by user")
    except FileNotFoundError:
        print("Could not attach to shared memory - is the producer running?")
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


if __name__ == "__main__":
    main()
