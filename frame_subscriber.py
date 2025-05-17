#!/usr/bin/env python3
# frame_subscriber.py
"""
Consumer for camera frames from shared memory FIFO for trajectory prediction.
This runs in the Python 3.11 environment with JAX/CUDA.

This improved version:
1. Only opens shared memory once and keeps it open
2. Uses a simpler, safer sampling helper
3. Properly cleans up resources
"""
import numpy as np
import time
from multiprocessing import shared_memory
import argparse
from typing import List, Tuple, Optional

# Configuration - must match producer
H, W = 320, 320            # Frame dimensions
FPS_HINT = 30              # Expected camera frame rate (approximate)
SPAN_SEC = 2.0             # Time span to maintain in the buffer
CAPACITY = 1 << (int(np.ceil(np.log2(FPS_HINT * SPAN_SEC))))  # e.g., 64

# Shared memory names - must match producer
SHM_IMG = "fifo_frames"
SHM_META = "fifo_meta"


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


def sample_window(
    frames: np.ndarray,
    stamps: np.ndarray,
    cursor: np.ndarray,
    span: float = 2.0,
    k: int = 5
) -> Optional[np.ndarray]:
    """
    Sample k evenly spaced frames from the most recent frames within time span.
    
    Args:
        frames: Numpy array of all frames (from shared memory)
        stamps: Numpy array of timestamps (from shared memory)
        cursor: Numpy array with current write position (from shared memory)
        span: Time span to sample from (seconds)
        k: Number of frames to retrieve
        
    Returns:
        Numpy array of k frames evenly spaced across the time span,
        or None if not enough frames are available
    """
    write_pos = int(cursor[0])
    now = time.time()
    
    # Collect indices of frames within time span
    idx = []
    for off in range(1, CAPACITY + 1):
        i = (write_pos - off) & (CAPACITY - 1)
        if now - stamps[i] > span:  # Too old
            break
        idx.append(i)
    
    # Check if we have enough frames
    if len(idx) < k:
        return None
    
    # Select evenly spaced indices
    sel = np.linspace(0, len(idx) - 1, num=k, dtype=int)
    
    # Extract and normalize frames
    window = frames[np.array(idx)[sel]].astype(np.float32) / 255.0  # (k, H, W, 3)
    
    # Calculate actual time span
    newest_time = stamps[idx[0]]
    oldest_time = stamps[idx[sel[-1]]]
    print(f"Retrieved {k} frames spanning {newest_time - oldest_time:.2f}s")
    
    return window


def main():
    """Main function for FIFO consumer."""
    parser = argparse.ArgumentParser(description="Consume frames from shared memory FIFO")
    parser.add_argument("--span", type=float, default=2.0, 
                        help="Time span to sample from (seconds)")
    parser.add_argument("--frames", type=int, default=5,
                        help="Number of frames to retrieve")
    parser.add_argument("--interval", type=float, default=0.1,
                        help="Time between polling attempts (seconds)")
    parser.add_argument("--frames-name", type=str, default=SHM_IMG,
                        help=f"Name of frames shared memory (default: {SHM_IMG})")
    parser.add_argument("--meta-name", type=str, default=SHM_META,
                        help=f"Name of metadata shared memory (default: {SHM_META})")
    
    args = parser.parse_args()
    
    print(f"FIFO consumer started. Sampling {args.frames} frames over {args.span}s span.")
    print(f"Connecting to shared memory: {args.frames_name} and {args.meta_name}")
    
    try:
        # Attach to shared memory once at the beginning
        shm_frames, shm_meta, frames_array, timestamps, cursor = attach_blocks(
            frames_name=args.frames_name,
            meta_name=args.meta_name
        )
        
        print("Successfully attached to shared memory blocks")
        
        # Main loop - sample frames at regular intervals
        while True:
            # Get a window of frames
            window = sample_window(
                frames=frames_array,
                stamps=timestamps,
                cursor=cursor,
                span=args.span,
                k=args.frames
            )
            
            if window is not None:
                # Process the frames here
                for i, frame in enumerate(window):
                    mean_pixel = np.mean(frame)
                    print(f"  Frame {i+1}/{args.frames}: mean pixel value = {mean_pixel:.4f}")
                
                # In actual use, you would send these frames to your model:
                # prediction = my_model(window)
            else:
                print("Not enough frames available yet. Waiting...")
                
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("Consumer stopped by user")
    except FileNotFoundError:
        print("Could not attach to shared memory - is the producer running?")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up properly, but only if we successfully attached
        try:
            shm_frames.close()
            shm_meta.close()
            print("Closed shared memory blocks")
        except:
            pass


if __name__ == "__main__":
    main()
