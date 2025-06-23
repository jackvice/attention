# preprocess_dataset.py
from typing import Dict, Tuple, List, Optional, Any, NamedTuple, Callable, Iterator
import os
import numpy as np
import jax.numpy as jnp
import jax
import onnxruntime as ort
from pathlib import Path
import time
import logging
import onnxruntime as ort
from functools import partial
import cv2
from config_temporal import FUTURE_OFFSET_F, PAST_OFFSETS_F, CAM_ID, SIGMA_PX


# Import your existing utility functions
from trajectory_utils import (
    scan_dataset,
    detect_pedestrians_yolo_onnx,
    load_and_preprocess_frame,
    create_masks_from_pedestrians,
    create_target_heatmap_from_pedestrians,
    TrajectorySequence,
    compute_trajectories,
    Frame
)

# In preprocess_dataset.py, find the function that creates frame sequences
# Update it to use the defined temporal offsets

def create_frame_sequences(
    frames: List[Frame],
    sequence_length: int = 5,
    stride: int = 1,
) -> List[Tuple[List[Frame], Frame]]:
    """
    Build (past‑sequence, future‑frame) pairs where all frames
    belong to the *same* (sequence_id, camera_id) clip and cover
    the offsets in PAST_OFFSETS_F plus FUTURE_OFFSET_F.
    """
    # Ensure time order
    frames = sorted(frames, key=lambda f: f.frame_id)

    # Quick reject if the clip is too short
    min_history = abs(min(PAST_OFFSETS_F))
    if len(frames) < min_history + FUTURE_OFFSET_F + 1:
        return []

    sequences = []
    max_start = len(frames) - FUTURE_OFFSET_F - 1

    for base_idx in range(min_history, max_start, stride):
        base = frames[base_idx]

        # ---- safety: refuse to cross a clip boundary ----
        if any(fr.sequence_id != base.sequence_id
               for fr in frames[base_idx - min_history : base_idx + 1]):
            continue
        # -------------------------------------------------

        past = []
        for off in PAST_OFFSETS_F:
            past.append(frames[base_idx + off])

        future = frames[base_idx + FUTURE_OFFSET_F]

        sequences.append((past, future))

    return sequences



# Updated preprocess_dataset function to add to preprocess_dataset.py
def preprocess_dataset_memmap(
    dataset_path: str,
    output_path: str,
    sequence_length: int = 5,
    target_width: int = 320,
    target_height: int = 320,
    yolo_model_path: str = "yolo11n.onnx",
    stride: int = 1,
    max_per_sequence: Optional[int] = None,
    debug_image_dir: Optional[str] = None,
    chunk_size: int = 100  # Process in chunks for better memory management
) -> str:
    """
    Preprocess a dataset once and save tensors to disk for fast loading.
    Uses memory mapping for reduced memory usage.
    
    Args:
        dataset_path: Path to original dataset
        output_path: Path to save preprocessed data
        sequence_length: Length of frame sequences
        target_width: Target frame width
        target_height: Target frame height
        yolo_model_path: Path to YOLO model
        stride: Stride for frame sequences
        max_per_sequence: Maximum frames per sequence (optional)
        debug_image_dir: Directory to save debug images (optional)
        chunk_size: Number of sequences to process at once
        
    Returns:
        Path to the saved dataset file
    """
    import tempfile
    
    logging.info(f"Preprocessing dataset at {dataset_path} using memory mapping")
    start_time = time.time()
    
    # Convert output_path to absolute path
    output_path = os.path.abspath(output_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Generate a filename based on parameters
    params = f"seq{sequence_length}_stride{stride}_w{target_width}_h{target_height}"
    if max_per_sequence:
        params += f"_max{max_per_sequence}"
    output_file = os.path.join(output_path, f"{Path(dataset_path).name}_{params}.npz")
    
    # Check if preprocessed file already exists
    if os.path.exists(output_file):
        logging.info(f"Preprocessed dataset already exists at {output_file}")
        return output_file
    
    # Scan dataset
    cameras = scan_dataset(dataset_path, max_per_sequence=max_per_sequence)
    
    # Create frame sequences
    all_frame_sequences = []
    for camera_id, frames in cameras.items():
        logging.info(f"Processing camera {camera_id} with {len(frames)} frames")
        if len(frames) >= sequence_length:
            frame_sequences = create_frame_sequences(
                frames, 
                sequence_length=sequence_length, 
                stride=stride
            )
            all_frame_sequences.extend(frame_sequences)
            logging.info(f"Created {len(frame_sequences)} sequences from camera {camera_id}")
    
    logging.info(f"Created {len(all_frame_sequences)} total frame sequences")
    
    # Set up a YOLO session to be reused

    yolo_session = ort.InferenceSession(
        yolo_model_path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    
    # detection function that maintains the signature expected by compute_trajectories
    # but reuses the YOLO session
    def detect_yolo_with_session(image):
        pedestrians, _ = detect_pedestrians_yolo_onnx(
            image,
            onnx_path=yolo_model_path,
            session=yolo_session
        )
        return pedestrians
    
    # Set __name__ attribute to match what compute_trajectories expects
    detect_yolo_with_session.__name__ = 'detect_pedestrians_yolo_onnx'
    
    # Compute trajectories to identify valid sequences
    logging.info("Computing trajectories to identify valid sequences...")
    trajectory_sequences = compute_trajectories(
        frame_sequences=all_frame_sequences,
        detect_fn=detect_yolo_with_session,
        target_width=target_width,
        target_height=target_height,
        yolo_model_path=yolo_model_path
    )
    
    # Filter valid sequences (with trajectories)
    valid_sequences = [seq for seq in trajectory_sequences if seq.trajectories]
    num_sequences = len(valid_sequences)
    
    logging.info(f"Found {num_sequences} valid sequences with trajectories")
    
    if not valid_sequences:
        logging.error("No valid sequences found, cannot create dataset")
        return ""
    
    # Create a temporary directory for memory-mapped files
    temp_dir = tempfile.mkdtemp(dir=output_path)
    
    # Initialize memmap variables to None for cleanup in finally block
    rgb_memmap = None
    mask_memmap = None
    target_memmap = None
    
    try:
        # Define memmap file paths
        rgb_memmap_path = os.path.join(temp_dir, "rgb_memmap.dat")
        mask_memmap_path = os.path.join(temp_dir, "mask_memmap.dat")
        target_memmap_path = os.path.join(temp_dir, "target_memmap.dat")
        
        # Create memory-mapped arrays
        rgb_shape = (num_sequences, sequence_length, target_height, target_width, 3)
        mask_shape = (num_sequences, sequence_length, target_height, target_width, 1)
        target_shape = (num_sequences, target_height, target_width, 1)
        
        logging.info(f"Creating memory-mapped arrays with shapes: RGB {rgb_shape}, mask {mask_shape}, target {target_shape}")
        
        rgb_memmap = np.memmap(
            rgb_memmap_path, 
            dtype=np.float16, 
            mode='w+', 
            shape=rgb_shape
        )
        
        mask_memmap = np.memmap(
            mask_memmap_path, 
            dtype=np.float16, 
            mode='w+', 
            shape=mask_shape
        )
        
        target_memmap = np.memmap(
            target_memmap_path, 
            dtype=np.float16, 
            shape=target_shape,
            mode='w+'
        )
        
        # Process sequences in chunks to further reduce memory usage
        num_chunks = (num_sequences + chunk_size - 1) // chunk_size
        logging.info(f"Processing {num_sequences} sequences in {num_chunks} chunks of size {chunk_size}")
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, num_sequences)
            chunk_size_actual = end_idx - start_idx
            
            logging.info(f"Processing chunk {chunk_idx+1}/{num_chunks}, sequences {start_idx}-{end_idx-1}")
            
            # Process each sequence in the chunk
            for i in range(start_idx, end_idx):
                traj_seq = valid_sequences[i]
                
                # Process RGB frames
                for j, frame in enumerate(traj_seq.frames):
                    # Load and preprocess the frame
                    rgb_data = load_and_preprocess_frame(
                        frame.path,
                        target_width=target_width,
                        target_height=target_height
                    )
                    rgb_memmap[i, j] = rgb_data.astype(np.float16)
                
                # Process mask frames
                for j, pedestrians in enumerate(traj_seq.pedestrians):
                    mask_data = create_masks_from_pedestrians(
                        pedestrians, 
                        height=target_height, 
                        width=target_width
                    )
                    mask_memmap[i, j] = mask_data.astype(np.float16)
                
                # Create target heatmap
                target_data = create_target_heatmap_from_pedestrians(
                    traj_seq.future_pedestrians,
                    target_height=target_height,
                    target_width=target_width,
                    sigma=SIGMA_PX
                )
                target_memmap[i] = target_data.astype(np.float16)
                
                # Save debug images if requested (keeping your original debug image code)
                if debug_image_dir and i % 20 == 0:  
                    os.makedirs(debug_image_dir, exist_ok=True)
                    
                    # Visualize input frames (from earliest to latest)
                    for j, frame in enumerate(traj_seq.frames):
                        rgb_img = load_and_preprocess_frame(frame.path, target_width, target_height)
                        rgb_img = (rgb_img * 255).astype(np.uint8)
                        bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                    
                        # Add timestamp text
                        time_offset = PAST_OFFSETS_F[j] / 10.0  # Convert to seconds assuming 10 FPS
                        cv2.putText(
                            bgr_img, 
                            f"t={time_offset:.1f}s", 
                            (20, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1, 
                            (255, 255, 255), 
                            2
                        )
                    
                        output_path_img = os.path.join(debug_image_dir,
                                                   f"sample_{i}_input_t{time_offset:.1f}s.png")
                        cv2.imwrite(output_path_img, bgr_img)
                
                    # Load and save future frame with YOLO detections
                    future_img = load_and_preprocess_frame(
                        traj_seq.future_frame.path,
                        target_width=target_width,
                        target_height=target_height
                    )
                    future_img_viz = (future_img * 255).astype(np.uint8)
                
                    # Draw pedestrian detections on future frame
                    future_with_detections = future_img_viz.copy()
                    for ped in traj_seq.future_pedestrians:
                        # Draw bounding box
                        x1, y1, x2, y2 = ped.bbox.astype(int)
                        cv2.rectangle(future_with_detections, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                        # Add confidence text
                        cv2.putText(
                            future_with_detections,
                            f"{ped.confidence:.2f}",
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2
                        )
                    
                        # Draw center position
                        cx, cy = ped.position.astype(int)
                        cv2.circle(future_with_detections, (cx, cy), 4, (255, 0, 0), -1)
                
                    # Add timestamp text
                    cv2.putText(
                        future_with_detections, 
                        f"t=+{FUTURE_OFFSET_F/10.0:.1f}s (truth)", 
                        (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (255, 255, 255), 
                        2
                    )

                    # Save future frame with detections
                    future_with_detections_bgr = cv2.cvtColor(future_with_detections,
                                                          cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(debug_image_dir,
                                             f"sample_{i}_future_with_detections.png"),
                                future_with_detections_bgr)
                
                    # Save the heatmap separately
                    heat = target_data[..., 0]
                    heat_colored = cv2.applyColorMap((heat * 255).astype(np.uint8),
                                                 cv2.COLORMAP_JET)
                    cv2.imwrite(os.path.join(debug_image_dir, f"sample_{i}_heatmap.png"),
                                heat_colored)
            
            # Flush changes to disk after each chunk
            rgb_memmap.flush()
            mask_memmap.flush()
            target_memmap.flush()
            
            logging.info(f"Completed chunk {chunk_idx+1}/{num_chunks}")
        
        # Create a temporary file for the final NPZ
        with tempfile.NamedTemporaryFile(dir=output_path, suffix='.npz', delete=False) as temp_npz:
            temp_npz_path = temp_npz.name
            
            # Close the file as np.savez_compressed will open it again
            temp_npz.close()
            
            logging.info(f"Saving preprocessed dataset to temporary file {temp_npz_path}")
            np.savez(
                temp_npz_path,
                rgb=rgb_memmap,
                mask=mask_memmap,
                target=target_memmap
            )
            
            # Atomically replace the output file
            logging.info(f"Moving temporary file to final location: {output_file}")
            os.replace(temp_npz_path, output_file)
        
        logging.info(f"Successfully saved preprocessed dataset to {output_file}")
        total_time = time.time() - start_time
        logging.info(f"Total preprocessing time: {total_time:.2f} seconds")
        
        return output_file
        
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return ""
        
    finally:
        # Clean up memory-mapped files
        for memmap_obj in [rgb_memmap, mask_memmap, target_memmap]:
            if memmap_obj is not None:
                try:
                    memmap_obj._mmap.close()
                except Exception:
                    pass
                
        # Clean up temporary directory
        import shutil
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logging.info(f"Cleaned up temporary directory {temp_dir}")
            except Exception as e:
                logging.warning(f"Failed to clean up temporary directory: {e}")



def create_memmap_batch_provider(
    data_path: str,
    batch_size: int = 8,
    shuffle: bool = True,
    rng_seed: int = 42
) -> Callable[[], Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """
    Create a batch provider that efficiently loads data from memory-mapped arrays.
    
    Args:
        data_path: Path to preprocessed dataset
        batch_size: Batch size
        shuffle: Whether to shuffle sequences
        rng_seed: Random seed for shuffling
        
    Returns:
        Callable that yields batches of (rgb, mask, target)
    """
    # Load dataset info without loading the data
    dataset = np.load(data_path, mmap_mode='r')
    
    rgb_data = dataset['rgb']
    mask_data = dataset['mask']
    target_data = dataset['target']
    
    num_sequences = rgb_data.shape[0]
    
    def batch_generator() -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        # Create shuffled indices
        indices = np.arange(num_sequences)
        
        if shuffle:
            rng = np.random.RandomState(rng_seed)
            rng.shuffle(indices)
        
        # Yield batches
        for start_idx in range(0, num_sequences, batch_size):
            end_idx = min(start_idx + batch_size, num_sequences)
            batch_indices = indices[start_idx:end_idx]
            
            # Extract data for this batch
            rgb_batch = rgb_data[batch_indices]
            mask_batch = mask_data[batch_indices]
            target_batch = target_data[batch_indices]
            
            # Pad to batch_size if needed
            if len(batch_indices) < batch_size:
                pad_size = batch_size - len(batch_indices)
                
                # Pad with zeros or duplicate the last example
                rgb_pad = np.zeros((pad_size,) + rgb_data.shape[1:], dtype=rgb_data.dtype)
                mask_pad = np.zeros((pad_size,) + mask_data.shape[1:], dtype=mask_data.dtype)
                target_pad = np.zeros((pad_size,) + target_data.shape[1:], dtype=target_data.dtype)
                
                rgb_batch = np.concatenate([rgb_batch, rgb_pad], axis=0)
                mask_batch = np.concatenate([mask_batch, mask_pad], axis=0)
                target_batch = np.concatenate([target_batch, target_pad], axis=0)
            
            yield rgb_batch, mask_batch, target_batch
    
    return batch_generator


