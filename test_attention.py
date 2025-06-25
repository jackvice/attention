#!/usr/bin/env python3
# test_attention_model.py
"""
Simple test script to validate attention model on gazebo dataset.
Tests first 10 sequences that have humans detected in the last frame.
"""
import os
import argparse
import glob
import numpy as np
import onnxruntime as ort
from typing import List, Tuple, Optional
import jax.numpy as jnp

# Import your utilities
from trajectory_utils import (
    detect_pedestrians_yolo_onnx,
    create_masks_from_pedestrians,
    load_and_preprocess_frame,
    create_fused_observation_jax,
    save_debug_observation,
    estimate_depth_pytorch
)


def load_attention_model(checkpoint_path: str):
    """Load attention model from checkpoint."""
    import pickle
    import jax
    from trajectory_model import SpatiotemporalAttention, ModelConfig
    
    print(f"Loading attention model from {checkpoint_path}")
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    params = checkpoint['params']
    config_dict = checkpoint.get('config', {})
    
    if isinstance(config_dict, dict):
        config = ModelConfig(**config_dict)
    else:
        config = config_dict
    
    model = SpatiotemporalAttention(config=config)
    
    @jax.jit
    def predict_fn(rgb_frames, mask_frames):
        if rgb_frames.ndim == 4:
            rgb_frames = rgb_frames[None, ...]  # Add batch dim
        if mask_frames.ndim == 4:
            mask_frames = mask_frames[None, ...]  # Add batch dim
        return model.apply({'params': params}, rgb_frames, mask_frames, training=False)
    
    return predict_fn, config


def get_sequence_frames(sequence_path: str, num_frames: int = 5) -> Optional[List[str]]:
    """Get consecutive frames from a sequence."""
    frame_pattern = os.path.join(sequence_path, "cam_img/1/data_rgb/*.png")
    frame_files = sorted(glob.glob(frame_pattern), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    
    if len(frame_files) < num_frames:
        return None
    
    # Take frames from middle of sequence
    start_idx = len(frame_files) // 2 - num_frames // 2
    return frame_files[start_idx:start_idx + num_frames]


def process_sequence(
    frame_paths: List[str],
    yolo_session: ort.InferenceSession,
    predict_fn,
    target_size: Tuple[int, int] = (320, 320)
) -> Tuple[Optional[np.ndarray], bool]:
    """
    Process a sequence and return fused observation if human detected in last frame.
    
    Returns:
        Tuple of (fused_observation, has_human_in_last_frame)
    """
    # Load and preprocess frames
    rgb_frames = []
    for frame_path in frame_paths:
        frame = load_and_preprocess_frame(frame_path, target_size[0], target_size[1])
        rgb_frames.append(frame)
    
    rgb_frames = np.array(rgb_frames)  # [T, H, W, 3]
    
    # Detect pedestrians in each frame
    all_pedestrians = []
    for frame in rgb_frames:
        pedestrians, _ = detect_pedestrians_yolo_onnx(frame, session=yolo_session)
        all_pedestrians.append(pedestrians)
    
    # Check if last frame has humans
    has_human_last = len(all_pedestrians[-1]) > 0
    if not has_human_last:
        return None, False
    
    # Create mask frames
    mask_frames = []
    for pedestrians in all_pedestrians:
        mask = create_masks_from_pedestrians(pedestrians, target_size[1], target_size[0])
        mask_frames.append(mask)
    
    mask_frames = np.array(mask_frames)  # [T, H, W, 1]
    
    # Run attention model
    rgb_jax = jnp.array(rgb_frames)
    mask_jax = jnp.array(mask_frames)
    heatmap = predict_fn(rgb_jax, mask_jax)
    heatmap_np = np.array(heatmap[0])  # Remove batch dim
    
    # Get depth for last frame (placeholder - using zeros)
    depth_frame = np.zeros(rgb_frames[-1].shape[:2], dtype=np.float32)
    
    # Create fused observation
    fused_obs = create_fused_observation_jax(
        rgb=jnp.array(rgb_frames[-1]),           # Last RGB frame
        depth=jnp.array(depth_frame),            # Depth (zeros for now)
        heatmap=jnp.array(heatmap_np),           # Predicted heatmap
        pedestrian_masks=jnp.array(mask_frames[-1]),  # Last mask frame
        target_height=96,
        target_width=96
    )
    
    return np.array(fused_obs), True


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test attention model on gazebo dataset")
    parser.add_argument("--checkpoint", type=str, required=True,
                      help="Path to attention model checkpoint")
    parser.add_argument("--dataset_path", type=str, 
                      default="/home/jack/data/social_nav/gazebo/gazebo_001",
                      help="Path to gazebo_001 dataset")
    parser.add_argument("--yolo_model", type=str, default="models/yolo11n.onnx",
                      help="Path to YOLO model")
    parser.add_argument("--output_dir", type=str, default="./test_attention",
                      help="Output directory for test results")
    
    args = parser.parse_args()
    
    print(f"Testing attention model: {args.checkpoint}")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Output directory: {args.output_dir}")
    
    # Load models
    predict_fn, config = load_attention_model(args.checkpoint)
    yolo_session = ort.InferenceSession(
        args.yolo_model,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    
    # Get all sequences from gazebo_001
    sequence_pattern = os.path.join(args.dataset_path, "sequence_*")
    all_sequences = sorted(glob.glob(sequence_pattern))
    print(f"Found {len(all_sequences)} sequences")
    
    # Process sequences until we find 10 with humans
    sequences_with_humans = 0
    sequences_processed = 0
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for sequence_dir in all_sequences:
        sequences_processed += 1
        sequence_name = os.path.basename(sequence_dir)
        
        print(f"Processing {sequence_name} ({sequences_processed}/{len(all_sequences)})")
        
        # Get 5 consecutive frames
        frame_paths = get_sequence_frames(sequence_dir, num_frames=5)
        if frame_paths is None:
            print(f"  Skipping {sequence_name} - not enough frames")
            continue
        
        # Process sequence
        fused_obs, has_human = process_sequence(frame_paths, yolo_session, predict_fn)
        
        if has_human:
            sequences_with_humans += 1
            
            # Save debug output
            output_subdir = os.path.join(args.output_dir, sequence_name)
            save_debug_observation(sequences_with_humans, fused_obs, output_subdir)
            
            print(f"  ✓ Found human in {sequence_name} - saved to {output_subdir}")
            
            # Stop after 10 sequences with humans
            if sequences_with_humans >= 10:
                break
        else:
            print(f"  No human detected in last frame of {sequence_name}")
    
    print(f"\nCompleted! Found {sequences_with_humans} sequences with humans out of {sequences_processed} processed.")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
