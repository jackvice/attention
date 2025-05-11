# test_preprocessing.py
import os
import numpy as np
import psutil
import gc
from preprocess_dataset import preprocess_dataset

def test_memmap_preprocessing(
    small_dataset_path: str,
    output_path: str,
    yolo_model_path: str
) -> None:
    """
    Test the memory-mapped preprocessing implementation with a small dataset.
    
    Args:
        small_dataset_path: Path to a small test dataset
        output_path: Path to save preprocessed output
        yolo_model_path: Path to YOLO model
    """
    # Force garbage collection to get accurate memory measurements
    gc.collect()
    
    # Get initial memory usage
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    
    # Run preprocessing with your existing implementation
    output_file = preprocess_dataset(
        dataset_path=small_dataset_path,
        output_path=output_path,
        sequence_length=5,
        target_width=320,
        target_height=320,
        yolo_model_path=yolo_model_path,
        stride=1,
        max_per_sequence=50  # Small number for testing
    )
    
    # Check if file was created
    assert os.path.exists(output_file), f"Output file {output_file} was not created"
    
    # Get peak memory usage
    peak_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"Peak memory usage: {peak_memory:.2f} MB")
    print(f"Memory increase: {peak_memory - initial_memory:.2f} MB")
    
    # Load and validate the dataset
    data = np.load(output_file)
    
    # Check for required keys
    assert 'rgb' in data, "RGB data not found in output file"
    assert 'mask' in data, "Mask data not found in output file"
    assert 'target' in data, "Target data not found in output file"
    
    # Check shapes
    rgb_data = data['rgb']
    mask_data = data['mask']
    target_data = data['target']
    
    num_sequences = rgb_data.shape[0]
    
    assert rgb_data.shape == (num_sequences, 5, 320, 320, 3), f"Unexpected RGB shape: {rgb_data.shape}"
    assert mask_data.shape == (num_sequences, 5, 320, 320, 1), f"Unexpected mask shape: {mask_data.shape}"
    assert target_data.shape == (num_sequences, 320, 320, 1), f"Unexpected target shape: {target_data.shape}"
    
    print(f"Successfully validated {num_sequences} sequences in the output file")
    print("Preprocessing test passed!")


if __name__ == "__main__":
    small_dataset_path = "/home/jack/data/social_nav/crossroad"  # Update with your path
    output_path = "./test_output"
    yolo_model_path = "/home/jack/src/attention/models/yolo11n.onnx"  # Update with your path
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Run the test
    test_memmap_preprocessing(
        small_dataset_path=small_dataset_path,
        output_path=output_path,
        yolo_model_path=yolo_model_path
    )
