# run_efficient_training.py
import argparse
import logging
import os
from typing import Optional

# very top of run_efficient_training.py  (before importing NumPy)
import os, tempfile
os.environ["TMPDIR"] = "/home/jack/data/social_nav/tmp"   # any big partition
tempfile.tempdir = os.environ["TMPDIR"]                   # safety



def main():
    """Main function for efficient trajectory prediction training."""
    parser = argparse.ArgumentParser(description="Efficient Pedestrian Trajectory Prediction")
    
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to dataset (e.g., /home/jack/data/social_nav/crossroad)")
    parser.add_argument("--preprocess_only", action="store_true",
                        help="Only preprocess the dataset, don't train")
    parser.add_argument("--output_dir", type=str, default="./model_output",
                        help="Output directory for model and results")
    parser.add_argument("--preprocessed_dir", type=str, default="./preprocessed_data",
                        help="Directory to store preprocessed data")
    parser.add_argument("--num_epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--sequence_length", type=int, default=5,
                        help="Sequence length")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--target_width", type=int, default=320,
                        help="Target frame width")
    parser.add_argument("--target_height", type=int, default=320,
                        help="Target frame height")
    parser.add_argument("--yolo_model_path", type=str,
                        default="/home/jack/src/attention/models/yolo11n.onnx",
                        help="Path to YOLO model")
    parser.add_argument("--embedding_dim", type=int, default=128,
                        help="Embedding dimension")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="Number of attention heads")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("efficient_trajectory")
    
    # Import necessary modules
    from preprocess_dataset import preprocess_dataset, train_trajectory_model_efficient
    
    # Create preprocessed data directory
    os.makedirs(args.preprocessed_dir, exist_ok=True)
    
    # Preprocess training data (stride=1)
    train_data_path = preprocess_dataset(
        dataset_path=args.dataset_path,
        output_path=args.preprocessed_dir,
        sequence_length=args.sequence_length,
        target_width=args.target_width,
        target_height=args.target_height,
        yolo_model_path=args.yolo_model_path,
        stride=1,  # Use stride 1 for training
        max_per_sequence=1000  # Limit for faster iteration
    )
    
    # Preprocess validation data (stride=2)
    val_data_path = preprocess_dataset(
        dataset_path=args.dataset_path,
        output_path=args.preprocessed_dir,
        sequence_length=args.sequence_length,
        target_width=args.target_width,
        target_height=args.target_height,
        yolo_model_path=args.yolo_model_path,
        stride=2,  # Use stride 2 for validation (different subset)
        max_per_sequence=500  # Smaller set for validation
    )
    
    if not train_data_path:
        logger.error("Preprocessing failed on train data, cannot continue")
        return

    if not val_data_path:
        logger.error("Preprocessing failed on val data, cannot continue")
        return
    
    if args.preprocess_only:
        logger.info("Preprocessing completed, skipping training as requested")
        return
    
    # Train the model with preprocessed data
    train_trajectory_model_efficient(
        preprocessed_train_path=train_data_path,
        preprocessed_val_path=val_data_path,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        debug_image_dir=os.path.join(args.output_dir, "debug_images")
    )
    
    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()

    
