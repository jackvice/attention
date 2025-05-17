fixes!!:
loading depth and yolo from disk at each inference needs to be fixed to load once.
change numpy to jax.numpy


python run_efficient_training.py --dataset_path /home/jack/data/social_nav/subway --num_epochs 1000 --batch_size 4 --sequence_length 5 --target_width 320 --target_height 320 --yolo_model_path ../models/yolo11n.onnx --embedding_dim 128 --num_heads 4



python run_efficient_training.py --dataset_path /home/jack/data/social_nav/crossroad --num_epochs 30


python run_efficient_training.py --dataset_path /home/jack/data/social_nav/crossroad --num_epochs 30 --batch_size 8 --sequence_length 5 --target_width 320 --target_height 320 --yolo_model_path /path/to/your/yolo11n.onnx --embedding_dim 128 --num_heads 4


python run_efficient_training.py --dataset_path /home/jack/data/social_nav/crossroad --num_epochs 30

python run_trajectory_prediction.py --mode train --dataset_path /home/jack/data/social_nav/alley/ --num_epochs 1 --target_width 320 --target_height 320

unset LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.30


python run_efficient_training.py --dataset_path /home/jack/data/social_nav/subway --num_epochs 30 --batch_size 8 --sequence_length 5 --target_width 320 --target_height 320 --yolo_model_path ../models/yolo11n.onnx --embedding_dim 128 --num_heads 4
