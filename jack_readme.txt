ros2 launch roverrobotics_gazebo 4wd_rover_gazebo.launch.py
ros2 launch roverrobotics_gazebo Leo_rover_gazebo.launch.py

~/src/RoboTerrain/ros2_ws/src$ python ign_ros2_pose_topic.py island leo_rover
~/src/RoboTerrain/ros2_ws/src$ python ign_ros2_pose_topic.py inspect rover_zero4wd

~/src/attention/inference$ python ros2_mem_share.py 

/dynamic_obstacles$ python spawn.py trajectory_file_name actor_name walk_name world_name

python inference.py --attention_mode ./checkpoints/checkpoint_epoch_



python run_efficient_training.py --dataset_path /home/jack/data/social_nav --num_epochs 1000 --batch_size 4 --sequence_length 5 --target_width 320 --target_height 320 --yolo_model_path models/yolo11n.onnx --embedding_dim 128 --num_heads 4 --resume_checkpoint checkpoints/checkpoint_epoch

python run_efficient_training.py --dataset_path /home/jack/data/social_nav --num_epochs 1000 --batch_size 4 --sequence_length 5 --target_width 320 --target_height 320 --yolo_model_path models/yolo11n.onnx --embedding_dim 128 --num_heads 4 --resume_checkpoint checkpoints/checkpoint_epoch_2351.pkl 

(sb3) jack@HAL:~/src/RoboTerrain/ros2_ws/src/sb3$ python sb3_SAC.py --mode train --load False --world inspect --vision True



ros2 run teleop_twist_keyboard teleop_twist_keyboard
ros2 run rqt_image_view rqt_image_view



######################    fixes!!:
-change publisher to match frame rate of sim not wall clock time.
-negative reward for pointing toward human or prediction
-continuing training of attention mechanism
-test LeSTA



# Show sizes of all directories in current folder, sorted:
du -h --max-depth=1 | sort -h
du -sh ./*




camera test
ros2 launch roverrobotics_gazebo 4wd_rover_gazebo.launch.py

python run_efficient_training.py --dataset_path /home/jack/data/social_nav --num_epochs 1000 --batch_size 4 --sequence_length 5 --target_width 320 --target_height 320 --yolo_model_path ../models/yolo11n.onnx --embedding_dim 128 --num_heads 4

python run_efficient_training.py --dataset_path /home/jack/data/social_nav --num_epochs 1000 --batch_size 4 --sequence_length 5 --target_width 320 --target_height 320 --yolo_model_path ../models/yolo11n.onnx --embedding_dim 128 --num_heads 4 --resume_checkpoint ./model_output/checkpoint_epoch_1501.pkl


python cleanup_shm.py --all --force



python run_efficient_training.py --dataset_path /home/jack/data/social_nav/crossroad --num_epochs 30


python run_efficient_training.py --dataset_path /home/jack/data/social_nav/crossroad --num_epochs 30 --batch_size 8 --sequence_length 5 --target_width 320 --target_height 320 --yolo_model_path /path/to/your/yolo11n.onnx --embedding_dim 128 --num_heads 4


python run_efficient_training.py --dataset_path /home/jack/data/social_nav/crossroad --num_epochs 30

python run_trajectory_prediction.py --mode train --dataset_path /home/jack/data/social_nav/alley/ --num_epochs 1 --target_width 320 --target_height 320

unset LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.30


python run_efficient_training.py --dataset_path /home/jack/data/social_nav/subway --num_epochs 30 --batch_size 8 --sequence_length 5 --target_width 320 --target_height 320 --yolo_model_path ../models/yolo11n.onnx --embedding_dim 128 --num_heads 4


(attent) jack@HAL:~/src/attention$ python run_efficient_training.py --dataset_path /home/jack/data/social_nav --num_epochs 1000 --batch_size 1 --sequence_length 5 --target_width 320 --target_height 320 --yolo_model_path ../models/yolo11n.onnx --embedding_dim 128 --num_heads 4 --resume_checkpoint checkpoints/checkpoint_epoch_2351.pkl 
xFormers not available
xFormers not available
2025-06-22 14:16:46,375 - root - INFO - Preprocessing dataset at /home/jack/data/social_nav using memory mapping
2025-06-22 14:16:46,375 - root - INFO - Preprocessed dataset already exists at /home/jack/src/attention/preprocessed_data/social_nav_seq5_stride1_w320_h320_max1000.npz
2025-06-22 14:16:46,375 - root - INFO - Preprocessing dataset at /home/jack/data/social_nav using memory mapping
2025-06-22 14:16:46,375 - root - INFO - Preprocessed dataset already exists at /home/jack/src/attention/preprocessed_data/social_nav_seq5_stride2_w320_h320_max500.npz
2025-06-22 14:16:46,375 - root - INFO - Starting efficient training with 1000 epochs
