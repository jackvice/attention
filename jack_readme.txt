ros2 launch roverrobotics_gazebo 4wd_rover_gazebo.launch.py

~/src/RoboTerrain/ros2_ws/src$ python ign_ros2_pose_topic.py inspect rover_zero4wd

~/src/attention/yolo/inference$ python ros2_mem_share.py 

~/src/RoboTerrain/ros2_ws/src/dynamic_obstacles$ python spawn.py 

python inference.py --attention_mode ./model_output/checkpoint_epoch_1501.pkl
~/src/attention/yolo/inference/ros2_mem_share.py 


(sb3) jack@HAL:~/src/RoboTerrain/ros2_ws/src/sb3$ python sb3_SAC_fused.py --mode train --load False --world inspect --vision True


fixes!!:
loading depth and yolo from disk at each inference needs to be fixed to load once.
change numpy to jax.numpy

camera test
ros2 launch roverrobotics_gazebo 4wd_rover_gazebo.launch.py

python run_efficient_training.py --dataset_path /home/jack/data/social_nav/subway --num_epochs 1000 --batch_size 4 --sequence_length 5 --target_width 320 --target_height 320 --yolo_model_path ../models/yolo11n.onnx --embedding_dim 128 --num_heads 4


python cleanup_shm.py --all --force



python run_efficient_training.py --dataset_path /home/jack/data/social_nav/crossroad --num_epochs 30


python run_efficient_training.py --dataset_path /home/jack/data/social_nav/crossroad --num_epochs 30 --batch_size 8 --sequence_length 5 --target_width 320 --target_height 320 --yolo_model_path /path/to/your/yolo11n.onnx --embedding_dim 128 --num_heads 4


python run_efficient_training.py --dataset_path /home/jack/data/social_nav/crossroad --num_epochs 30

python run_trajectory_prediction.py --mode train --dataset_path /home/jack/data/social_nav/alley/ --num_epochs 1 --target_width 320 --target_height 320

unset LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.30


python run_efficient_training.py --dataset_path /home/jack/data/social_nav/subway --num_epochs 30 --batch_size 8 --sequence_length 5 --target_width 320 --target_height 320 --yolo_model_path ../models/yolo11n.onnx --embedding_dim 128 --num_heads 4
