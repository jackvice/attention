#!/usr/bin/env python3
# ros2_camera_buffer.py
"""
ROS2 node for subscribing to a camera topic and maintaining a buffer of frames
for pedestrian trajectory prediction.
"""
import os
import time
from collections import deque
from typing import List, Tuple, Optional, NamedTuple, Deque, Dict, Any
from dataclasses import dataclass

import numpy as np
import cv2
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

#from trajectory_utils import load_and_preprocess_frame

from multiprocessing.managers import SyncManager
from multiprocessing import Lock

BUFFER = []
BUFFER_LOCK = Lock()

class BufferManager(SyncManager): pass
BufferManager.register("get_buffer", callable=lambda: BUFFER)
BufferManager.register("get_lock",   callable=lambda: BUFFER_LOCK)


# Image processing utilities
def load_and_preprocess_frame(
    frame_path: str,
    target_width: int = 320,
    target_height: int = 320
) -> np.ndarray:
    """
    Load and preprocess a single frame.
    
    Args:
        frame_path: Path to the frame image
        target_width: Width to resize to
        target_height: Height to resize to
        
    Returns:
        Preprocessed image as numpy array [H,W,3] with float values in [0,1]
    """
    try:
        # Load image
        img = cv2.imread(frame_path)
        if img is None:
            logger.warning(f"Could not read image at {frame_path}")
            return np.zeros((target_height, target_width, 3), dtype=np.float32)
        
        # Resize
        img = cv2.resize(img, (target_width, target_height))
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0,1]
        img = img.astype(np.float32) / 255.0
        
        return img
    
    except Exception as e:
        logger.error(f"Error preprocessing frame {frame_path}: {e}")
        return np.zeros((target_height, target_width, 3), dtype=np.float32)

@dataclass
class TimestampedFrame:
    """A frame with its timestamp and preprocessed image data."""
    timestamp: float
    frame: np.ndarray
    original_frame: Optional[np.ndarray] = None


class CameraSubscriber(Node):
    """
    ROS2 node that subscribes to a camera topic and maintains a buffer
    of frames for trajectory prediction.
    """
    
    def __init__(
        self,
        camera_topic: str = "/camera/image_raw",
        buffer_size: int = 5,
        frame_interval: float = 0.5,  # Time between frames in seconds (0.5s = 2Hz)
        target_width: int = 320,
        target_height: int = 320,
        shared_buffer: Optional[Any] = None,
        lock: Optional[Any] = None
    ):
        """
        Initialize the camera subscriber node.
        
        Args:
            camera_topic: ROS2 topic for camera images
            buffer_size: Number of frames to keep in buffer
            frame_interval: Target time between consecutive frames
            target_width: Width to resize images to
            target_height: Height to resize images to
        """
        super().__init__("camera_trajectory_node")
        
        # Save parameters
        self.camera_topic = camera_topic
        self.buffer_size = buffer_size
        self.frame_interval = frame_interval
        self.target_width = target_width
        self.target_height = target_height
        
        # Initialize frame buffer
        self.frame_buffer: Deque[TimestampedFrame] = deque(maxlen=buffer_size)
        self.last_frame_time: float = 0.0
        
        # Create CV bridge for converting ROS images to OpenCV format
        self.cv_bridge = CvBridge()
        
        # Create subscription to camera topic
        self.subscription = self.create_subscription(
            Image,
            camera_topic,
            self.camera_callback,
            10  # QoS profile depth
        )

        self.shared_buffer = shared_buffer
        self.lock = lock
        
        self.get_logger().info(
            f"Camera subscriber initialized. Listening on {camera_topic}"
        )

    def publish_shared_buffer(self) -> None:
        if self.shared_buffer is None or self.lock is None:
            return
        
        self.lock.acquire()
        try:
            # replace entire contents
            self.shared_buffer.clear()                     # ← works on proxy
            self.shared_buffer.extend(
                [item.frame for item in self.frame_buffer] # must be iterable
            )
        finally:
            self.lock.release()



    def camera_callback(self, msg: Image) -> None:
        """
        Process incoming camera frames and update the buffer.
        
        Args:
            msg: ROS Image message
        """
        try:
            # Convert ROS Image message to OpenCV image
            current_time = self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec * 1e-9
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            
            # Preprocess the frame
            processed_frame = preprocess_frame(
                cv_image, 
                target_width=self.target_width,
                target_height=self.target_height
            )
            
            # Update the frame buffer
            self.update_buffer(current_time, processed_frame, cv_image)
            self.publish_shared_buffer()

            # Log status periodically
            if current_time - self.last_frame_time >= 1.0:  # Log every second
                self.log_buffer_status()
                self.last_frame_time = current_time
                
        except Exception as e:
            self.get_logger().error(f"Error processing camera frame: {str(e)}")
    
    def update_buffer(
        self, 
        timestamp: float, 
        processed_frame: np.ndarray,
        original_frame: np.ndarray
    ) -> None:
        """
        Add a new frame to the buffer, handling initialization if needed.
        
        Args:
            timestamp: Frame timestamp
            processed_frame: Preprocessed frame data
            original_frame: Original frame for visualization
        """
        # If buffer is empty, duplicate the first frame to fill buffer
        if len(self.frame_buffer) == 0:
            self.get_logger().info("Initializing frame buffer with first frame")
            
            # Create duplicates with decreasing timestamps
            for i in range(self.buffer_size - 1, -1, -1):
                duplicate_timestamp = timestamp - (i * self.frame_interval)
                self.frame_buffer.append(
                    TimestampedFrame(
                        timestamp=duplicate_timestamp,
                        frame=processed_frame.copy(),
                        original_frame=original_frame.copy()
                    )
                )
        else:
            # Check if enough time has passed since the last frame
            last_timestamp = self.frame_buffer[-1].timestamp
            time_diff = timestamp - last_timestamp
            
            # Only add frame if it's been at least frame_interval/2 seconds
            # (allowing some flexibility in timing)
            #if time_diff >= self.frame_interval / 2:
            if time_diff >= self.frame_interval:
                self.frame_buffer.append(
                    TimestampedFrame(
                        timestamp=timestamp,
                        frame=processed_frame,
                        original_frame=original_frame
                    )
                )
                
                # Log when we add a new frame to the trajectory buffer
                self.get_logger().debug(
                    f"Added frame to buffer, time diff: {time_diff:.2f}s"
                )
    
    def get_buffer_frames(self) -> Tuple[List[np.ndarray], List[float]]:
        """
        Get all frames and timestamps from the buffer.
        
        Returns:
            Tuple of (frames, timestamps)
        """
        frames = [item.frame for item in self.frame_buffer]
        timestamps = [item.timestamp for item in self.frame_buffer]
        return frames, timestamps
    
    def log_buffer_status(self) -> None:
        """Log information about the current state of the frame buffer."""
        if not self.frame_buffer:
            self.get_logger().info("Frame buffer is empty")
            return
            
        # Get time span of buffer
        oldest_time = self.frame_buffer[0].timestamp
        newest_time = self.frame_buffer[-1].timestamp
        time_span = newest_time - oldest_time
        
        self.get_logger().info(
            f"Buffer status: {len(self.frame_buffer)}/{self.buffer_size} frames, "
            f"spanning {time_span:.2f}s"
        )


def preprocess_frame(
    frame: np.ndarray,
    target_width: int = 320,
    target_height: int = 320
) -> np.ndarray:
    """
    Preprocess a frame for use with the trajectory prediction model.
    
    Args:
        frame: Input RGB frame
        target_width: Width to resize to
        target_height: Height to resize to
    
    Returns:
        Preprocessed frame as a numpy array with shape [H,W,3] and values in [0,1]
    """
    # Ensure frame is in RGB format (should already be from cv_bridge)
    
    # Resize to target dimensions
    resized = cv2.resize(frame, (target_width, target_height))
    
    # Convert to float32 and normalize to [0,1]
    normalized = resized.astype(np.float32) / 255.0
    
    return normalized

"""
def main_old(args=None):

    rclpy.init(args=args)
    
    # Create and run node with default parameters
    node = CameraSubscriber()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        node.destroy_node()
        rclpy.shutdown()
"""


# camera_shared_launcher.py
from multiprocessing import Manager, Lock, Process
import time
import numpy as np

import rclpy
from ros2_camera import CameraSubscriber  # assumes it's in ros2_camera.py

def start_ros_camera_node(shared_buffer, lock):
    rclpy.init()
    node = CameraSubscriber(shared_buffer=shared_buffer, lock=lock)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()



from multiprocessing.managers import SyncManager
from multiprocessing import RLock

BUFFER = []
BUFFER_LOCK = RLock()  # or Lock()



def main():
    # Start manager server in background thread


    class BufferManager(SyncManager): pass
    BufferManager.register(
        "get_buffer",
        callable=lambda: BUFFER,
        exposed=[
            '__getitem__', '__setitem__', '__delitem__',
            '__len__', 'append', 'extend', 'clear', '__iter__'
        ]
    )

    BufferManager.register(
        "get_lock",
        callable=lambda: BUFFER_LOCK,
        exposed=['acquire', 'release']
    )
    mgr = BufferManager(address=("localhost", 50055), authkey=b"secret")
    mgr.start()
    print("[ROS] Manager started on port 50055")

    # Get shared buffer and lock (local access)
    shared_buffer = mgr.get_buffer()
    lock = mgr.get_lock()

    # Start ROS node with access to shared buffer
    rclpy.init()
    node = CameraSubscriber(shared_buffer=shared_buffer, lock=lock)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
        

