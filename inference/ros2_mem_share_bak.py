#!/usr/bin/env python3
# camera_fifo_pub.py
"""
ROS2 node for subscribing to a camera topic and maintaining a ring buffer FIFO
for pedestrian trajectory prediction.

This improved version:
1. Always properly unlinks shared memory in destroy_node
2. Uses a more robust approach to handle cleanup
3. Uses consistent naming conventions
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from multiprocessing import shared_memory
import numpy as np
import cv2
import time
import atexit
from typing import List, Tuple, Optional
import signal
import sys

# Configuration
H, W = 320, 320            # Frame dimensions
FPS_HINT = 30              # Expected camera frame rate (approximate)
SPAN_SEC = 2.0             # Time span to maintain in the buffer
# Power-of-2 capacity ensures efficient wrapping with bitwise AND
CAPACITY = 1 << (int(np.ceil(np.log2(FPS_HINT * SPAN_SEC))))  # e.g., 64

# Shared memory names - IMPORTANT: These must match what the consumer expects
SHM_IMG = "fifo_frames"
SHM_META = "fifo_meta"

# Global variables for cleanup
g_shm_frames = None
g_shm_meta = None


def cleanup_shared_memory():
    """Clean up shared memory on exit."""
    global g_shm_frames, g_shm_meta
    
    try:
        if g_shm_frames is not None:
            g_shm_frames.close()
            g_shm_frames.unlink()
            print("Unlinked frames shared memory")
            
        if g_shm_meta is not None:
            g_shm_meta.close()
            g_shm_meta.unlink()
            print("Unlinked metadata shared memory")
    except Exception as e:
        print(f"Error cleaning up shared memory: {e}")


# Register cleanup function at exit
atexit.register(cleanup_shared_memory)


# Add signal handlers for SIGINT and SIGTERM
def signal_handler(sig, frame):
    print("Received signal, cleaning up...")
    cleanup_shared_memory()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


class CameraFIFO(Node):
    """
    ROS2 node that subscribes to a camera topic and maintains a ring buffer FIFO
    in shared memory. All frames for the last 2 seconds are stored, allowing
    the consumer to sample evenly spaced frames as needed.
    """
    
    def __init__(
        self,
        camera_topic: str = "/camera/image_raw",
        target_width: int = W, 
        target_height: int = H,
        capacity: int = CAPACITY,
        span_seconds: float = SPAN_SEC,
        shm_frames_name: str = SHM_IMG,
        shm_meta_name: str = SHM_META
    ):
        """
        Initialize the camera FIFO node.
        
        Args:
            camera_topic: ROS2 topic for camera images
            target_width: Width to resize images to
            target_height: Height to resize images to
            capacity: Number of frames to store in ring buffer
            span_seconds: Time span to maintain in seconds
            shm_frames_name: Name for frames shared memory block
            shm_meta_name: Name for metadata shared memory block
        """
        super().__init__("camera_fifo_node")
        
        # Save parameters
        self.camera_topic = camera_topic
        self.target_width = target_width
        self.target_height = target_height
        self.capacity = capacity
        self.span_seconds = span_seconds
        
        # Create CV bridge for converting ROS images to OpenCV format
        self.bridge = CvBridge()
        
        # Clean up any existing shared memory with these names
        self._cleanup_existing_shm(shm_frames_name, shm_meta_name)
        
        # Initialize shared memory for frames
        bytes_per_img = target_height * target_width * 3  # uint8 RGB
        self.shm_frames = shared_memory.SharedMemory(
            create=True,
            size=capacity * bytes_per_img,
            name=shm_frames_name
        )
        self.frames_array = np.ndarray(
            (capacity, target_height, target_width, 3),
            dtype=np.uint8,
            buffer=self.shm_frames.buf
        )
        
        # Initialize shared memory for metadata (timestamps + cursor)
        meta_size = capacity * 8 + 4  # float64 timestamps + uint32 cursor
        self.shm_meta = shared_memory.SharedMemory(
            create=True,
            size=meta_size,
            name=shm_meta_name
        )
        meta_buf = self.shm_meta.buf
        self.timestamps = np.ndarray(
            (capacity,),
            dtype=np.float64,
            buffer=meta_buf[:capacity * 8]
        )
        self.cursor = np.ndarray(
            (1,),
            dtype=np.uint32, 
            buffer=meta_buf[capacity * 8:]
        )
        # Initialize cursor and timestamps
        self.cursor[0] = 0
        self.timestamps.fill(0.0)
        
        # Set global references for cleanup
        global g_shm_frames, g_shm_meta
        g_shm_frames = self.shm_frames
        g_shm_meta = self.shm_meta
        
        # Create subscription to camera topic
        self.subscription = self.create_subscription(
            Image,
            camera_topic,
            self.camera_callback,
            10  # QoS profile depth
        )
        
        self.get_logger().info(
            f"Camera FIFO initialized. Listening on {camera_topic}, "
            f"FIFO capacity: {capacity} frames, spanning {span_seconds}s, "
            f"Shared memory: {shm_frames_name} and {shm_meta_name}"
        )
    
    def _cleanup_existing_shm(self, frames_name: str, meta_name: str) -> None:
        """Attempt to clean up any existing shared memory with these names."""
        try:
            shared_memory.SharedMemory(name=frames_name).unlink()
            self.get_logger().info(f"Cleaned up existing shared memory: {frames_name}")
        except FileNotFoundError:
            pass
        except Exception as e:
            self.get_logger().warning(f"Error cleaning up existing shared memory {frames_name}: {e}")
            
        try:
            shared_memory.SharedMemory(name=meta_name).unlink()
            self.get_logger().info(f"Cleaned up existing shared memory: {meta_name}")
        except FileNotFoundError:
            pass
        except Exception as e:
            self.get_logger().warning(f"Error cleaning up existing shared memory {meta_name}: {e}")
    
    def camera_callback(self, msg: Image) -> None:
        """
        Process incoming camera frames and add to the ring buffer.
        
        Args:
            msg: ROS Image message
        """
        try:
            # Calculate index in the ring (wrap around using modulo capacity)
            idx = int(self.cursor[0] % self.capacity)
            
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            
            # Resize to target dimensions
            resized = cv2.resize(cv_image, (self.target_width, self.target_height))
            
            # Store in shared memory
            self.frames_array[idx] = resized
            
            # Store timestamp (as Unix epoch time)
            self.timestamps[idx] = time.time()
            
            # Update cursor atomically (single assignment)
            self.cursor[0] = idx + 1
            
            # Log status periodically
            if idx % 300 == 0:  # Log approximately once per second at 300fps
                # Calculate number of frames within time span
                now = time.time()
                count = sum(1 for ts in self.timestamps if now - ts <= self.span_seconds)
                
                self.get_logger().info(
                    f"FIFO status: {count} frames within {self.span_seconds}s span, "
                    f"write index: {idx}/{self.capacity}"
                )
                
        except Exception as e:
            self.get_logger().error(f"Error processing camera frame: {str(e)}")
    
    def destroy_node(self):
        """Clean up shared memory when node is destroyed."""
        try:
            # Close and unlink the shared memory segments
            if hasattr(self, 'shm_frames'):
                self.shm_frames.close()
                self.shm_frames.unlink()
                
            if hasattr(self, 'shm_meta'):
                self.shm_meta.close()
                self.shm_meta.unlink()
                
            # Clear global references
            global g_shm_frames, g_shm_meta
            g_shm_frames = None
            g_shm_meta = None
            
            self.get_logger().info("Closed and unlinked shared memory")
        except Exception as e:
            self.get_logger().error(f"Error cleaning up shared memory: {str(e)}")
        
        super().destroy_node()


def main(args=None):
    """Run the ROS2 node."""
    rclpy.init(args=args)
    
    # Create and run node with default parameters
    node = CameraFIFO()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
