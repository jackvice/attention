#!/usr/bin/env python3
"""
ROS2 node for writing camera frames to a single shared memory slot.
Consumer maintains its own FILO buffer.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from multiprocessing import shared_memory
import numpy as np
import cv2
import time
import struct
import atexit
import signal
import sys
from typing import Optional

# Configuration
H, W = 320, 320
SHM_NAME = "camera_latest"

# Global for cleanup
g_shm = None

def cleanup_shared_memory():
    """Clean up shared memory on exit."""
    global g_shm
    try:
        if g_shm is not None:
            g_shm.close()
            g_shm.unlink()
            print("Unlinked shared memory")
    except Exception as e:
        print(f"Error cleaning up: {e}")

atexit.register(cleanup_shared_memory)

def signal_handler(sig, frame):
    cleanup_shared_memory()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

class CameraSingleSlot(Node):
    """ROS2 node that writes latest camera frame to single shared memory slot."""
    
    def __init__(self, camera_topic: str = "/camera/image_raw"):
        super().__init__("camera_single_slot")
        
        self.bridge = CvBridge()
        
        # Clean up existing shared memory
        try:
            shared_memory.SharedMemory(name=SHM_NAME).unlink()
        except FileNotFoundError:
            pass
        except Exception as e:
            self.get_logger().warning(f"Error cleaning existing memory: {e}")
        
        # Create shared memory: [timestamp:8][frame_data:H*W*3]
        frame_size = H * W * 3
        total_size = 8 + frame_size  # timestamp + frame
        
        self.shm = shared_memory.SharedMemory(
            create=True,
            size=total_size,
            name=SHM_NAME
        )
        
        # Set global reference for cleanup
        global g_shm
        g_shm = self.shm
        
        # Create subscription
        self.subscription = self.create_subscription(
            Image,
            camera_topic,
            self.camera_callback,
            10
        )
        
        self.get_logger().info(f"Camera single slot ready: {SHM_NAME}")
    
    def camera_callback(self, msg: Image) -> None:
        """Write latest frame to shared memory."""
        try:
            # Convert and resize
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            resized = cv2.resize(cv_image, (W, H))
            
            # Write timestamp first
            timestamp = time.time()
            struct.pack_into('<d', self.shm.buf, 0, timestamp)
            
            # Write frame data
            frame_bytes = resized.tobytes()
            self.shm.buf[8:8+len(frame_bytes)] = frame_bytes
            
        except Exception as e:
            self.get_logger().error(f"Error writing frame: {e}")
    
    def destroy_node(self):
        """Clean up on destruction."""
        try:
            if hasattr(self, 'shm'):
                self.shm.close()
                self.shm.unlink()
            global g_shm
            g_shm = None
        except Exception as e:
            self.get_logger().error(f"Error in cleanup: {e}")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CameraSingleSlot()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
    
