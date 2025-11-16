
#!/usr/bin/env python3
"""
ROS2 node for writing camera frames to a single shared memory slot.
Consumer (inference.py) reads a 320x320 RGB frame and treats it as
a ~60° pinhole view: we take the center 600x600 crop from a 1600x600
camera image, then downsample to 320x320.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from multiprocessing import shared_memory
import numpy as np
import cv2
import struct
import atexit
import signal
import sys
from typing import Optional

# Configuration: inference expects 320x320 RGB
H, W = 320, 320
SHM_NAME = "camera_latest"

# Global for cleanup
g_shm: Optional[shared_memory.SharedMemory] = None


def cleanup_shared_memory() -> None:
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


def signal_handler(sig, frame) -> None:
    cleanup_shared_memory()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def crop_center_square(image: np.ndarray) -> np.ndarray:
    """
    From a 600x1600 RGB image, extract the central 600x600 square:

    - Input:  image shape [H_in, W_in, 3], e.g. [600, 1600, 3]
    - Output: image[:, 500:1100, :] -> [600, 600, 3]

    This approximates reducing HFOV from ~160° to ~60° by keeping
    the central portion of a rectilinear camera.
    """
    h, w, _ = image.shape
    # Ensure height <= width so we can take a central square of size h
    side = min(h, w)
    start_x = (w - side) // 2
    end_x = start_x + side
    return image[:, start_x:end_x, :]


class CameraSingleSlot(Node):
    """ROS2 node that writes latest camera frame to single shared memory slot."""

    def __init__(self, camera_topic: str = "/camera/image_raw") -> None:
        super().__init__("camera_single_slot")

        self.bridge = CvBridge()

        # Clean up any existing shared memory with same name
        try:
            shared_memory.SharedMemory(name=SHM_NAME).unlink()
        except FileNotFoundError:
            pass
        except Exception as e:
            self.get_logger().warning(f"Error cleaning existing memory: {e}")

        # Create shared memory: [timestamp:8][frame_data:H*W*3]
        frame_size = H * W * 3
        total_size = 8 + frame_size  # timestamp + frame bytes

        self.shm = shared_memory.SharedMemory(
            create=True,
            size=total_size,
            name=SHM_NAME,
        )

        # Set global reference for cleanup
        global g_shm
        g_shm = self.shm

        # Subscribe to camera topic
        self.subscription = self.create_subscription(
            Image,
            camera_topic,
            self.camera_callback,
            10,
        )

        self.get_logger().info(
            f"Camera single slot ready on {camera_topic}, SHM={SHM_NAME}"
        )

    def camera_callback(self, msg: Image) -> None:
        """Convert, center-crop, downsample, and write latest frame to shared memory."""
        try:
            # 1. ROS Image -> RGB NumPy (H_in, W_in, 3)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

            # Expecting something like 600x1600; take center 600x600
            center_square = crop_center_square(cv_image)

            # 2. Downsample 600x600 -> 320x320
            resized = cv2.resize(
                center_square,
                (W, H),
                interpolation=cv2.INTER_AREA,
            )

            # 3. Write timestamp (double, little-endian)
            timestamp = self.get_clock().now().nanoseconds / 1e9
            struct.pack_into("<d", self.shm.buf, 0, timestamp)

            # 4. Write frame data
            frame_bytes = resized.tobytes()
            self.shm.buf[8:8 + len(frame_bytes)] = frame_bytes

        except Exception as e:
            self.get_logger().error(f"Error writing frame: {e}")

    def destroy_node(self) -> None:
        """Clean up on destruction."""
        try:
            if hasattr(self, "shm"):
                self.shm.close()
                self.shm.unlink()
            global g_shm
            g_shm = None
        except Exception as e:
            self.get_logger().error(f"Error in cleanup: {e}")
        super().destroy_node()


def main(args=None) -> None:
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
