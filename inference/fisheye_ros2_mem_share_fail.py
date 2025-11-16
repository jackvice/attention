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


# ---- geometry: HFOV -> intrinsics (rectilinear) ----
def intrinsics_from_hfov(width: int, height: int, hfov_rad: float) -> np.ndarray:
    """Pinhole intrinsics K from HFOV; square pixels."""
    fx = width / (2.0 * np.tan(hfov_rad / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[fx, 0, cx],
                     [0,  fy, cy],
                     [0,   0,  1]], dtype=np.float32)



# ---- build a yaw-specific rectilinear->rectilinear remap ----
def make_yaw_lut_rect_to_rect(
    src_size: Tuple[int, int],      # (W_src, H_src) AFTER CROP
    dst_hw: int,                    # 320 for 320x320
    hfov_src: float,                # radians (2.8)
    hfov_win: float,                # radians (np.deg2rad(60))
    yaw_deg: float,                 # window center yaw
    crop_top_px: int                # how many rows were removed from the top
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Remap a local 60° pinhole view (dst) into the wide source rectilinear image (src).
    Returns (map_x, map_y) for cv2.remap.
    """
    Wc, Hc = src_size  # cropped source size
    assert Wc > 0 and Hc > 0

    # Source intrinsics for the ORIGINAL full frame, then adjust for top crop.
    # Full height = Hc + crop_top_px
    K_src_full = intrinsics_from_hfov(Wc, Hc + crop_top_px, hfov_src)
    K_src = adjust_K_for_crop(K_src_full, crop_top_px)
    fx_s, fy_s, cx_s, cy_s = K_src[0,0], K_src[1,1], K_src[0,2], K_src[1,2]


    Wd = Hd = int(dst_hw)
    K_dst = intrinsics_from_hfov(Wd, Hd, hfov_win)
    fx_d, fy_d, cx_d, cy_d = K_dst[0,0], K_dst[1,1], K_dst[0,2], K_dst[1,2]

    # Meshgrid of destination pixels
    xi, yi = np.meshgrid(np.arange(Wd, dtype=np.float32),
                         np.arange(Hd, dtype=np.float32))

    # Rays in target camera (before yaw)
    x = (xi - cx_d) / fx_d
    y = (yi - cy_d) / fy_d
    z = np.ones_like(x)
    # Normalize rays (optional; not strictly needed for pinhole projection)
    norm = np.sqrt(x*x + y*y + z*z)
    x /= norm; y /= norm; z /= norm

    # Rotate by yaw around the camera vertical axis (Y axis)
    psi = np.deg2rad(yaw_deg)
    c, s = np.cos(psi), np.sin(psi)
    # R_yaw * [x, y, z]
    xr =  c * x + s * z
    yr =  y
    zr = -s * x + c * z

    # Project into source image
    # If zr <= 0, mark outside (behind camera)
    eps = 1e-6
    zr = np.maximum(zr, eps)
    u = fx_s * (xr / zr) + cx_s
    v = fy_s * (yr / zr) + cy_s

    return u.astype(np.float32), v.astype(np.float32)


def build_lut_bank(
        src_size: Tuple[int, int],   # (Wc, Hc) after crop
        hfov_src: float,
        yaws_deg: Tuple[float, ...],
        hfov_win_deg: float = 60.0,
        out_hw: int = 320,
        crop_top_px: int = 0
) -> Dict[int, Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]]:
    """Precompute remap LUTs for each yaw bin."""
    bank: Dict[int, Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]] = {}
    for idx, yaw in enumerate(yaws_deg):
        mx, my = make_yaw_lut_rect_to_rect(
            src_size=src_size,
            dst_hw=out_hw,
            hfov_src=hfov_src,
            hfov_win=np.deg2rad(hfov_win_deg),
            yaw_deg=yaw,
            crop_top_px=crop_top_px
        )
        bank[idx] = (mx, my)
    return bank


def extract_view_with_lut(src_cropped: ImageRGB, lut: Tuple[npt.NDArray[np.float32],
                                                            npt.NDArray[np.float32]]) -> ImageRGB:
    """Apply precomputed (map_x, map_y) to the cropped source image."""
    map_x, map_y = lut
    return cv2.remap(src_cropped, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)



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
        self.lut_bank = None
        self.crop_top_px = None

        
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
        """Write latest rectified frame to shared memory."""
        try:
            # 1. Convert ROS → RGB NumPy
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

            # 2. Crop top 1/3 (removes sky)
            cropped = crop_top_fraction(cv_image)

            # 3. Build LUT once on first frame
            if self.lut_bank is None:
                crop_top_px = int(round(cv_image.shape[0] * top_frac))
                src_size = (cropped.shape[1], cropped.shape[0])   # (Wc, Hc)

                self.lut_bank = build_lut_bank(
                    src_size=src_size,
                    hfov_src=2.8,            # SDF camera HFOV
                    yaws_deg=(0.0,),         # center yaw only
                    out_hw= 320,              # rectified intermediate resolution
                    crop_top_px=crop_top_px
                )

            # 4. Rectify with LUT → 144×144
            rectified_320 = extract_view_with_lut(cropped, self.lut_bank[0])

            # 5. Resize rectified → 320×320
            rectified_320 = cv2.resize(
                rectified_320,
                (W, H),
                interpolation=cv2.INTER_AREA
            )

            # 6. Write timestamp
            timestamp = self.get_clock().now().nanoseconds / 1e9
            struct.pack_into('<d', self.shm.buf, 0, timestamp)

            # 7. Write final image to shared memory
            frame_bytes = rectified_320.tobytes()
            self.shm.buf[8:8 + len(frame_bytes)] = frame_bytes

        except Exception as e:
            self.get_logger().error(f"Error writing frame: {e}")


        
    def camera_callback(self, msg: Image) -> None:
        """Write latest frame to shared memory."""
        try:
            # Convert and resize
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

            cropped = crop_top_fraction(cv_image)
    
            if self.lut_bank is None:
                compute crop_top_px
                build LUT bank for yaw=0
                
            rectified = extract_view_with_lut(...)
    
            resized = cv2.resize(cv_image, (W, H))
            
            # Write timestamp first
            #timestamp = time.time() # wall clock
            timestamp = self.get_clock().now().nanoseconds / 1e9 # gazebo
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
    
