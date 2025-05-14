import os
import glob

import cv2
import numpy as np
import numpy as np
import jax.numpy as jnp
import cv2
import time
import hashlib, json, pickle
import logging
import numpy.typing as npt
from typing import NamedTuple, List, Tuple, Dict, Optional, Callable, Iterator, Any, Union
from pathlib import Path
from functools import partial
from typing import List
from config_temporal import FUTURE_OFFSET_F, PAST_OFFSETS_F, CAM_ID, SIGMA_PX

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("trajectory_prediction")

# Core data structures
class Frame(NamedTuple):
    """Single video frame with metadata"""
    path: str
    frame_id: int
    sequence_id: str
    camera_id: str


class Pedestrian(NamedTuple):
    """Detected pedestrian in a frame"""
    position: np.ndarray  # [x, y] center position
    bbox: np.ndarray  # [x1, y1, x2, y2] bounding box
    mask: Optional[np.ndarray]  # Binary mask if available
    confidence: float



class TrajectorySequence(NamedTuple):
    frames: List[Frame]
    pedestrians: List[List[Pedestrian]]
    trajectories: List[np.ndarray]
    future_pedestrians: List[Pedestrian] = []
    future_frame: Optional[Frame] = None  # Store the future frame itself
    

class ModelConfig(NamedTuple):
    """Configuration for spatiotemporal attention model"""
    embedding_dim: int = 256
    num_heads: int = 8
    dropout_rate: float = 0.1
    feature_dim: int = 64
    max_len: int = 5000
    sequence_length: int = 5
    output_height: int = 320  # New parameter
    output_width: int = 320   # New parameter


# trajectory_cache.py
#from __future__ import annotations



def _sha1(buf: bytes) -> str:
    h = hashlib.sha1()
    h.update(buf)
    return h.hexdigest()


def create_fused_observation(
    rgb_frame: np.ndarray,
    depth_frame: np.ndarray,
    trajectories: List[np.ndarray],
    confidences: List[float],
    target_height: int,
    target_width: int
) -> np.ndarray:
    """
    Create a 3-channel observation for the RL agent.
    
    Args:
        rgb_frame: RGB image [H, W, 3] with values in [0, 1]
        depth_frame: Depth image [H, W] with normalized values
        trajectories: List of predicted trajectories, each [T, 2]
        confidences: Confidence scores for each trajectory
        target_height: Height of the output tensor
        target_width: Width of the output tensor
        
    Returns:
        3-channel observation [H, W, 3] with:
            Channel 0: Grayscale image
            Channel 1: Traversability heatmap
            Channel 2: Depth map
    """
    # Convert RGB to grayscale
    grayscale = np.mean(rgb_frame, axis=2)
    
    # Create traversability heatmap
    heatmap = create_traversability_heatmap(
        trajectories,
        confidences,
        height=target_height,
        width=target_width
    )
    
    # Ensure depth map has the right shape and is normalized
    if depth_frame.shape != (target_height, target_width):
        depth_frame = cv2.resize(depth_frame, (target_width, target_height))
    
    if np.max(depth_frame) > 1.0:
        depth_frame = depth_frame / np.max(depth_frame)
    
    # Stack the channels
    observation = np.stack([grayscale, heatmap, depth_frame], axis=2)
    
    return observation


def _dataset_signature(dataset_root: str, params: Dict[str, Any]) -> str:
    """
    Finger-print every PNG’s *path* + *mtime* + the params that influence
    pedestrian extraction / tracking.  If you touch a file or change a
    parameter the hash changes → cache miss.
    """
    root = Path(dataset_root)
    parts: list[bytes] = []

    for p in sorted(root.rglob("*.png")):
        stat = p.stat()
        parts.append(str(p).encode())               # path
        parts.append(str(stat.st_mtime_ns).encode())  # last-modified ns

    parts.append(json.dumps(params, sort_keys=True).encode())
    return _sha1(b"".join(parts))


    
def detect_pedestrians_yolo_onnx(
    image: np.ndarray,
    onnx_path: str = "yolo11n.onnx",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    person_class_id: int = 0,  # Typically person is class 0 in YOLO models
    session = None,
) -> List[Pedestrian]:
    """
    Detect pedestrians using YOLOv11n ONNX model.
    
    Args:
        image: Input RGB image [H,W,3] with values in [0,1]
        onnx_path: Path to ONNX model
        conf_threshold: Confidence threshold for detections
        iou_threshold: IOU threshold for NMS
        person_class_id: Class ID for person in the model
        session: Cached ONNX session (optional)
        
    Returns:
        List of Pedestrian objects and session
    """
    import onnxruntime as ort
    
    # Create session if not provided
    if session is None:
        session = ort.InferenceSession(
            onnx_path, 
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
    
    # Get input name
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    
    # Prepare image
    img = (image * 255).astype(np.uint8)
    H, W = img.shape[:2]
    
    # Preprocess for YOLO
    input_size = (640, 640)  # Standard YOLO input size
    img_resized = cv2.resize(img, input_size)
    
    # Normalize
    img_input = img_resized.astype(np.float32) / 255.0
    img_input = np.transpose(img_input, (2, 0, 1))  # HWC to CHW
    img_input = np.expand_dims(img_input, axis=0)  # Add batch dimension
    
    # Run inference
    outputs = session.run(output_names, {input_name: img_input})
    
    # Process YOLO output
    predictions = outputs[0]  # Shape: (1, 84, 8400)
    
    # Transpose to make shape (1, 8400, 84)
    predictions = np.transpose(predictions, (0, 2, 1))
    
    # Extract data from predictions
    boxes = predictions[0, :, :4]  # (8400, 4) - (x, y, w, h)
    scores = predictions[0, :, 4:]  # (8400, 80) - class scores
    
    # Get class IDs and confidence scores
    class_scores = np.max(scores, axis=1)  # Maximum class score for each box
    class_ids = np.argmax(scores, axis=1)  # Class ID with maximum score
    
    # Filter for person class
    person_mask = (class_ids == person_class_id) & (class_scores > conf_threshold)
    
    filtered_boxes = boxes[person_mask]
    filtered_scores = class_scores[person_mask]
    
    # Apply non-max suppression
    keep_indices = []
    if len(filtered_boxes) > 0:
        # Convert boxes from (x, y, w, h) to (x1, y1, x2, y2)
        x = filtered_boxes[:, 0]
        y = filtered_boxes[:, 1]
        w = filtered_boxes[:, 2]
        h = filtered_boxes[:, 3]
        
        x1 = x - w/2
        y1 = y - h/2
        x2 = x + w/2
        y2 = y + h/2
        
        # Perform NMS
        areas = w * h
        order = filtered_scores.argsort()[::-1]
        
        while order.size > 0:
            i = order[0]
            keep_indices.append(i)
            
            # Compute IoU
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w_inter = np.maximum(0.0, xx2 - xx1)
            h_inter = np.maximum(0.0, yy2 - yy1)
            inter = w_inter * h_inter
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
    
    # Create pedestrian objects
    pedestrians = []
    
    for idx in keep_indices:
        # Get box coordinates
        x, y, w, h = filtered_boxes[idx]
        
        # Scale back to original image
        x_orig = int(x * W / input_size[0])
        y_orig = int(y * H / input_size[1])
        w_orig = int(w * W / input_size[0])
        h_orig = int(h * H / input_size[1])
        
        # Calculate corners
        x1 = int(x_orig - w_orig/2)
        y1 = int(y_orig - h_orig/2)
        x2 = int(x_orig + w_orig/2)
        y2 = int(y_orig + h_orig/2)
        
        # Create pedestrian object
        pedestrian = Pedestrian(
            position=np.array([x_orig, y_orig]),
            bbox=np.array([x1, y1, x2, y2]),
            mask=None,  # We'd need segmentation output for this
            confidence=float(filtered_scores[idx])
        )
        pedestrians.append(pedestrian)
    
    return pedestrians, session


def visualize_and_save_detections(
    image: np.ndarray,
    pedestrians: List[Pedestrian],
    output_path: str = "people.png",
    show_masks: bool = True
) -> None:
    """
    Visualize detected pedestrians on an image and save to file.
    
    Args:
        image: Input RGB image [H,W,3] with values in [0,1]
        pedestrians: List of detected pedestrians
        output_path: Path to save the visualization
        show_masks: Whether to show masks or just bounding boxes
    """
    # Convert to uint8 for OpenCV
    vis_img = (image * 255).astype(np.uint8).copy()
    
    # Draw pedestrians
    for ped in pedestrians:
        # Draw bounding box
        x1, y1, x2, y2 = ped.bbox.astype(int)
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add confidence text
        text = f"{ped.confidence:.2f}"
        cv2.putText(
            vis_img, text, (x1, y1 - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )
        
        # Draw center position
        center_x, center_y = ped.position.astype(int)
        cv2.circle(vis_img, (center_x, center_y), 4, (255, 0, 0), -1)
        
        # Optionally show masks
        if show_masks and ped.mask is not None:
            mask_colored = np.zeros_like(vis_img)
            mask_colored[ped.mask] = (0, 0, 255)
            alpha = 0.3
            vis_img = cv2.addWeighted(vis_img, 1, mask_colored, alpha, 0)
    
    # Save image
    cv2.imwrite(output_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
    print(f"Saved visualization to {output_path}")

    
# Dataset utilities
def extract_frame_info(file_path: str) -> Optional[Frame]:
    """
    Extract frame information from file path.
    
    Args:
        file_path: Path to frame image file
        
    Returns:
        Frame object with metadata or None if parsing fails
    """
    try:
        path_obj = Path(file_path)
        
        # Extract frame_id from filename (remove extension and convert to int)
        try:
            frame_id = int(path_obj.stem)
        except ValueError:
            logger.warning(f"Could not parse frame_id from {file_path}")
            return None
        
        # Extract camera_id from parent directory structure
        # Assuming path like: .../cam_img/1/data_rgb/1.png where 1 is camera_id
        parts = path_obj.parts
        cam_idx = -1
        for i, part in enumerate(parts):
            if part == "cam_img":
                cam_idx = i
                break
        
        if cam_idx >= 0 and cam_idx + 1 < len(parts):
            camera_id = parts[cam_idx + 1]
        else:
            camera_id = "unknown"
        
        # Extract sequence_id from directory structure
        # Try to find identifiable sequence markers (outdoor_1, cafe_1, etc.)
        sequence_markers = ["outdoor", "cafe", "courtyard", "crossroad", "three_way", "subway"]
        sequence_id = "unknown"
        
        for part in parts:
            for marker in sequence_markers:
                if marker in part and any(c.isdigit() for c in part):
                    sequence_id = part
                    break
            if sequence_id != "unknown":
                break
        
        return Frame(
            path=file_path,
            frame_id=frame_id,
            sequence_id=sequence_id,
            camera_id=camera_id
        )
    
    except Exception as e:
        logger.error(f"Error parsing frame from {file_path}: {e}")
        return None


def scan_dataset(
    root_path: str, 
    max_per_sequence: Optional[int] = None
) -> Dict[Tuple[str, str], List[Frame]]:
    logger.info(f"Scanning dataset at {root_path}")
    start_time = time.time()
    
    # Find all PNG files recursively
    pattern = os.path.join(root_path, "**", "*.png")
    all_files = glob.glob(pattern, recursive=True)
    logger.info(f"Found {len(all_files)} PNG files")
    
    # Process files to extract metadata
    cameras: Dict[Tuple[str, str], List[Frame]] = {}
    for file_path in all_files:
        frame = extract_frame_info(file_path)
        if frame is not None:
            # Group by camera_id regardless of sequence_id
            camera_key = (frame.sequence_id, frame.camera_id)   # tuple
            if camera_key not in cameras:
                cameras[camera_key] = []
            cameras[camera_key].append(frame)

    # Create a new dictionary to store the limited frames
    limited_cameras = {}
    
    # ---- scan_dataset() / logging block ----
    for (seq_id, camera_id), frames in cameras.items():
        frames.sort(key=lambda f: f.frame_id)
        logger.info(
            f"Seq {seq_id}  Cam {camera_id}: Sorted {len(frames)} frames"
        )

        # Print first few frame IDs to verify sorting
        if frames:
            logger.info(f"Camera {camera_id}: First 5 frame IDs = {[f.frame_id for f in frames[:5]]}")
        
        # Optionally limit the number of frames per camera
        if max_per_sequence is not None:
            limited_frames = frames[:max_per_sequence]
            limited_cameras[(seq_id, camera_id)] = limited_frames
        else:
            limited_cameras[(seq_id, camera_id)] = frames
    
    elapsed = time.time() - start_time
    logger.info(f"Dataset scanning completed in {elapsed:.2f} seconds")
    logger.info(f"Found {len(limited_cameras)} cameras")
    
    return limited_cameras
    

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


def compute_trajectories(
    frame_sequences: List[Tuple[List[Frame], Frame]],
    detect_fn: Callable[[np.ndarray], List[Pedestrian]],
    target_width: int = 320,
    target_height: int = 320,
    min_track_length: int = 3,
    yolo_model_path: str = "yolo11n.onnx"
) -> List[TrajectorySequence]:
    """
    Compute pedestrian trajectories from frame sequences and generate future ground truth.
    """
    trajectory_sequences = []
    
    # Initialize YOLO session once for reuse
    import onnxruntime as ort
    yolo_session = None
    
    # Check if detect_fn is a YOLO detection function
    is_yolo_detect = hasattr(detect_fn, '__code__') and 'session' in detect_fn.__code__.co_varnames
    
    if is_yolo_detect:
        yolo_session = ort.InferenceSession(
            yolo_model_path, 
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
    
    for seq_idx, (frame_sequence, future_frame) in enumerate(frame_sequences):
        # Process each frame in the sequence
        all_pedestrians = []
        
        for frame in frame_sequence:
            # Load and preprocess frame
            img = load_and_preprocess_frame(
                frame.path, 
                target_width=target_width,
                target_height=target_height
            )
            
            # Detect pedestrians
            if is_yolo_detect:
                pedestrians, yolo_session = detect_pedestrians_yolo_onnx(img,
                                                                         session=yolo_session)
            else:
                pedestrians = detect_fn(img)
                
            all_pedestrians.append(pedestrians)
        
        # Process future frame for ground truth
        future_img = load_and_preprocess_frame(
            future_frame.path,
            target_width=target_width,
            target_height=target_height
        )
        
        # Detect pedestrians in future frame
        if is_yolo_detect:
            future_pedestrians, yolo_session = detect_pedestrians_yolo_onnx(future_img,
                                                                            session=yolo_session)
        else:
            future_pedestrians = detect_fn(future_img)
        
        # Skip sequences with no pedestrians in the future frame
        if len(future_pedestrians) == 0:
            continue
        
        # Track pedestrians across frames
        trajectories = track_pedestrians_simple(
            frame_sequence,
            all_pedestrians,
            min_track_length=min_track_length
        )

        traj_seq = TrajectorySequence(
            frames=frame_sequence,
            pedestrians=all_pedestrians,
            trajectories=trajectories,
            future_pedestrians=future_pedestrians,
            future_frame=future_frame
        )
        
        trajectory_sequences.append(traj_seq)
    
    return trajectory_sequences

    
def compute_trajectories_old(
    frame_sequences: List[Tuple[List[Frame], Frame]],  # Changed from List[List[Frame]]
    detect_fn: Callable[[np.ndarray], List[Pedestrian]],
    target_width: int = 320,
    target_height: int = 320,
    min_track_length: int = 3,
    yolo_model_path: str = "yolo11n.onnx"
) -> List[TrajectorySequence]:

    """
    Compute pedestrian trajectories from frame sequences and generate future ground truth.
    """
    trajectory_sequences = []
    
    # Initialize YOLO session once for reuse
    import onnxruntime as ort
    yolo_session = None
    
    # Check if detect_fn is a YOLO detection function
    is_yolo_detect = hasattr(detect_fn, '__code__') and 'session' in detect_fn.__code__.co_varnames
    
    if is_yolo_detect:
        yolo_session = ort.InferenceSession(
            yolo_model_path, 
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
    
    for seq_idx, (frame_sequence, future_frame) in enumerate(frame_sequences):
        # Process each frame in the sequence
        all_pedestrians = []
        
        for frame in frame_sequence:
            # Load and preprocess frame
            img = load_and_preprocess_frame(
                frame.path, 
                target_width=target_width,
                target_height=target_height
            )
            
            # Detect pedestrians
            if is_yolo_detect:
                pedestrians, yolo_session = detect_pedestrians_yolo_onnx(img,
                                                                         session=yolo_session)
            else:
                pedestrians = detect_fn(img)
                
            all_pedestrians.append(pedestrians)
        
        # Process future frame for ground truth
        future_img = load_and_preprocess_frame(
            future_frame.path,
            target_width=target_width,
            target_height=target_height
        )
        
        # Detect pedestrians in future frame
        if is_yolo_detect:
            future_pedestrians, yolo_session = detect_pedestrians_yolo_onnx(future_img,
                                                                            session=yolo_session)
        else:
            future_pedestrians = detect_fn(future_img)
        
        # Track pedestrians across frames
        trajectories = track_pedestrians_simple(
            frame_sequence,
            all_pedestrians,
            min_track_length=min_track_length
        )

        traj_seq = TrajectorySequence(
            frames=frame_sequence,
            pedestrians=all_pedestrians,
            trajectories=trajectories,
            future_pedestrians=future_pedestrians,
            future_frame=future_frame  # Add this line
        )
        
        trajectory_sequences.append(traj_seq)
    
    return trajectory_sequences
    

def create_target_heatmap_from_pedestrians(
    pedestrians: List[Pedestrian],
    target_height: int = 320,
    target_width: int = 320,
    sigma: float = SIGMA_PX
) -> np.ndarray:
    """
    Create a target heatmap from detected pedestrians in future frame.
    
    Args:
        pedestrians: List of pedestrians in future frame
        target_height: Output height
        target_width: Output width
        sigma: Spread parameter for Gaussian
        
    Returns:
        Heatmap of shape [H, W, 1]
    """
    heatmap = np.zeros((target_height, target_width), dtype=np.float32)
    
    for ped in pedestrians:
        # Use center position of the pedestrian
        x, y = ped.position.astype(int)
        
        # Ensure the position is within bounds
        if 0 <= x < target_width and 0 <= y < target_height:
            # Add a Gaussian centered at this position
            y_indices, x_indices = np.mgrid[:target_height, :target_width]
            gaussian = np.exp(-((x_indices - x)**2 + (y_indices - y)**2) / (2 * sigma**2))
            
            # Accumulate to the heatmap using maximum to avoid damping overlapping Gaussians
            heatmap = np.maximum(heatmap, gaussian)
    
    # Normalize to [0, 1]
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)
    
    return heatmap[..., np.newaxis]  # Add channel dimension


def create_traversability_heatmap(
    predicted_trajectories: List[np.ndarray],
    confidence_scores: List[float],
    height: int,
    width: int,
    sigma: float = 5.0,
    time_decay: float = 0.9
) -> np.ndarray:
    """
    Create a traversability heatmap from multiple predicted trajectories.
    
    Args:
        predicted_trajectories: List of trajectories, each shape [T, 2] with x,y coords
        confidence_scores: Confidence in each trajectory prediction
        height, width: Dimensions of the output heatmap
        sigma: Spatial spread of each person's influence
        time_decay: How quickly future predictions decay in influence
        
    Returns:
        Heatmap of shape [height, width] with values between 0-1
    """
    heatmap = np.zeros((height, width), dtype=np.float32)
    
    for trajectory, confidence in zip(predicted_trajectories, confidence_scores):
        for t, (x, y) in enumerate(trajectory):
            # Skip if position is outside the map
            if not (0 <= x < width and 0 <= y < height):
                continue
                
            # Create a decaying weight based on time step
            weight = confidence * (time_decay ** t)
            
            # Add a Gaussian centered at this position
            y_indices, x_indices = np.mgrid[:height, :width]
            gaussian = weight * np.exp(
                -((x_indices - x)**2 + (y_indices - y)**2) / (2 * sigma**2)
            )
            
            # Accumulate to the heatmap
            heatmap += gaussian
    
    # Normalize the heatmap to [0, 1]
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)
        
    return heatmap


def track_pedestrians_simple(
    frames: List[Frame],
    frame_pedestrians: List[List[Pedestrian]],
    min_track_length: int = 3,
    max_distance: float = 50.0  # Maximum distance for matching
) -> List[np.ndarray]:
    """
    Track pedestrians across frames using a simple distance-based approach.
    
    Args:
        frames: List of frames in sequence
        frame_pedestrians: List of pedestrian lists for each frame
        min_track_length: Minimum number of frames a pedestrian must appear in
        max_distance: Maximum distance for matching pedestrians between frames
        
    Returns:
        List of trajectory arrays [T, 2]
    """
    seq_len = len(frames)
    
    # Initialize track_id -> position mapping
    tracks = {}
    next_track_id = 0
    
    # Process first frame
    if frame_pedestrians and frame_pedestrians[0]:
        for ped_idx, ped in enumerate(frame_pedestrians[0]):
            # Initialize track with position for first frame
            tracks[next_track_id] = {
                "positions": [None] * seq_len,
                "boxes": [None] * seq_len
            }
            tracks[next_track_id]["positions"][0] = ped.position
            tracks[next_track_id]["boxes"][0] = ped.bbox
            next_track_id += 1
    
    # Process subsequent frames
    for frame_idx in range(1, seq_len):
        curr_pedestrians = frame_pedestrians[frame_idx]
        if not curr_pedestrians:
            continue
            
        # Get active tracks from previous frame
        active_tracks = {}
        for track_id, track_data in tracks.items():
            if track_data["positions"][frame_idx - 1] is not None:
                active_tracks[track_id] = track_data
        
        # Match current pedestrians with active tracks
        matched_ped_indices = set()
        
        for track_id, track_data in active_tracks.items():
            prev_pos = track_data["positions"][frame_idx - 1]
            
            # Find closest pedestrian
            best_ped_idx = -1
            best_distance = max_distance
            
            for ped_idx, ped in enumerate(curr_pedestrians):
                if ped_idx in matched_ped_indices:
                    continue
                
                # Calculate distance
                distance = np.linalg.norm(ped.position - prev_pos)
                
                if distance < best_distance:
                    best_distance = distance
                    best_ped_idx = ped_idx
            
            # If found a match, update track
            if best_ped_idx >= 0:
                matched_ped_indices.add(best_ped_idx)
                ped = curr_pedestrians[best_ped_idx]
                
                track_data["positions"][frame_idx] = ped.position
                track_data["boxes"][frame_idx] = ped.bbox
        
        # Create new tracks for unmatched pedestrians
        for ped_idx, ped in enumerate(curr_pedestrians):
            if ped_idx not in matched_ped_indices:
                # Initialize new track
                tracks[next_track_id] = {
                    "positions": [None] * seq_len,
                    "boxes": [None] * seq_len
                }
                tracks[next_track_id]["positions"][frame_idx] = ped.position
                tracks[next_track_id]["boxes"][frame_idx] = ped.bbox
                next_track_id += 1
    
    # Convert tracks to trajectory format
    trajectories = []
    for track_id, track_data in tracks.items():
        # Count valid positions
        valid_positions = sum(1 for pos in track_data["positions"] if pos is not None)
        
        if valid_positions >= min_track_length:
            # Create trajectory array
            traj = np.zeros((seq_len, 2))
            for i, pos in enumerate(track_data["positions"]):
                if pos is not None:
                    traj[i] = pos
                elif i > 0 and track_data["positions"][i-1] is not None:
                    # Fill small gaps with previous position
                    traj[i] = traj[i-1]
            
            trajectories.append(traj)
    
    return trajectories


def bbox_to_mask(bbox: npt.NDArray[np.int_],
                 h: int,
                 w: int) -> npt.NDArray[np.float32]:
    """Return a single-pedestrian binary mask from a bbox."""
    x1, y1, x2, y2 = bbox.astype(int)
    m: npt.NDArray[np.float32] = np.zeros((h, w, 1), np.float32)
    m[y1:y2, x1:x2, 0] = 1.0
    return m


def create_masks_from_pedestrians(
    pedestrians: List[Pedestrian],
    height: int,
    width: int
) -> np.ndarray:
    """Return H×W×1 binary mask (float32) for all pedestrians."""
    acc = np.zeros((height, width), dtype=bool)          # <- bool buffer

    for ped in pedestrians:
        if ped.mask is not None:
            acc |= ped.mask.astype(bool)
        else:
            acc |= bbox_to_mask(ped.bbox, height, width)[:, :, 0].astype(bool)

    return acc.astype(np.float32)[..., None]             # back to float32



def write_debug_images(
    rgb_frames: np.ndarray,
    mask_frames: np.ndarray,
    prediction_heatmap: np.ndarray,
    epoch: int,
    output_dir: str = "./out_images",
    target_heatmap: Optional[np.ndarray] = None,
    future_frame: Optional[np.ndarray] = None,
    future_pedestrians: Optional[List[Pedestrian]] = None
) -> None:
    """
    Write input frames and prediction heatmap to image files for debugging.
    Shows how model predictions evolve during training.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Force conversion to NumPy arrays
    rgb_frames_np = np.asarray(rgb_frames)
    mask_frames_np = np.asarray(mask_frames)
    prediction_heatmap_np = np.asarray(prediction_heatmap)
    
    # Create a subdirectory for this epoch to keep things organized
    epoch_dir = os.path.join(output_dir, f"epoch_{epoch}")
    #epoch_dir = os.path.join(output_dir, f"epoch_test")
    os.makedirs(epoch_dir, exist_ok=True)
    
    # Write input RGB frames with timestamps
    for t, frame in enumerate(rgb_frames_np):
        # Convert from float [0,1] to uint8 [0,255]
        rgb_img = (frame * 255).astype(np.uint8)
        # Convert RGB to BGR for OpenCV
        bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        
        # Add timestamp text
        time_offset = PAST_OFFSETS_F[t] / 10.0 if t < len(PAST_OFFSETS_F) else 0  # Convert to seconds assuming 10 FPS
        cv2.putText(
            bgr_img, 
            f"t={time_offset:.1f}s", 
            (20, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (255, 255, 255), 
            2
        )
        
        # Save to file
        output_path = os.path.join(epoch_dir, f"input_frame_{t+1}.png")
        cv2.imwrite(output_path, bgr_img)
    
    # Write the predicted heatmap
    heatmap_img = (prediction_heatmap_np[..., 0] * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
    
    # Add timestamp text for prediction
    cv2.putText(
        heatmap_colored, 
        f"t=+{FUTURE_OFFSET_F/10.0:.1f}s (prediction)", 
        (20, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (255, 255, 255), 
        2
    )
    
    # Save prediction heatmap
    pred_path = os.path.join(epoch_dir, f"prediction_heatmap.png")
    cv2.imwrite(pred_path, heatmap_colored)
    
    # If target heatmap is provided, show that too
    if target_heatmap is not None:
        target_np = np.asarray(target_heatmap)
        target_img = (target_np[..., 0] * 255).astype(np.uint8)
        target_colored = cv2.applyColorMap(target_img, cv2.COLORMAP_JET)
        
        # Add timestamp text for ground truth
        cv2.putText(
            target_colored, 
            f"t=+{FUTURE_OFFSET_F/10.0:.1f}s (ground truth)", 
            (20, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (255, 255, 255), 
            2
        )
        
        # Save target heatmap
        target_path = os.path.join(epoch_dir, f"target_heatmap.png")
        cv2.imwrite(target_path, target_colored)
        
        # Create a comparison image (side by side)
        
        h, w = target_img.shape
        comparison = np.zeros((h, w*2, 3), dtype=np.uint8)
        comparison[:, :w] = target_colored
        comparison[:, w:] = heatmap_colored
        
        # Add labels
        cv2.putText(
            comparison, 
            "Ground Truth", 
            (20, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (255, 255, 255), 
            2
        )
        cv2.putText(
            comparison, 
            "Prediction", 
            (w + 20, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (255, 255, 255), 
            2
        )
        
        # Save comparison
        comparison_path = os.path.join(epoch_dir, f"comparison.png")
        cv2.imwrite(comparison_path, comparison)

    # If future frame and pedestrians are provided, visualize them
    if future_frame is not None and future_pedestrians is not None:
        # Convert from float [0,1] to uint8 [0,255]
        future_img = (future_frame * 255).astype(np.uint8)
    
        # Create a copy for drawing
        future_with_detections = future_img.copy()
    
        # Draw pedestrian detections
        for ped in future_pedestrians:
            # Draw bounding box
            x1, y1, x2, y2 = ped.bbox.astype(int)
            cv2.rectangle(future_with_detections, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
            # Add confidence text
            cv2.putText(
                future_with_detections,
                f"{ped.confidence:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
        
            # Draw center position
            cx, cy = ped.position.astype(int)
            cv2.circle(future_with_detections, (cx, cy), 4, (255, 0, 0), -1)
    
        # Add timestamp text
        cv2.putText(
            future_with_detections, 
            f"t=+{FUTURE_OFFSET_F/10.0:.1f}s (future)", 
            (20, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (255, 255, 255), 
            2
        )
    
        # Save future frame with segmentations
        future_with_detections_bgr = cv2.cvtColor(future_with_detections, cv2.COLOR_RGB2BGR)
        future_path = os.path.join(epoch_dir, f"future_frame_with_detections.png")
        cv2.imwrite(future_path, future_with_detections_bgr)
    
        # Create blended visualization (future frame + heatmap overlay)
        if prediction_heatmap_np.shape[:2] == future_img.shape[:2]:
            heatmap_overlay = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
            alpha = 0.5  # Transparency factor
            blended = cv2.addWeighted(future_with_detections, 1-alpha,
                                      heatmap_overlay, alpha, 0)
            
            # Save blended visualization
            blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
            blended_path = os.path.join(epoch_dir, f"future_frame_with_heatmap_overlay.png")
            cv2.imwrite(blended_path, blended_bgr)
    #else:
        #print('no future frame', future_frame, future_pedestrians)
        #exit()
    logging.info(f"Saved debug images for epoch {epoch} to {epoch_dir}")


