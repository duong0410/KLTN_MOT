#!/usr/bin/env python3
"""
Test YOLO11s Traffic Model on Video Frames
Extract frames from video and run detection with class labels
"""
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from collections import Counter

# ============================================================================
# Configuration
# ============================================================================

MODEL_PATH = r"D:\Learn\Year4\KLTN\Dataset\traffic_yolo\runs\yolo11s_traffic2\weights\best.pt"
CONF_THRESHOLD = 0.25  # Confidence threshold
IOU_THRESHOLD = 0.45   # NMS IoU threshold

# Test video path - change this to your video
TEST_VIDEO = r"D:\Download\Video Project.mp4"

# Output folderpython test_yolo11s.py
# Nhập: D:\path\to\video.mp4
# Nhập: 10 (hoặc Enter để dùng mặc định 5)
# → Tự động extract frame 0-9 và detect
OUTPUT_DIR = Path(r"D:\Learn\Year4\KLTN\test_yolo_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Frame extraction settings
NUM_FRAMES = 5  # Number of frames to extract
FRAME_INTERVAL = 30  # Extract every N frames

# ============================================================================
# Frame Extraction
# ============================================================================

def extract_frames_from_video(video_path, num_frames=5, frame_interval=30, frame_numbers=None):
    """
    Extract frames from video
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        frame_interval: Extract every N frames (if frame_numbers not provided)
        frame_numbers: Specific frame numbers to extract (e.g., [1, 10, 20, 50, 100])
    
    Returns:
        List of (frame_number, frame_image) tuples
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n{'='*60}")
    print(f"Video: {Path(video_path).name}")
    print(f"  Size: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"{'='*60}")
    
    # Determine which frames to extract
    if frame_numbers is not None:
        frames_to_extract = [f for f in frame_numbers if f < total_frames]
    else:
        frames_to_extract = list(range(0, min(num_frames * frame_interval, total_frames), frame_interval))[:num_frames]
    
    print(f"\nExtracting {len(frames_to_extract)} frames: {frames_to_extract}")
    
    extracted_frames = []
    current_frame = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if current_frame in frames_to_extract:
            extracted_frames.append((current_frame, frame.copy()))
            print(f"  Extracted frame {current_frame}")
        
        current_frame += 1
        
        # Early exit if we have all frames
        if len(extracted_frames) >= len(frames_to_extract):
            break
    
    cap.release()
    
    print(f"\n Successfully extracted {len(extracted_frames)} frames")
    return extracted_frames

# ============================================================================
# Detection and Visualization
# ============================================================================

def test_frame(frame_img, model, conf=0.25, iou=0.45, frame_number=None, save_output=True):
    """
    Run detection on a single frame and display results with class labels
    
    Args:
        frame_img: Frame image (numpy array)
        model: YOLO model
        conf: Confidence threshold
        iou: NMS IoU threshold
        frame_number: Frame number (for naming)
        save_output: Whether to save output image
    """
    filename = f"frame_{frame_number:06d}" if frame_number is not None else "frame"
    
    print(f"\n{'='*60}")
    print(f"Testing: {filename}")
    print(f"{'='*60}")
    
    h, w = frame_img.shape[:2]
    print(f"Frame size: {w}x{h}")
    
    # Run inference
    print(f"\nRunning detection...")
    print(f"  Confidence threshold: {conf}")
    print(f"  NMS IoU threshold: {iou}")
    
    results = model(frame_img, conf=conf, iou=iou, verbose=False)
    
    # Process results
    result = results[0]
    boxes = result.boxes
    
    if boxes is None or len(boxes) == 0:
        print(f"\n No detections found!")
        return
    
    # Get detections
    xyxy = boxes.xyxy.cpu().numpy()
    conf_scores = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    
    # Count detections by class
    class_counts = Counter([model.names[cls] for cls in classes])
    
    print(f"\n Found {len(boxes)} detections:")
    print(f"\nClass distribution:")
    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name}: {count}")
    
    print(f"\nDetailed detections:")
    print(f"{'#':<4} {'Class':<15} {'Confidence':<12} {'BBox (x1,y1,x2,y2)'}")
    print("-" * 70)
    
    # Draw on image
    output_img = frame_img.copy()
    
    # Generate colors for each class
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)
    
    for i, (box, score, cls) in enumerate(zip(xyxy, conf_scores, classes), 1):
        x1, y1, x2, y2 = map(int, box)
        class_name = model.names[cls]
        
        print(f"{i:<4} {class_name:<15} {score:<12.3f} ({x1},{y1},{x2},{y2})")
        
        # Get color for this class
        color = colors[cls % len(colors)].tolist()
        
        # Draw bounding box
        cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)
        
        # Draw label with class name and confidence
        label = f"{class_name}: {score:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Draw background for text
        cv2.rectangle(output_img, (x1, y1 - label_h - 10), 
                     (x1 + label_w + 10, y1), color, -1)
        
        # Draw text
        cv2.putText(output_img, label, (x1 + 5, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Add frame info at top
    info_text = f"Frame: {frame_number} | Detections: {len(boxes)}"
    cv2.putText(output_img, info_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Save output image (no display window)
    if save_output:
        output_path = OUTPUT_DIR / f"{filename}_result.jpg"
        cv2.imwrite(str(output_path), output_img)
        print(f"\n Saved to: {output_path}")
    
    print(f"{'='*60}\n")

# ============================================================================
# Main Testing Functions
# ============================================================================

def test_video_frames(video_path, model, conf=0.25, iou=0.45, num_frames=5, 
                      frame_interval=30, frame_numbers=None):
    """
    Extract frames from video and run detection on them
    
    Args:
        video_path: Path to video file
        model: YOLO model
        conf: Confidence threshold
        iou: NMS IoU threshold
        num_frames: Number of frames to extract
        frame_interval: Extract every N frames
        frame_numbers: Specific frame numbers to extract
    """
    # Extract frames
    frames = extract_frames_from_video(video_path, num_frames, frame_interval, frame_numbers)
    
    if not frames:
        print("No frames extracted!")
        return
    
    print(f"\n{'='*60}")
    print(f"Running detection on {len(frames)} frames...")
    print(f"{'='*60}\n")
    
    # Test each frame
    for i, (frame_num, frame_img) in enumerate(frames):
        test_frame(frame_img, model, conf, iou, frame_number=frame_num, save_output=True)
    
    print(f"\n{'='*60}")
    print(f" Completed testing {len(frames)} frames")
    print(f" Results saved to: {OUTPUT_DIR}")
    print(f"{'='*60}\n")


def main():
    """Main function - Extract first N frames from video and detect"""
    print("=" * 60)
    print("YOLO11s Traffic Model - Video Frame Testing")
    print("=" * 60)
    
    # Check model exists
    if not Path(MODEL_PATH).exists():
        print(f"\n Error: Model not found at {MODEL_PATH}")
        return
    
    # Load model
    print(f"\nLoading model: {Path(MODEL_PATH).name}")
    model = YOLO(MODEL_PATH)
    
    print(f" Model loaded successfully!")
    print(f"\nModel classes: {list(model.names.values())}")
    print(f"Number of classes: {len(model.names)}")
    
    # Get video path
    print("\n" + "=" * 60)
    video_path = input("Enter video path: ").strip().strip('"')
    
    if not Path(video_path).exists():
        print(f" Error: Video not found at {video_path}")
        return
    
    # Get number of frames to extract
    num_input = input(f"Number of frames to extract (default: {NUM_FRAMES}): ").strip()
    num_frames = int(num_input) if num_input else NUM_FRAMES
    
    print(f"\n Will extract first {num_frames} frames from video")
    print(f" Confidence threshold: {CONF_THRESHOLD}")
    print(f" NMS IoU threshold: {IOU_THRESHOLD}")
    print("=" * 60)
    
    # Extract first N frames (0, 1, 2, ..., N-1)
    frame_numbers = list(range(num_frames))
    test_video_frames(video_path, model, CONF_THRESHOLD, IOU_THRESHOLD, 
                     frame_numbers=frame_numbers)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Interrupted by user")
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
