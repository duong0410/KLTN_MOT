#!/usr/bin/env python3
"""
ByteTrack-YOLO: Unified Tracking + Violation Detection System
Single entry point for complete video tracking with traffic violation detection

Features:
- YOLO object detection
- ByteTrack multi-object tracking
- Traffic violation detection
- ROI-based filtering
- Real-time visualization

Usage:
    # GUI Mode (Interactive)
    python track_and_detect.py --mode gui
    
    # Command line mode (Process video file)
    python track_and_detect.py --mode cli --video video.mp4 --output output.mp4
    
    # With config file
    python track_and_detect.py --mode cli --config configs/default_config.yaml
"""

import sys
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.detector import YOLODetector
from src.tracker import BYTETracker, STrack, ViolationType, TrafficLane, ViolationDetector
from src.utils.visualization import TrackVisualizer
from src.utils.roi_utils import create_roi_mask, filter_detections_by_roi, ROISelector
from src.utils.bbox import tlbr_to_tlwh


# ============================================================================
# Core Tracking + Violation Detection Engine
# ============================================================================

class TrackingEngine:
    """Complete tracking engine with violation detection"""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.1, 
                 device: str = 'cuda', frame_limit: Optional[int] = None):
        """
        Initialize tracking engine
        
        Args:
            model_path: Path to YOLO model
            conf_threshold: Detection confidence threshold
            device: Device to use ('cuda' or 'cpu')
            frame_limit: Max frames to process (None = all)
        """
        self.detector = YOLODetector(
            model_path=model_path,
            conf_threshold=conf_threshold,
            device=device
        )
        self.tracker = BYTETracker(
            det_conf_high=0.5,
            det_conf_low=0.1,
            new_track_thresh=0.6,
            match_thresh_high=0.8,
            match_thresh_low=0.5,
            track_buffer=30,
            min_hits=1
        )
        self.violation_detector = ViolationDetector()
        self.visualizer = TrackVisualizer(
            class_names=self.detector.model.names
        )
        
        self.frame_count = 0
        self.frame_limit = frame_limit
        self.fps = 0
        self.roi_mask = None
        self.lanes: Dict[int, TrafficLane] = {}
    
    def add_lane(self, lane: TrafficLane):
        """Add a traffic lane for violation detection"""
        self.lanes[lane.lane_id] = lane
        self.violation_detector.add_lane(lane)
    
    def set_roi(self, roi_polygon: Optional[List[Tuple[int, int]]], frame_shape: Tuple[int, int]):
        """Set region of interest for detection filtering"""
        if roi_polygon is not None:
            self.roi_mask = create_roi_mask(frame_shape, roi_polygon)
        else:
            self.roi_mask = None
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[STrack], Dict]:
        """
        Process single frame with complete pipeline
        
        Args:
            frame: Input frame (BGR)
        
        Returns:
            frame_visual: Visualized frame
            tracks: List of active tracks
            info: Dictionary with detection/tracking info
        """
        self.frame_count += 1
        
        # Check frame limit
        if self.frame_limit and self.frame_count > self.frame_limit:
            return None, [], {}
        
        h, w = frame.shape[:2]
        
        # Step 1: Detect objects
        detections = self.detector.detect(frame)
        
        # Step 2: Filter by ROI if defined
        if self.roi_mask is not None:
            detections = filter_detections_by_roi(detections, self.roi_mask)
        
        # Step 3: Track objects
        output_stracks = self.tracker.update(detections, (h, w))
        
        # Step 4: Detect violations
        violations = self.violation_detector.detect_violations(
            output_stracks, self.frame_count
        )
        
        # Step 5: Visualize
        frame_visual = self.visualizer.draw_tracks(frame, output_stracks, draw_info=True)
        
        # Draw violations
        for track in output_stracks:
            if track.is_activated and track.violation_type != ViolationType.NONE:
                tlbr = track.tlbr
                x1, y1, x2, y2 = map(int, tlbr)
                # Draw red box for violations
                cv2.rectangle(frame_visual, (x1, y1), (x2, y2), (0, 0, 255), 3)
                # Draw violation type
                violation_text = track.violation_type.name
                cv2.putText(
                    frame_visual, f"VIOLATION: {violation_text}", 
                    (x1, y1 - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                )
        
        # Draw ROI if set
        if self.roi_mask is not None:
            contours = cv2.findContours(self.roi_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
            cv2.drawContours(frame_visual, contours, -1, (255, 255, 0), 2)
        
        # Prepare info
        info = {
            'frame': self.frame_count,
            'detections': len(detections),
            'tracks': len(output_stracks),
            'violations': len(violations),
            'fps': self.fps
        }
        
        return frame_visual, output_stracks, info
    
    def reset(self):
        """Reset tracker state"""
        self.tracker.reset()
        self.violation_detector = ViolationDetector()
        for lane_id, lane in self.lanes.items():
            self.violation_detector.add_lane(lane)
        self.frame_count = 0


# ============================================================================
# Video Processing
# ============================================================================

class VideoProcessor:
    """Process video files with tracking"""
    
    def __init__(self, engine: TrackingEngine, output_path: Optional[str] = None):
        """
        Initialize video processor
        
        Args:
            engine: TrackingEngine instance
            output_path: Path to save output video (None = display only)
        """
        self.engine = engine
        self.output_path = output_path
        self.writer = None
    
    def process(self, video_path: str, show: bool = True) -> bool:
        """
        Process video file
        
        Args:
            video_path: Path to input video
            show: Whether to display while processing
        
        Returns:
            True if successful
        """
        print("=" * 70)
        print("ByteTrack-YOLO: Video Processing with Tracking & Violation Detection")
        print("=" * 70)
        print(f"Input: {video_path}")
        if self.output_path:
            print(f"Output: {self.output_path}")
        print()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video file {video_path}")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video info: {width}x{height} @ {fps} FPS, {total_frames} frames")
        print()
        
        # Initialize video writer if output path specified
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(
                self.output_path, fourcc, fps, (width, height)
            )
        
        frame_idx = 0
        import time
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                frame_visual, tracks, info = self.engine.process_frame(frame)
                if frame_visual is None:
                    break
                
                # Update FPS
                elapsed = time.time() - start_time
                self.engine.fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0
                
                # Add info panel
                info_text = [
                    f"Frame: {info['frame']}/{total_frames}",
                    f"Detections: {info['detections']}",
                    f"Tracks: {info['tracks']}",
                    f"Violations: {info['violations']}",
                    f"FPS: {self.engine.fps:.1f}"
                ]
                
                y_offset = 30
                for text in info_text:
                    cv2.putText(
                        frame_visual, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                    )
                    y_offset += 25
                
                # Write to output
                if self.writer:
                    self.writer.write(frame_visual)
                
                # Display
                if show:
                    cv2.imshow('ByteTrack-YOLO', frame_visual)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                frame_idx += 1
                
                # Progress
                if frame_idx % 30 == 0:
                    print(f"Processed {frame_idx}/{total_frames} frames... "
                          f"({100*frame_idx/total_frames:.1f}%)")
        
        finally:
            cap.release()
            if self.writer:
                self.writer.release()
            if show:
                cv2.destroyAllWindows()
        
        print()
        print(f"✓ Processing complete! Processed {frame_idx} frames")
        if self.output_path:
            print(f"✓ Output saved to: {self.output_path}")
        
        return True


# ============================================================================
# GUI Mode
# ============================================================================

def run_gui_mode():
    """Run GUI mode"""
    try:
        from src.gui.app import create_gui
    except ImportError as e:
        print(f"Error: GUI modules not installed: {e}")
        print("Install with: pip install tkinter pillow")
        return False
    
    print("=" * 70)
    print("ByteTrack-YOLO: GUI Mode")
    print("=" * 70)
    print()
    
    # Find default model
    model_paths = [
        Path(__file__).parent / "models" / "yolo11s_traffic.pt",
        Path(__file__).parent / "models" / "best.pt",
    ]
    
    default_model = None
    for path in model_paths:
        if path.exists():
            default_model = str(path)
            print(f"Found model: {path.name}")
            break
    
    if not default_model:
        print("Warning: No model found in models/ directory")
        print("You can still use the GUI and select a model manually")
    
    print()
    print("Launching GUI...")
    create_gui(default_model)
    return True


# ============================================================================
# CLI Mode
# ============================================================================

def run_cli_mode(args):
    """Run CLI mode"""
    if not args.video:
        print("Error: --video is required for CLI mode")
        return False
    
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return False
    
    # Find model
    model_path = args.model
    if not model_path:
        model_candidates = [
            Path(__file__).parent / "models" / "yolo11s_traffic.pt",
            Path(__file__).parent / "models" / "best.pt",
        ]
        for path in model_candidates:
            if path.exists():
                model_path = str(path)
                break
    
    if not model_path or not Path(model_path).exists():
        print(f"Error: Model not found: {model_path}")
        return False
    
    print(f"Using model: {model_path}")
    
    # Output path
    output_path = args.output
    if not output_path:
        output_path = str(video_path.parent / f"{video_path.stem}_tracked.mp4")
    
    # Initialize engine
    engine = TrackingEngine(
        model_path=model_path,
        conf_threshold=args.conf_thresh,
        device=args.device,
        frame_limit=args.max_frames
    )
    
    # Process video
    processor = VideoProcessor(engine, output_path)
    return processor.process(str(video_path), show=not args.headless)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='ByteTrack-YOLO: Multi-Object Tracking with Violation Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run GUI
  python track_and_detect.py --mode gui
  
  # Process video
  python track_and_detect.py --mode cli --video input.mp4 --output output.mp4
  
  # Process with specific model
  python track_and_detect.py --mode cli --video input.mp4 --model path/to/model.pt
  
  # Process on CPU
  python track_and_detect.py --mode cli --video input.mp4 --device cpu
        """
    )
    
    parser.add_argument('--mode', type=str, choices=['gui', 'cli'], default='gui',
                        help='Run mode: gui (interactive) or cli (batch processing)')
    
    # Video processing arguments
    parser.add_argument('--video', type=str, help='Input video file path')
    parser.add_argument('--output', type=str, help='Output video file path (default: input_tracked.mp4)')
    parser.add_argument('--model', type=str, help='Path to YOLO model file')
    parser.add_argument('--conf-thresh', type=float, default=0.1,
                        help='Detection confidence threshold (default: 0.1)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda',
                        help='Device to use for detection (default: cuda)')
    parser.add_argument('--max-frames', type=int, help='Maximum frames to process (default: all)')
    parser.add_argument('--headless', action='store_true', help='Process without displaying video')
    
    args = parser.parse_args()
    
    print("\n")
    
    # Run appropriate mode
    if args.mode == 'gui':
        success = run_gui_mode()
    else:  # cli
        success = run_cli_mode(args)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
