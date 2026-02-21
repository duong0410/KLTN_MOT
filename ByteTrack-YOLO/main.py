"""
ByteTrack-YOLO Command Line Interface
Process videos with ByteTrack tracking from command line
"""

import sys
import argparse
import time
import yaml
from pathlib import Path

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.detector import YOLODetector
from src.tracker import BYTETracker, STrack
from src.utils.visualization import TrackVisualizer


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def process_video(video_path, output_path, config):
    """
    Process video with tracking
    
    Args:
        video_path: Path to input video
        output_path: Path to save output video
        config: Configuration dictionary
    """
    print("=" * 60)
    print("ByteTrack-YOLO: Video Processing")
    print("=" * 60)
    print(f"Input video: {video_path}")
    print(f"Output video: {output_path}")
    print()
    
    # Initialize detector
    print("Initializing YOLO detector...")
    detector = YOLODetector(
        model_path=config['detection']['model_path'],
        conf_threshold=config['detection']['conf_threshold'],
        device=config['detection']['device'],
        min_box_area=config['detection'].get('min_box_area', 400),
        edge_margin=config['detection'].get('edge_margin', 10)
    )
    print("Detector initialized")
    
    # Initialize tracker
    print("Initializing ByteTrack...")
    tracker = BYTETracker(
        det_conf_high=config['tracker']['det_conf_high'],
        det_conf_low=config['tracker']['det_conf_low'],
        new_track_thresh=config['tracker']['new_track_thresh'],
        match_thresh_high=config['tracker']['match_thresh_high'],
        match_thresh_low=config['tracker']['match_thresh_low'],
        track_buffer=config['tracker']['track_buffer'],
        min_hits=config['tracker']['min_hits']
    )
    print("Tracker initialized")
    
    # Initialize visualizer
    visualizer = TrackVisualizer(
        class_names=detector.class_names,
        seed=config['visualization']['color_seed']
    )
    
    # Open video
    print(f"\nOpening video...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print()
    
    # Setup video writer
    writer = None
    if config['video']['save_output']:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f" Output will be saved to: {output_path}")
    
    # Process video
    print("\n" + "=" * 60)
    print("Processing...")
    print(f"{'Frame':<10} {'Detections':<12} {'Tracks':<10} {'FPS':<10}")
    print("-" * 60)
    
    frame_id = 0
    fps_list = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_id += 1
            start_time = time.time()
            
            # Run detection
            detections = detector.detect(frame)
            
            # Run tracking
            online_tracks = tracker.update(detections, (height, width))
            
            # Calculate FPS
            elapsed = time.time() - start_time
            fps_val = 1.0 / elapsed if elapsed > 0 else 0
            fps_list.append(fps_val)
            
            # Print progress
            if frame_id % 30 == 0 or frame_id == 1:
                print(f"{frame_id:<10} {len(detections):<12} {len(online_tracks):<10} {fps_val:<10.2f}")
            
            # Visualize
            if config['visualization']['draw_tracks']:
                frame = visualizer.draw_tracks(frame, online_tracks)
            
            if config['visualization']['draw_info_panel']:
                frame = visualizer.draw_info_panel(
                    frame, frame_id, total_frames,
                    len(detections), len(online_tracks), fps_val
                )
            
            # Display
            if config['video']['display']:
                cv2.imshow('ByteTrack-YOLO', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nStopped by user")
                    break
            
            # Save
            if writer:
                writer.write(frame)
    
    finally:
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        if config['video']['display']:
            cv2.destroyAllWindows()
        
        # Print summary
        print("\n" + "=" * 60)
        print("Processing completed!")
        print(f"  Total frames processed: {frame_id}")
        if len(fps_list) > 0:
            print(f"  Average FPS: {np.mean(fps_list):.2f}")
        print(f"  Total tracks created: {STrack.track_id_count}")
        if output_path and writer:
            print(f"  Output saved to: {output_path}")
        print("=" * 60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="ByteTrack-YOLO: Multi-Object Tracking System"
    )
    parser.add_argument(
        '--video', '-v',
        required=True,
        help='Path to input video file'
    )
    parser.add_argument(
        '--output', '-o',
        help='Path to output video file (default: input_tracked.mp4)'
    )
    parser.add_argument(
        '--config', '-c',
        default='configs/default_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--model', '-m',
        help='Path to YOLO model (overrides config)'
    )
    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu'],
        help='Device to run on (overrides config)'
    )
    parser.add_argument(
        '--display',
        action='store_true',
        help='Display video while processing'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save output video'
    )
    
    args = parser.parse_args()
    
    # Check input video
    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
    
    # Load configuration
    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.model:
        config['detection']['model_path'] = args.model
    if args.device:
        config['detection']['device'] = args.device
    if args.display:
        config['video']['display'] = True
    if args.no_save:
        config['video']['save_output'] = False
    
    # Check model exists
    if not Path(config['detection']['model_path']).exists():
        print(f"Error: Model file not found: {config['detection']['model_path']}")
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        video_path = Path(args.video)
        output_path = str(video_path.parent / f"{video_path.stem}_tracked.mp4")
    
    # Process video
    try:
        process_video(args.video, output_path, config)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
