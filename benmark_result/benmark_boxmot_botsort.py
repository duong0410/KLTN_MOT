#!/usr/bin/env python3
"""
BoT-SORT using BoxMOT library (official implementation)
This script uses the official BoxMOT library to run BoT-SORT on MOT17 dataset
Compatible with BoxMOT 16.0.7

BoT-SORT improvements over ByteTrack:
1. Improved Kalman Filter
2. Camera Motion Compensation (GMC)
3. ReID features (optional)
"""

# Fix OpenMP error BEFORE any imports
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
from pathlib import Path
import motmetrics as mm
from collections import defaultdict
import configparser
import argparse

# Import BotSort from BoxMOT
try:
    from boxmot import BotSort
    print("BoxMOT BotSort imported successfully!")
except ImportError as e:
    print(f"Cannot import BotSort from boxmot: {e}")
    print("\nPlease install BoxMOT:")
    print("  pip install boxmot")
    exit(1)


class MOT17Dataset:
    """MOT17 Dataset loader"""
    
    def __init__(self, data_root, split='train'):
        self.data_root = Path(data_root)
        self.split = split
        self.split_dir = self.data_root / split
        
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")
        
        # Get all sequences
        self.sequences = sorted([
            d.name for d in self.split_dir.iterdir() 
            if d.is_dir() and d.name.startswith('MOT17-')
        ])
        
        print(f"Found {len(self.sequences)} sequences in {split} split")
    
    def get_sequence_info(self, seq_name):
        """Load sequence info from seqinfo.ini"""
        seq_path = self.split_dir / seq_name
        seqinfo_path = seq_path / 'seqinfo.ini'
        
        config = configparser.ConfigParser()
        config.read(seqinfo_path)
        
        return {
            'name': config.get('Sequence', 'name'),
            'imDir': config.get('Sequence', 'imDir'),
            'frameRate': int(config.get('Sequence', 'frameRate')),
            'seqLength': int(config.get('Sequence', 'seqLength')),
            'imWidth': int(config.get('Sequence', 'imWidth')),
            'imHeight': int(config.get('Sequence', 'imHeight')),
            'imExt': config.get('Sequence', 'imExt')
        }
    
    def load_detections(self, seq_name):
        """Load detections from det.txt"""
        seq_path = self.split_dir / seq_name
        det_file = seq_path / 'det' / 'det.txt'
        
        detections = defaultdict(list)
        
        with open(det_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 7:
                    continue
                
                frame_id = int(parts[0])
                x, y, w, h = map(float, parts[2:6])
                conf = float(parts[6])
                
                # Convert to x1y1x2y2 format
                detection = [x, y, x + w, y + h, conf]
                detections[frame_id].append(detection)
        
        return detections


def track_mot17_with_botsort(dataset, output_dir, args):
    """
    Track MOT17 using BoxMOT's BoT-SORT
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("Running BoT-SORT")
    print("="*80)
    
    for seq_name in dataset.sequences:
        print(f"\nProcessing: {seq_name}")
        
        # Load sequence info
        seq_info = dataset.get_sequence_info(seq_name)
        detections = dataset.load_detections(seq_name)
        
        # Initialize BoxMOT BotSort
        # BoxMOT 16.0.7 BotSort parameters
        tracker = BotSort(
            reid_weights=Path('osnet_x0_25_msmt17.pt'), # Required parameter (but won't be used with with_reid=False)
            device='cuda:0' if args.device == 'cuda' else 'cpu',  # Device
            half=args.half,                              # Use FP16 half precision
            track_high_thresh=args.track_thresh,         # High confidence threshold
            track_low_thresh=args.track_low_thresh,      # Low confidence threshold
            new_track_thresh=args.new_track_thresh,      # New track threshold
            track_buffer=args.track_buffer,              # Buffer for lost tracks
            match_thresh=args.match_thresh,              # Matching threshold
            proximity_thresh=args.proximity_thresh,      # Proximity threshold for ReID
            appearance_thresh=args.appearance_thresh,    # Appearance threshold
            cmc_method=args.cmc_method,                  # Camera Motion Compensation method
            frame_rate=seq_info['frameRate'],            # Frame rate
            with_reid=False                              # DISABLE ReID features
        )
        
        # Process each frame
        results = []
        num_frames = seq_info['seqLength']
        
        for frame_id in range(1, num_frames + 1):
            frame_dets = detections.get(frame_id, [])
            
            # Prepare detections in the format BoxMOT expects
            # Format: [x1, y1, x2, y2, conf, cls]
            if len(frame_dets) == 0:
                dets = np.empty((0, 6), dtype=np.float32)
            else:
                dets = np.array(frame_dets, dtype=np.float32)
                # Add class column (0 for person class)
                cls_column = np.zeros((dets.shape[0], 1), dtype=np.float32)
                dets = np.hstack([dets, cls_column])
            
            # Update tracker
            # BoxMOT BotSort.update() requires: update(dets, img)
            # Create dummy image (CMC won't work without real images)
            dummy_img = np.zeros((seq_info['imHeight'], seq_info['imWidth'], 3), dtype=np.uint8)
            
            try:
                tracks = tracker.update(dets, dummy_img)
            except Exception as e:
                if frame_id == 1:
                    print(f"  Warning: Error updating tracker: {e}")
                tracks = np.empty((0, 8))
            
            # Process tracks and save results
            if tracks is not None and len(tracks) > 0:
                for track in tracks:
                    # BotSort output format: [x1, y1, x2, y2, track_id, conf, cls, det_ind]
                    if len(track) >= 5:
                        x1, y1, x2, y2, track_id = track[:5]
                        w = x2 - x1
                        h = y2 - y1
                        
                        # MOT format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, -1, -1, -1
                        result = f"{frame_id},{int(track_id)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n"
                        results.append(result)
        
        # Save results
        output_file = output_dir / f"{seq_name}.txt"
        with open(output_file, 'w') as f:
            f.writelines(results)
        
        print(f"  Saved: {output_file} ({len(results)} detections)")


def compute_mot_metrics(data_root, pred_dir):
    """Compute MOT metrics using motmetrics"""
    print("\n" + "="*80)
    print("Computing MOT Metrics")
    print("="*80)
    
    data_root = Path(data_root)
    pred_dir = Path(pred_dir)
    gt_dir = data_root / 'train'
    
    # Find all prediction files
    pred_files = sorted(pred_dir.glob('MOT17-*.txt'))
    
    if not pred_files:
        print(f"No prediction files found in {pred_dir}")
        return
    
    # Group by detector type
    detector_metrics = defaultdict(list)
    
    for pred_file in pred_files:
        seq_name = pred_file.stem
        
        # Determine detector type
        detector = None
        if 'DPM' in seq_name:
            detector = 'DPM'
        elif 'FRCNN' in seq_name:
            detector = 'FRCNN'
        elif 'SDP' in seq_name:
            detector = 'SDP'
        else:
            continue
        
        # Load ground truth
        gt_file = gt_dir / seq_name / 'gt' / 'gt.txt'
        if not gt_file.exists():
            print(f"Ground truth not found for {seq_name}")
            continue
        
        # Load GT and predictions
        gt_data = defaultdict(list)
        with open(gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 8:
                    continue
                
                frame = int(parts[0])
                track_id = int(parts[1])
                x, y, w, h = map(float, parts[2:6])
                conf = int(parts[6])
                cls = int(parts[7])
                
                # Filter: only pedestrians (cls=1), consider (conf=1)
                if cls == 1 and conf == 1:
                    gt_data[frame].append((track_id, x, y, w, h))
        
        pred_data = defaultdict(list)
        with open(pred_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 6:
                    continue
                
                frame = int(parts[0])
                track_id = int(parts[1])
                x, y, w, h = map(float, parts[2:6])
                pred_data[frame].append((track_id, x, y, w, h))
        
        # Compute metrics
        acc = mm.MOTAccumulator(auto_id=True)
        
        all_frames = sorted(set(gt_data.keys()) | set(pred_data.keys()))
        
        num_updates = 0
        for frame in all_frames:
            gt_boxes = gt_data.get(frame, [])
            pred_boxes = pred_data.get(frame, [])
            
            if len(gt_boxes) == 0 and len(pred_boxes) == 0:
                continue
            
            gt_ids = [box[0] for box in gt_boxes]
            pred_ids = [box[0] for box in pred_boxes]
            
            # Compute IoU distance
            if len(gt_boxes) > 0 and len(pred_boxes) > 0:
                gt_bboxes = np.array([[x, y, x+w, y+h] for _, x, y, w, h in gt_boxes])
                pred_bboxes = np.array([[x, y, x+w, y+h] for _, x, y, w, h in pred_boxes])
                
                # Compute IoU matrix
                ious = np.zeros((len(gt_boxes), len(pred_boxes)))
                for i, gt_box in enumerate(gt_bboxes):
                    for j, pred_box in enumerate(pred_bboxes):
                        xx1 = max(gt_box[0], pred_box[0])
                        yy1 = max(gt_box[1], pred_box[1])
                        xx2 = min(gt_box[2], pred_box[2])
                        yy2 = min(gt_box[3], pred_box[3])
                        
                        w = max(0, xx2 - xx1)
                        h = max(0, yy2 - yy1)
                        inter = w * h
                        
                        area_gt = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                        area_pred = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
                        union = area_gt + area_pred - inter
                        
                        ious[i, j] = inter / union if union > 0 else 0
                
                distances = 1 - ious
            else:
                distances = np.empty((len(gt_boxes), len(pred_boxes)))
            
            acc.update(gt_ids, pred_ids, distances)
            num_updates += 1
        
        # Debug: Check if accumulator has data
        if num_updates == 0:
            print(f"  Warning: {seq_name} - No frames processed!")
            continue
            
        # Store accumulator
        detector_metrics[detector].append((seq_name, acc))
    
    # Compute summary
    print("\n" + "="*100)
    print(f"{'Detector':<15} {'MOTA':<10} {'IDF1':<10} {'MOTP':<10} {'Precision':<12} {'Recall':<10} {'ID_Sw':<10} {'Frag':<10} {'FP':<10} {'FN':<10}")
    print("-"*100)
    
    summary_results = []
    
    for detector in ['DPM', 'FRCNN', 'SDP']:
        if detector not in detector_metrics:
            continue
        
        # Merge all accumulators for this detector
        accs = [acc for _, acc in detector_metrics[detector]]
        names = [name for name, _ in detector_metrics[detector]]
        
        mh = mm.metrics.create()
        summary = mh.compute_many(
            accs,
            metrics=['mota', 'motp', 'idf1', 'precision', 'recall', 
                    'num_switches', 'num_fragmentations', 'num_false_positives', 'num_misses'],
            names=names
        )
        
        # Get overall metrics
        # motmetrics might use different index names
        if 'OVERALL' in summary.index:
            overall = summary.loc['OVERALL']
        elif len(summary) > 0:
            # Use the last row which is typically the overall summary
            overall = summary.iloc[-1]
        else:
            print(f"{detector:<15} No metrics computed - skipping")
            continue
        
        mota = overall['mota'] * 100
        idf1 = overall['idf1'] * 100
        motp = overall['motp']
        precision = overall['precision'] * 100
        recall = overall['recall'] * 100
        id_sw = int(overall['num_switches'])
        frag = int(overall['num_fragmentations'])
        fp = int(overall['num_false_positives'])
        fn = int(overall['num_misses'])
        
        print(f"{detector:<15} {mota:>6.2f}%   {idf1:>6.2f}%   {motp:>6.3f}    {precision:>8.2f}%   {recall:>6.2f}%   {id_sw:>6}    {frag:>6}   {fp:>6}   {fn:>8}")
        
        summary_results.append({
            'detector': detector,
            'mota': mota,
            'idf1': idf1,
            'motp': motp,
            'precision': precision,
            'recall': recall,
            'id_sw': id_sw,
            'frag': frag,
            'fp': fp,
            'fn': fn
        })
    
    # Compute average
    if summary_results:
        avg_mota = np.mean([r['mota'] for r in summary_results])
        avg_idf1 = np.mean([r['idf1'] for r in summary_results])
        avg_motp = np.mean([r['motp'] for r in summary_results])
        avg_precision = np.mean([r['precision'] for r in summary_results])
        avg_recall = np.mean([r['recall'] for r in summary_results])
        avg_id_sw = sum([r['id_sw'] for r in summary_results])
        avg_frag = sum([r['frag'] for r in summary_results])
        avg_fp = sum([r['fp'] for r in summary_results])
        avg_fn = sum([r['fn'] for r in summary_results])
        
        print("-"*100)
        print(f"{'AVERAGE':<15} {avg_mota:>6.2f}%   {avg_idf1:>6.2f}%   {avg_motp:>6.3f}    {avg_precision:>8.2f}%   {avg_recall:>6.2f}%   {avg_id_sw:>6}    {avg_frag:>6}   {avg_fp:>6}   {avg_fn:>8}")
        print("="*100)
    
    # Return detector_metrics for saving detailed results
    return detector_metrics, summary_results


def save_detailed_summary(pred_dir, detector_metrics, summary_results, args):
    """Save detailed evaluation summary to file"""
    summary_file = pred_dir / 'evaluation_summary.txt'
    
    with open(summary_file, 'w') as f:
        f.write("="*100 + "\n")
        f.write(f"{'MOT17 EVALUATION RESULTS - BoT-SORT (BoxMOT Library)':^100}\n")
        f.write("="*100 + "\n\n")
        
        # Write configuration
        f.write("BoT-SORT Configuration:\n")
        f.write(f"  track_thresh:       {args.track_thresh} (high confidence threshold)\n")
        f.write(f"  track_low_thresh:   {args.track_low_thresh} (low confidence threshold)\n")
        f.write(f"  new_track_thresh:   {args.new_track_thresh} (new track threshold)\n")
        f.write(f"  match_thresh:       {args.match_thresh} (IoU matching threshold)\n")
        f.write(f"  proximity_thresh:   {args.proximity_thresh} (proximity for ReID)\n")
        f.write(f"  appearance_thresh:  {args.appearance_thresh} (appearance threshold)\n")
        f.write(f"  track_buffer:       {args.track_buffer} (frames to keep lost tracks)\n")
        f.write(f"  cmc_method:         {args.cmc_method} (camera motion compensation)\n")
        f.write(f"  device:             {args.device} (computation device)\n")
        f.write(f"  half:               {args.half} (FP16 precision)\n\n")
        
        # Write summary table
        f.write("-"*100 + "\n")
        f.write(f"{'Detector':<15} {'MOTA':<10} {'IDF1':<10} {'MOTP':<10} {'Precision':<12} {'Recall':<10} {'ID_Sw':<10} {'Frag':<10} {'FP':<10} {'FN':<10}\n")
        f.write("-"*100 + "\n")
        
        for r in summary_results:
            f.write(f"{r['detector']:<15} "
                   f"{r['mota']:>6.2f}%   "
                   f"{r['idf1']:>6.2f}%   "
                   f"{r['motp']:>6.3f}    "
                   f"{r['precision']:>8.2f}%   "
                   f"{r['recall']:>6.2f}%   "
                   f"{r['id_sw']:>6}    "
                   f"{r['frag']:>6}   "
                   f"{r['fp']:>6}   "
                   f"{r['fn']:>8}\n")
        
        # Calculate and write average
        if summary_results:
            avg_mota = np.mean([r['mota'] for r in summary_results])
            avg_idf1 = np.mean([r['idf1'] for r in summary_results])
            avg_motp = np.mean([r['motp'] for r in summary_results])
            avg_precision = np.mean([r['precision'] for r in summary_results])
            avg_recall = np.mean([r['recall'] for r in summary_results])
            avg_id_sw = sum([r['id_sw'] for r in summary_results])
            avg_frag = sum([r['frag'] for r in summary_results])
            avg_fp = sum([r['fp'] for r in summary_results])
            avg_fn = sum([r['fn'] for r in summary_results])
            
            f.write("-"*100 + "\n")
            f.write(f"{'AVERAGE':<15} "
                   f"{avg_mota:>6.2f}%   "
                   f"{avg_idf1:>6.2f}%   "
                   f"{avg_motp:>6.3f}    "
                   f"{avg_precision:>8.2f}%   "
                   f"{avg_recall:>6.2f}%   "
                   f"{avg_id_sw:>6}    "
                   f"{avg_frag:>6}   "
                   f"{avg_fp:>6}   "
                   f"{avg_fn:>8}\n")
            f.write("="*100 + "\n\n")
        
        # Write detailed results for each detector
        for detector in ['DPM', 'FRCNN', 'SDP']:
            if detector not in detector_metrics:
                continue
            
            f.write(f"\n{detector} Detailed Results:\n")
            f.write("-"*100 + "\n")
            
            # Get accumulators and names
            accs = [acc for _, acc in detector_metrics[detector]]
            names = [name for name, _ in detector_metrics[detector]]
            
            # Compute detailed metrics
            mh = mm.metrics.create()
            summary = mh.compute_many(
                accs,
                metrics=['mota', 'motp', 'idf1', 'precision', 'recall', 
                        'num_switches', 'num_fragmentations', 'num_false_positives', 'num_misses'],
                names=names
            )
            
            f.write(summary.to_string())
            f.write("\n\n")
    
    print(f"\nDetailed summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description='BoT-SORT using BoxMOT library on MOT17')
    
    parser.add_argument('--data_root', type=str, 
                       default='D:\Learn\Year4\KLTN\Dataset\MOT17',
                       help='Root directory of MOT17 dataset')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'test'],
                       help='Dataset split to use')
    parser.add_argument('--output_dir', type=str, 
                       default='results_boxmot_botsort',
                       help='Directory to save tracking results')
    
    # BoxMOT BoTSORT parameters
    parser.add_argument('--track_thresh', type=float, default=0.5,
                       help='High confidence threshold for track activation')
    parser.add_argument('--track_low_thresh', type=float, default=0.1,
                       help='Low confidence threshold for second association')
    parser.add_argument('--new_track_thresh', type=float, default=0.6,
                       help='Threshold for creating new tracks')
    parser.add_argument('--match_thresh', type=float, default=0.8,
                       help='IoU threshold for matching')
    parser.add_argument('--proximity_thresh', type=float, default=0.5,
                       help='Proximity threshold for ReID matching')
    parser.add_argument('--appearance_thresh', type=float, default=0.25,
                       help='Appearance similarity threshold')
    parser.add_argument('--track_buffer', type=int, default=30,
                       help='Number of frames to keep lost tracks')
    parser.add_argument('--cmc_method', type=str, default='sof',
                       choices=['ecc', 'orb', 'sift', 'sof', 'none'],
                       help='Camera Motion Compensation method (sof=Sparse Optical Flow)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to run tracker on')
    parser.add_argument('--half', action='store_true',
                       help='Use FP16 half precision (only for CUDA)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("BoT-SORT using BoxMOT Library - MOT17 Benchmark")
    print("="*80)
    print(f"Data root: {args.data_root}")
    print(f"Split: {args.split}")
    print(f"Output dir: {args.output_dir}")
    print("\nBoT-SORT Parameters:")
    print(f"  track_thresh:       {args.track_thresh}")
    print(f"  track_low_thresh:   {args.track_low_thresh}")
    print(f"  new_track_thresh:   {args.new_track_thresh}")
    print(f"  match_thresh:       {args.match_thresh}")
    print(f"  proximity_thresh:   {args.proximity_thresh}")
    print(f"  appearance_thresh:  {args.appearance_thresh}")
    print(f"  track_buffer:       {args.track_buffer}")
    print(f"  cmc_method:         {args.cmc_method}")
    print(f"  device:             {args.device}")
    print(f"  half:               {args.half}")
    print("="*80)
    
    # Initialize dataset
    dataset = MOT17Dataset(args.data_root, args.split)
    
    # Run tracking
    track_mot17_with_botsort(dataset, args.output_dir, args)
    
    print("\n" + "="*80)
    print("Tracking completed!")
    print(f"Results saved to: {args.output_dir}")
    print("="*80)
    
    # Compute metrics if ground truth available
    if args.split == 'train':
        result = compute_mot_metrics(args.data_root, args.output_dir)
        if result:
            detector_metrics, summary_results = result
            save_detailed_summary(Path(args.output_dir), detector_metrics, summary_results, args)


if __name__ == '__main__':
    main()
