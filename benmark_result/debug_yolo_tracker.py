#!/usr/bin/env python3
"""Debug YOLO + ByteTrack tracking"""

import numpy as np
import sys
sys.path.insert(0, '.')

# Import from benmark_bytetrack
from benmark_bytetrack import YOLODetector, BYTETracker, YOLO_PERSON_CLASS_ID

# Test trên frame đầu tiên
img_path = r"D:\Learn\Year4\KLTN\Dataset\MOT17\train\MOT17-02-FRCNN\img1\000001.jpg"

print("="*80)
print("DEBUG YOLO + ByteTrack")
print("="*80)

# 1. Test YOLO detection
print("\n1. YOLO Detection:")
detector = YOLODetector(
    model_path='yolo11m.pt',
    conf_thresh=0.2,  # TANG LEN 0.2!
    device='cuda:0',
    person_class_id=YOLO_PERSON_CLASS_ID
)

frame_dets = detector.detect(img_path)
print(f"   So detection: {len(frame_dets)}")
if len(frame_dets) > 0:
    print(f"   Sample detection (raw): {frame_dets[0]}")
    print(f"   Format: [x, y, w, h, score]")

# 2. Convert to tracker format
print("\n2. Convert to [x1, y1, x2, y2, score]:")
converted_dets = []
for det in frame_dets:
    x, y, w, h, score = det
    converted_dets.append([x, y, x + w, y + h, score])
converted_dets = np.array(converted_dets)

if len(converted_dets) > 0:
    print(f"   Sample converted: {converted_dets[0]}")
    print(f"   Format: [x1, y1, x2, y2, score]")

# 3. Check ByteTrack filtering
print("\n3. ByteTrack Filtering:")
tracker = BYTETracker(
    det_conf_high=0.2,  # Match YOLO
    det_conf_low=0.1,
    new_track_thresh=0.2
)
print(f"   det_conf_high: {tracker.det_conf_high}")
print(f"   det_conf_low: {tracker.det_conf_low}")
print(f"   new_track_thresh: {tracker.new_track_thresh}")

# Count detections by threshold
high_score = converted_dets[converted_dets[:, 4] > tracker.det_conf_high]
low_score = converted_dets[(converted_dets[:, 4] > tracker.det_conf_low) & 
                            (converted_dets[:, 4] <= tracker.det_conf_high)]
very_low = converted_dets[converted_dets[:, 4] <= tracker.det_conf_low]

print(f"   High score (>{tracker.det_conf_high}): {len(high_score)}")
print(f"   Low score ({tracker.det_conf_low}-{tracker.det_conf_high}): {len(low_score)}")
print(f"   Very low (<{tracker.det_conf_low}): {len(very_low)} ← BỊ BỎ!")

# 4. Run tracker
print("\n4. Run Tracker:")
img_shape = (1080, 1920)
online_targets = tracker.update(converted_dets, img_shape)
print(f"   Tracked objects: {len(online_targets)}")

if len(online_targets) > 0:
    track = online_targets[0]
    print(f"   Sample track:")
    print(f"     ID: {track.track_id}")
    print(f"     tlwh: {track.tlwh}")
    print(f"     score: {track.score}")

# 5. Ground truth comparison
print("\n5. Ground Truth:")
gt_file = r"D:\Learn\Year4\KLTN\Dataset\MOT17\train\MOT17-02-FRCNN\gt\gt.txt"
gt_frame1 = []
with open(gt_file) as f:
    for line in f:
        parts = line.strip().split(',')
        if int(parts[0]) == 1 and int(parts[7]) == 1 and int(parts[6]) == 1:
            gt_frame1.append(parts)

print(f"   GT objects frame 1: {len(gt_frame1)}")
print(f"   Detection rate: {len(online_targets)}/{len(gt_frame1)} = {len(online_targets)/len(gt_frame1)*100:.1f}%")

print("\n" + "="*80)
print("PHÂN TÍCH:")
print(f"- YOLO detect: {len(frame_dets)}")
print(f"- Sau khi filter ByteTrack: {len(high_score)} high + {len(low_score)} low")
print(f"- Bị bỏ: {len(very_low)} detections có score < {tracker.det_conf_low}")
print(f"- Output: {len(online_targets)} tracks")
print(f"- Ground truth: {len(gt_frame1)} people")
print("="*80)
