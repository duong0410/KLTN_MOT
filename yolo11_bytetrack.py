import cv2
import numpy as np
import time
import psutil
import os
import traceback
from collections import deque

from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# =================================================================
#   GPU & CPU MONITORING
# =================================================================
try:
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    def get_gpu_usage():
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu
        except Exception:
            return 0
except Exception:
    handle = None
    def get_gpu_usage():
        return 0

def get_cpu_usage():
    try:
        return psutil.cpu_percent(interval=None)
    except Exception:
        return 0

# =================================================================
#   KALMAN FILTER 
# =================================================================
class KalmanBoxTracker:
    """
    Kalman Filter for bbox tracking với state [x, y, s, r, vx, vy, vs]
    x, y: center
    s: scale (area)
    r: aspect ratio (width/height)
    """
    count = 0
    
    def __init__(self, bbox, score=None, class_id=None):
        # State: [cx, cy, scale, ratio, vcx, vcy, vs]
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # State transition matrix
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        # Measurement noise 
        self.kf.R[2:, 2:] *= 10.0
        
        # Process noise
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        # Covariance matrix
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        
        # Initialize state
        self.kf.x[:4] = self._bbox_to_z(bbox)
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = deque(maxlen=30)
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        
        # Store score and class_id (for visualization only, not for matching)
        self.score = score
        self.class_id = class_id
        
    def _bbox_to_z(self, bbox):
        """Convert [x1,y1,x2,y2] to [cx, cy, s, r]"""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))
    
    def _z_to_bbox(self, z):
        """Convert [cx, cy, s, r] to [x1, y1, x2, y2]"""
        w = np.sqrt(z[2] * z[3])
        h = z[2] / w
        return np.array([
            z[0] - w/2.,
            z[1] - h/2.,
            z[0] + w/2.,
            z[1] + h/2.
        ]).reshape((1, 4))
    
    def update(self, bbox, score=None, class_id=None):
        """Update với detection mới"""
        self.time_since_update = 0
        self.history = deque(maxlen=30)
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._bbox_to_z(bbox))
        
        # Update score and class_id (for display only)
        if score is not None:
            self.score = score
        if class_id is not None:
            self.class_id = class_id
    
    def predict(self):
        """Predict vị trí tiếp theo"""
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
        
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        
        self.history.append(self._z_to_bbox(self.kf.x[:4]))
        return self.history[-1]
    
    def get_state(self):
        """Get bbox hiện tại"""
        return self._z_to_bbox(self.kf.x[:4])

# =================================================================
#   IOU CALCULATION 
# =================================================================
def iou_batch(bboxes1, bboxes2):
    """
    Tính IOU giữa 2 batch bboxes
    bboxes: (N, 4) với format [x1, y1, x2, y2]
    Returns: (N, M) IOU matrix
    """
    if len(bboxes1) == 0 or len(bboxes2) == 0:
        return np.zeros((len(bboxes1), len(bboxes2)))
    
    bboxes1 = np.asarray(bboxes1)
    bboxes2 = np.asarray(bboxes2)
    
    # Expand dims for broadcasting
    bb1 = np.expand_dims(bboxes1, 1)  # (N, 1, 4)
    bb2 = np.expand_dims(bboxes2, 0)  # (1, M, 4)
    
    # Calculate intersection
    xx1 = np.maximum(bb1[..., 0], bb2[..., 0])
    yy1 = np.maximum(bb1[..., 1], bb2[..., 1])
    xx2 = np.minimum(bb1[..., 2], bb2[..., 2])
    yy2 = np.minimum(bb1[..., 3], bb2[..., 3])
    
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    intersection = w * h
    
    # Calculate union
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
    area1 = np.expand_dims(area1, 1)
    area2 = np.expand_dims(area2, 0)
    
    union = area1 + area2 - intersection
    
    iou = intersection / np.maximum(union, 1e-6)
    
    return iou

# =================================================================
#   BYTETRACK
# =================================================================
class ByteTrack:
    def __init__(self, 
                 det_conf_high=0.5,
                 det_conf_low=0.1,
                 match_thresh_high=0.8,
                 match_thresh_low=0.5,
                 max_age=30,
                 min_hits=3):
        """
        ByteTrack với 2-phase matching 
        
        Args:
            det_conf_high: Ngưỡng confidence cho high-score detections
            det_conf_low: Ngưỡng confidence cho low-score detections
            match_thresh_high: IOU threshold cho phase 1 matching
            match_thresh_low: IOU threshold cho phase 2 matching
            max_age: Số frame tối đa track bị miss trước khi xóa
            min_hits: Số hit tối thiểu để track được confirmed
        """
        self.det_conf_high = det_conf_high
        self.det_conf_low = det_conf_low
        self.match_thresh_high = match_thresh_high
        self.match_thresh_low = match_thresh_low
        self.max_age = max_age
        self.min_hits = min_hits
        
        self.trackers = []
        self.frame_count = 0
        
    def _linear_assignment(self, cost_matrix, thresh):
        """
        Hungarian algorithm với threshold 
        Returns: matches, unmatched_a, unmatched_b
        """
        if cost_matrix.size == 0:
            return np.empty((0, 2), dtype=int), \
                   np.arange(cost_matrix.shape[0], dtype=int), \
                   np.arange(cost_matrix.shape[1], dtype=int)
        
        # Convert IOU to cost (1 - IOU)
        cost_matrix = 1 - cost_matrix
        cost_matrix[cost_matrix > 1 - thresh] = 1e6  # Filter by threshold only
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matches = []
        unmatched_a = []
        unmatched_b = []
        
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 1e5:  # Valid match
                matches.append([int(r), int(c)])
            else:
                unmatched_a.append(int(r))
        
        matches = np.array(matches, dtype=int) if len(matches) > 0 else np.empty((0, 2), dtype=int)
        
        # Find unmatched detections
        for c in range(cost_matrix.shape[1]):
            if c not in col_ind:
                unmatched_b.append(int(c))
        
        # Find remaining unmatched tracks
        for r in range(cost_matrix.shape[0]):
            if r not in row_ind or r in unmatched_a:
                if r not in unmatched_a:
                    unmatched_a.append(int(r))
        
        return matches, np.array(unmatched_a, dtype=int), np.array(unmatched_b, dtype=int)
    
    def update(self, detections, scores, class_ids=None):
        """
        ByteTrack 2-phase update 
        
        Args:
            detections: numpy array (N, 4) [x1, y1, x2, y2]
            scores: numpy array (N,) confidence scores
            class_ids: numpy array (N,) class IDs (for display only)
        """
        self.frame_count += 1
        
        # ==== PREDICT ALL TRACKS ====
        for trk in self.trackers:
            trk.predict()
        
        # ==== SPLIT DETECTIONS ====
        if len(detections) > 0:
            high_indices = scores >= self.det_conf_high
            low_indices = (scores >= self.det_conf_low) & (scores < self.det_conf_high)
            
            dets_high = detections[high_indices]
            scores_high = scores[high_indices]
            class_ids_high = class_ids[high_indices] if class_ids is not None else None
            
            dets_low = detections[low_indices]
            scores_low = scores[low_indices]
            class_ids_low = class_ids[low_indices] if class_ids is not None else None
        else:
            dets_high = np.empty((0, 4))
            scores_high = np.empty((0,))
            class_ids_high = None
            dets_low = np.empty((0, 4))
            scores_low = np.empty((0,))
            class_ids_low = None
        
        # ==== PHASE 1: Match với HIGH-SCORE detections ====
        if len(self.trackers) > 0 and len(dets_high) > 0:
            trk_bboxes = np.array([trk.get_state()[0] for trk in self.trackers])
            iou_matrix = iou_batch(trk_bboxes, dets_high)
            matches, unmatched_trks, unmatched_dets = \
                self._linear_assignment(iou_matrix, self.match_thresh_high)
        else:
            matches = np.empty((0, 2), dtype=int)
            unmatched_trks = np.arange(len(self.trackers), dtype=int)
            unmatched_dets = np.arange(len(dets_high), dtype=int)
        
        # ==== UPDATE MATCHED TRACKS ====
        for m in matches:
            trk_idx = int(m[0])
            det_idx = int(m[1])
            score = float(scores_high[det_idx])
            cls_id = int(class_ids_high[det_idx]) if class_ids_high is not None else None
            self.trackers[trk_idx].update(dets_high[det_idx], score, cls_id)
        
        # ==== PHASE 2: Match UNMATCHED tracks với LOW-SCORE detections ====
        if len(unmatched_trks) > 0 and len(dets_low) > 0:
            unmatched_trackers = [self.trackers[i] for i in unmatched_trks]
            unmatched_trk_bboxes = np.array([trk.get_state()[0] for trk in unmatched_trackers])
            iou_matrix_low = iou_batch(unmatched_trk_bboxes, dets_low)
            matches_low, unmatched_trks_2, unmatched_dets_low = \
                self._linear_assignment(iou_matrix_low, self.match_thresh_low)
            
            # Update matched tracks
            for m in matches_low:
                trk_idx = int(unmatched_trks[int(m[0])])
                det_idx = int(m[1])
                score = float(scores_low[det_idx])
                cls_id = int(class_ids_low[det_idx]) if class_ids_low is not None else None
                self.trackers[trk_idx].update(dets_low[det_idx], score, cls_id)
            
            unmatched_trks_2 = np.asarray(unmatched_trks_2, dtype=int)
            unmatched_trks = np.asarray(unmatched_trks, dtype=int)[unmatched_trks_2]
        
        # ==== CREATE NEW TRACKS từ HIGH-SCORE detections ====
        for i in unmatched_dets:
            i = int(i)
            if i < len(dets_high):
                score = float(scores_high[i])
                cls_id = int(class_ids_high[i]) if class_ids_high is not None else None
                trk = KalmanBoxTracker(dets_high[i], score, cls_id)
                self.trackers.append(trk)
        
        # ==== REMOVE DEAD TRACKS ====
        i = len(self.trackers)
        ret = []
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            
            if trk.time_since_update < 1:
                if trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits:
                    ret.append((d, trk.id, trk.hits, trk.class_id))
            
            i -= 1
            
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        
        if len(ret) > 0:
            return np.array([x[0] for x in ret]), \
                   np.array([x[1] for x in ret]), \
                   np.array([x[2] for x in ret]), \
                   np.array([x[3] if x[3] is not None else -1 for x in ret])
        
        return np.empty((0, 4)), np.empty((0,), dtype=int), np.empty((0,), dtype=int), np.empty((0,), dtype=int)

# =================================================================
#   MAIN TRACKING FUNCTION
# =================================================================
def run_tracking(video_path, 
                 output_path="output_tracked.mp4", 
                 model_path="best.pt",
                 save_output=True, 
                 show_window=True):
    
    # Create output directory
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    # Load YOLO model
    print("Loading model:", model_path)
    model = YOLO(model_path)
    
    # Initialize ByteTrack 
    tracker = ByteTrack(
        det_conf_high=0.5,
        det_conf_low=0.1,
        match_thresh_high=0.8,
        match_thresh_low=0.5,
        max_age=30,
        min_hits=3
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Output will be saved to: {output_path}")
    else:
        writer = None
    
    # Stats
    fps_list = []
    gpu_list = []
    cpu_list = []
    frame_count = 0
    
    # Color palette for tracks
    np.random.seed(42)
    colors = {}
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            t_start = time.time()
            
            # ==== YOLO DETECTION ====
            results = model(frame, verbose=False)[0]
            
            # Extract detections
            if hasattr(results, 'boxes') and len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()
                scores = results.boxes.conf.cpu().numpy()
                class_ids = results.boxes.cls.cpu().numpy().astype(int)
            else:
                boxes = np.empty((0, 4))
                scores = np.empty((0,))
                class_ids = np.empty((0,), dtype=int)
            
            # ==== BYTETRACK UPDATE ====
            tracked_bboxes, track_ids, hits, tracked_class_ids = tracker.update(boxes, scores, class_ids)
            
            # ==== DRAW RESULTS ====
            class_names = ['car', 'truck', 'bus', 'motor', 'bicycle', 'person']
            
            for bbox, track_id, hit, cls_id in zip(tracked_bboxes, track_ids, hits, tracked_class_ids):
                x1, y1, x2, y2 = map(int, bbox)
                
                # Clip coordinates
                x1 = max(0, min(x1, width-1))
                y1 = max(0, min(y1, height-1))
                x2 = max(0, min(x2, width-1))
                y2 = max(0, min(y2, height-1))
                
                # Get consistent color cho mỗi ID
                if track_id not in colors:
                    colors[track_id] = tuple(map(int, np.random.randint(0, 255, 3)))
                color = colors[track_id]
                
                # Get class name
                if cls_id >= 0 and cls_id < len(class_names):
                    class_name = class_names[cls_id]
                else:
                    class_name = f"cls{cls_id}" if cls_id >= 0 else "unknown"
                
                # Draw bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{class_name}:{track_id}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1-20), (x1+w, y1), color, -1)
                cv2.putText(frame, label, (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            
            # ==== STATS ====
            elapsed = time.time() - t_start
            fps_val = 1.0 / max(elapsed, 1e-6)
            fps_list.append(fps_val)
            gpu_list.append(get_gpu_usage())
            cpu_list.append(get_cpu_usage())
            
            # Draw stats
            cv2.putText(frame, f"FPS: {fps_val:.1f} (avg: {np.mean(fps_list):.1f})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.putText(frame, f"GPU: {gpu_list[-1]:.0f}% | CPU: {cpu_list[-1]:.0f}%", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            cv2.putText(frame, f"Tracks: {len(track_ids)} | Detections: {len(boxes)}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            
            # Show frame
            if show_window:
                cv2.imshow("ByteTrack", frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    print("Early exit requested")
                    break
            
            # Write frame
            if save_output and writer is not None:
                writer.write(frame)
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nException: {e}")
        traceback.print_exc()
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show_window:
            cv2.destroyAllWindows()
    
    # ==== SUMMARY ====
    print("\n" + "="*50)
    print("TRACKING SUMMARY")
    print("="*50)
    print(f"Total frames: {frame_count}")
    print(f"Average FPS: {np.mean(fps_list):.2f}")
    print(f"Average GPU usage: {np.mean(gpu_list):.1f}%")
    print(f"Average CPU usage: {np.mean(cpu_list):.1f}%")
    print(f"Total unique tracks: {len(colors)}")
    if save_output:
        print(f"Output saved to: {os.path.abspath(output_path)}")
    print("="*50)

# =================================================================
#   RUN
# =================================================================
if __name__ == "__main__":
    run_tracking(
        video_path="D:\\Download\\Video Project 3.mp4",
        output_path="output_tracked_3.mp4",
        model_path="Dataset\\traffic_yolo\\runs\\yolo11s_traffic2\\weights\\best.pt",
        save_output=True,
        show_window=True
    )