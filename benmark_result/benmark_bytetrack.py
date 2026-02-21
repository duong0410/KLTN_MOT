#!/usr/bin/env python3
"""
ByteTrack implementation for MOT17 benchmark
This script runs ByteTrack tracking algorithm on MOT17 dataset and evaluates performance
"""

import os
import numpy as np
import cv2
from collections import defaultdict
import argparse
from pathlib import Path
import motmetrics as mm
from typing import List, Tuple, Optional
import scipy.linalg
import lap


# ============================================================================
# ByteTrack Core Classes
# ============================================================================

class STrack:
    """Single target track with Kalman filter state"""
    
    shared_kalman = None
    track_id_count = 0
    
    def __init__(self, tlwh, score):
        # tlwh format: top-left width height
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        
        self.score = score
        self.tracklet_len = 0
        
        # These will be set in activate()
        self.track_id = 0
        self.frame_id = 0
        self.start_frame = 0
        
        
        self.state = TrackState.New
        
    def predict(self):
        """Predict next state using Kalman filter"""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)
        
    @staticmethod
    def multi_predict(stracks):
        """Predict multiple tracks"""
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov
                
    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        # ByteTrack official: activate on frame 1, otherwise need confirmation
        if frame_id == 1:
            self.is_activated = True
        # For other frames, will be activated in update() after min_hits
        self.frame_id = frame_id
        self.start_frame = frame_id
        
    def re_activate(self, new_track, frame_id, new_id=False):
        """Reactivate a lost track"""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        # ByteTrack official: reset tracklet_len to 0
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        
    def update(self, new_track, frame_id, min_hits=1):
        """
        Update a matched track
        
        Args:
            new_track: New detection to update with
            frame_id: Current frame ID
            min_hits: Minimum hits before track is activated (default: 1, matching BoxMOT)
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        
        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh)
        )
        
        # BoxMOT-style: Activate immediately (min_hits=1 by default)
        if self.tracklet_len >= min_hits:
            self.is_activated = True
            self.state = TrackState.Tracked
        
        self.score = new_track.score
    
    def mark_lost(self):
        """Mark track as lost"""
        self.state = TrackState.Lost
    
    def mark_removed(self):
        """Mark track as removed"""
        self.state = TrackState.Removed
        
    @property
    def tlwh(self):
        """Get current position in tlwh format"""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret
    
    @property
    def tlbr(self):
        """Convert tlwh to tlbr (top-left, bottom-right)"""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret
    
    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert tlwh to xyah (center x, center y, aspect ratio, height)"""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
    
    @property
    def end_frame(self):
        """Get the last frame id when the track was updated"""
        return self.frame_id
    
    @staticmethod
    def next_id():
        STrack.track_id_count += 1
        return STrack.track_id_count
    
    def __repr__(self):
        return f'Track_{self.track_id}_({self.start_frame}-{self.frame_id})'


class TrackState:
    """Enumeration type for track state"""
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class KalmanFilter:
    """Kalman filter for track state estimation"""
    
    def __init__(self):
        ndim, dt = 4, 1.
        
        # Create Kalman filter model matrices
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        
        # Motion and observation uncertainty
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160
        
    def initiate(self, measurement):
        """Create track from unassociated measurement"""
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance
    
    def predict(self, mean, covariance):
        """Run Kalman filter prediction step"""
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        
        return mean, covariance
    
    def project(self, mean, covariance):
        """Project state distribution to measurement space"""
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        innovation_cov = np.diag(np.square(std))
        
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov
    
    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step"""
        projected_mean, projected_cov = self.project(mean, covariance)
        
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean
        
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance
    
    def multi_predict(self, mean, covariance):
        """Run prediction step for multiple tracks"""
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3]
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T
        
        motion_cov = []
        for i in range(len(mean)):
            motion_cov.append(np.diag(sqr[i]))
        motion_cov = np.asarray(motion_cov)
        
        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov
        
        return mean, covariance


class BYTETracker:
    """
    ByteTrack multi-object tracker
    
    ByteTrack uses 2-phase matching strategy:
    - Phase 1: Match tracks with high-score detections (det_conf_high)
    - Phase 2: Match remaining tracks with low-score detections (det_conf_low)
    
    This approach reduces miss/fragmentation during occlusion and motion blur.
    """
    
    def __init__(self, det_conf_high=0.5, det_conf_low=0.1, new_track_thresh=0.6,
                 match_thresh_high=0.8, match_thresh_low=0.5, track_buffer=30, min_hits=1):
        """
        Args:
            det_conf_high: Confidence threshold for high-score detections (default: 0.5)
            det_conf_low: Confidence threshold for low-score detections (default: 0.1)
            new_track_thresh: Threshold for creating new tracks (default: 0.6, higher than det_conf_high)
            match_thresh_high: Cost threshold for first association (default: 0.8, IoU > 0.2, like BoxMOT)
            match_thresh_low: Cost threshold for second association (default: 0.5, IoU > 0.5)
            track_buffer: Number of frames to keep lost tracks (default: 30)
            min_hits: Minimum hits before track is confirmed (default: 1, matching BoxMOT behavior)
        """
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        
        self.frame_id = 0
        
        # ByteTrack thresholds
        self.det_conf_high = det_conf_high
        self.det_conf_low = det_conf_low
        self.new_track_thresh = new_track_thresh  # For creating new tracks
        self.match_thresh_high = match_thresh_high
        self.match_thresh_low = match_thresh_low
        self.max_time_lost = track_buffer
        self.min_hits = min_hits  # BoxMOT-style: minimum hits before confirmed
        
        self.kalman_filter = KalmanFilter()
        STrack.shared_kalman = self.kalman_filter
        
    def update(self, output_results, img_shape):
        """
        Update tracker with new detections using ByteTrack algorithm (Algorithm 1)
        
        Following the pseudo-code:
        - Line 6-13: Split detections D into D_high and D_low by threshold τ
        - Line 14-16: Predict T with Kalman Filter
        - Line 17: First association - Match T and D_high (Similarity#1)
        - Line 20: Second association - Match T_remain and D_low (Similarity#2)
        - Line 22: Delete unmatched tracks T_re-remain from T
        - Line 23-25: Initialize new tracks from D_remain
        
        Args:
            output_results: numpy array of detections [x1, y1, x2, y2, score, class]
            img_shape: tuple of (height, width)
            
        Returns:
            list of active tracks T
        """
        self.frame_id += 1  # Frame j_t in algorithm
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        
        # Parse detection results from detector Det(f_t)
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]
        
        # ======================================================================
        # Algorithm 1, Line 6-13: Split detections by score threshold
        # for d in D_t do
        #     if d.score > τ then D_high ← D_high ∪ {d}
        #     else D_low ← D_low ∪ {d}
        # ======================================================================
        # D_high: High-score detections (threshold τ = det_conf_high ≈ 0.5)
        remain_inds_high = scores > self.det_conf_high
        dets_high = bboxes[remain_inds_high]
        scores_high = scores[remain_inds_high]
        
        # D_low: Low-score detections (det_conf_low ≈ 0.1 < score ≤ det_conf_high)
        # These help recover tracks during occlusion/motion blur
        remain_inds_low = np.logical_and(scores > self.det_conf_low, scores <= self.det_conf_high)
        dets_low = bboxes[remain_inds_low]
        scores_low = scores[remain_inds_low]
        
        # Convert to STrack objects
        if len(dets_high) > 0:
            detections_high = [STrack(tlbr_to_tlwh(tlbr), s) for tlbr, s in zip(dets_high, scores_high)]
        else:
            detections_high = []
            
        if len(dets_low) > 0:
            detections_low = [STrack(tlbr_to_tlwh(tlbr), s) for tlbr, s in zip(dets_low, scores_low)]
        else:
            detections_low = []
            
        # Separate confirmed tracks (T) and unconfirmed tracks
        # T: tracked stracks that are already activated
        unconfirmed = []
        tracked_stracks = []  # This is T in the algorithm
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        
        # ======================================================================
        # CRITICAL FIX: First association with BOTH tracked AND lost tracks!
        # ======================================================================
        # Combine tracked and lost tracks into one pool
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        
        # Predict ALL tracks (both tracked and lost) with Kalman Filter
        STrack.multi_predict(strack_pool)
        
        # ======================================================================
        # Algorithm 1, Line 17: First Association
        # Associate (T + Lost) with D_high using high IoU threshold
        # ======================================================================
        dists = iou_distance(strack_pool, detections_high)
        matches, u_track, u_detection_high = linear_assignment(dists, thresh=self.match_thresh_high)
        
        # Update matched tracks
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections_high[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id, self.min_hits)
                activated_stracks.append(track)
            else:  # Lost track found again
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        
        # ======================================================================
        # Algorithm 1, Line 18-19: Get remaining detections and tracks
        # D_remain ← remaining detection boxes from D_high
        # T_remain ← remaining TRACKED tracks only (not lost!)
        # ======================================================================
        dets_remain_high = [detections_high[i] for i in u_detection_high]
        
        # CRITICAL: Only take TRACKED tracks from unmatched pool
        # ByteTrack official: lost tracks don't go to second association
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        
        # ======================================================================
        # Algorithm 1, Line 20: Second Association
        # Associate remaining TRACKED tracks with D_low
        # ======================================================================
        dists_low = iou_distance(r_tracked_stracks, detections_low)
        matches_low, u_track_remain, u_detection_low = linear_assignment(dists_low, thresh=self.match_thresh_low)
        
        # Update tracks matched with low-score detections
        for itracked, idet in matches_low:
            track = r_tracked_stracks[itracked]
            det = detections_low[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id, self.min_hits)
                activated_stracks.append(track)
            else:  # Should not happen, but handle it
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        
        # ======================================================================
        # Mark unmatched TRACKED tracks as lost
        # ======================================================================
        for it in u_track_remain:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        
        # ======================================================================
        # Handle unconfirmed tracks (newly created but not yet stable)
        # Match with REMAINING high detections
        # ======================================================================
        detections = [detections_high[i] for i in u_detection_high]
        dists = iou_distance(unconfirmed, detections)
        # Use same threshold as second association
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.5)
        
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id, self.min_hits)
            # Add to activated regardless - they need to stay in tracked_stracks to accumulate hits
            activated_stracks.append(unconfirmed[itracked])
        
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)
        
        # ======================================================================
        # Initialize new tracks
        # ByteTrack official: Use new_track_thresh (higher than track_high_thresh)
        # This prevents creating tracks from false positive detections
        # ======================================================================
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            # Add to activated - track needs to be in tracked_stracks to accumulate hits
            # Output filtering will handle only returning confirmed tracks
            activated_stracks.append(track)
            
        # Step 5: Remove lost tracks that exceeded max_time_lost
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
                
        # Update state
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        
        # Get current active tracks
        # BoxMOT-style: Output all activated tracks (min_hits=1 means immediate activation)
        output_stracks = [track for track in self.tracked_stracks 
                         if track.is_activated]
        
        return output_stracks


# ============================================================================
# Utility Functions
# ============================================================================

def tlbr_to_tlwh(tlbr):
    """Convert tlbr to tlwh format"""
    ret = np.asarray(tlbr).copy()
    ret[2:] -= ret[:2]
    return ret


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU between tracks
    """
    if len(atracks) > 0 and isinstance(atracks[0], np.ndarray) or len(btracks) > 0 and isinstance(btracks[0], np.ndarray):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
        
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if ious.size == 0:
        return ious
    
    ious = bbox_ious(np.ascontiguousarray(atlbrs, dtype=np.float32),
                     np.ascontiguousarray(btlbrs, dtype=np.float32))
    
    cost_matrix = 1 - ious
    return cost_matrix


def bbox_ious(atlbrs, btlbrs):
    """Compute IoU between two sets of boxes"""
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if ious.size == 0:
        return ious
    
    for i, atlbr in enumerate(atlbrs):
        for j, btlbr in enumerate(btlbrs):
            ious[i, j] = bbox_iou(atlbr, btlbr)
    return ious


def bbox_iou(box1, box2):
    """Compute IoU between two boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def linear_assignment(cost_matrix, thresh):
    """
    Perform linear assignment using Hungarian algorithm
    
    Args:
        cost_matrix: Cost matrix where cost = 1 - IoU
        thresh: Cost threshold - matches accepted if cost < thresh
                This means IoU > (1 - thresh)
                Example: thresh=0.8 means IoU > 0.2
                         thresh=0.5 means IoU > 0.5
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    
    return matches, unmatched_a, unmatched_b


def joint_stracks(tlista, tlistb):
    """Join two lists of tracks"""
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    """Remove tracks in tlistb from tlista"""
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    """Remove duplicate tracks"""
    pdist = iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = [], []
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb


# ============================================================================
# MOT17 Dataset Handler
# ============================================================================

class MOT17Dataset:
    """Handle MOT17 dataset loading and processing"""
    
    def __init__(self, data_root, split='train'):
        self.data_root = Path(data_root)
        self.split = split
        self.sequences = self._get_sequences()
        
    def _get_sequences(self):
        """Get all sequence names for the split"""
        split_dir = self.data_root / self.split
        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")
            
        sequences = []
        for seq_dir in sorted(split_dir.iterdir()):
            if seq_dir.is_dir() and not seq_dir.name.startswith('.'):
                sequences.append(seq_dir.name)
        return sequences
    
    def get_sequence_info(self, seq_name):
        """Get sequence information"""
        seq_dir = self.data_root / self.split / seq_name
        seqinfo_path = seq_dir / 'seqinfo.ini'
        
        info = {}
        if seqinfo_path.exists():
            with open(seqinfo_path, 'r') as f:
                for line in f:
                    if '=' in line and not line.startswith('['):
                        key, value = line.strip().split('=')
                        info[key] = value
        return info
    
    def load_detections(self, seq_name, detector='DPM'):
        """Load detection results for a sequence"""
        det_file = self.data_root / self.split / seq_name / 'det' / 'det.txt'
        
        if not det_file.exists():
            print(f"Warning: Detection file not found: {det_file}")
            return {}
        
        detections = defaultdict(list)
        with open(det_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 7:
                    continue
                    
                frame_id = int(parts[0])
                x, y, w, h = map(float, parts[2:6])
                score = float(parts[6]) if len(parts) > 6 else 1.0
                
                # Convert to [x1, y1, x2, y2, score]
                detection = [x, y, x + w, y + h, score]
                detections[frame_id].append(detection)
        
        return detections


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_mot17(tracker, dataset, output_dir, args):
    """
    Run tracker on MOT17 dataset and evaluate
    
    Args:
        tracker: BYTETracker instance
        dataset: MOT17Dataset instance
        output_dir: directory to save results
        args: command line arguments for per-detector tuning
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for seq_name in dataset.sequences:
        print(f"\nProcessing sequence: {seq_name}")
        
        # ============================================================
        # Adjust parameters per detector type to match BoxMOT behavior
        # ============================================================
        if 'DPM' in seq_name:
            tracker.det_conf_high = 0.5
            tracker.det_conf_low = 0.1
            tracker.new_track_thresh = 0.6
            tracker.match_thresh_high = 0.8  # Cost < 0.8 (IoU > 0.2, like BoxMOT)
            tracker.match_thresh_low = 0.5   # Cost < 0.5 (IoU > 0.5)
        elif 'FRCNN' in seq_name:
            tracker.det_conf_high = 0.5
            tracker.det_conf_low = 0.1
            tracker.new_track_thresh = 0.6
            tracker.match_thresh_high = 0.8  # Match BoxMOT
            tracker.match_thresh_low = 0.5
        else:  # SDP
            tracker.det_conf_high = 0.5
            tracker.det_conf_low = 0.1
            tracker.new_track_thresh = 0.6
            tracker.match_thresh_high = 0.8  # Match BoxMOT
            tracker.match_thresh_low = 0.5
        
        # Reset tracker state for new sequence
        tracker.frame_id = 0
        tracker.tracked_stracks = []
        tracker.lost_stracks = []
        tracker.removed_stracks = []
        STrack.track_id_count = 0
        
        # Load detections
        detections = dataset.load_detections(seq_name)
        seq_info = dataset.get_sequence_info(seq_name)
        
        img_height = int(seq_info.get('imHeight', 1080))
        img_width = int(seq_info.get('imWidth', 1920))
        img_shape = (img_height, img_width)
        
        # Process each frame
        seq_results = []
        num_frames = int(seq_info.get('seqLength', len(detections)))
        
        for frame_id in range(1, num_frames + 1):
            frame_dets = detections.get(frame_id, [])
            
            if len(frame_dets) == 0:
                continue
            
            # Convert to numpy array
            dets = np.array(frame_dets)
            
            # Update tracker
            online_targets = tracker.update(dets, img_shape)
            
            # Save results
            for track in online_targets:
                tlwh = track.tlwh
                tid = track.track_id
                seq_results.append([frame_id, tid, tlwh[0], tlwh[1], tlwh[2], tlwh[3]])
        
        # Save sequence results
        output_file = output_dir / f"{seq_name}.txt"
        with open(output_file, 'w') as f:
            for row in seq_results:
                f.write(f"{row[0]},{row[1]},{row[2]:.2f},{row[3]:.2f},{row[4]:.2f},{row[5]:.2f},1,-1,-1,-1\n")
        
        print(f"Saved results to: {output_file}")
        results[seq_name] = seq_results
    
    return results


def compute_mot_metrics(data_root, pred_dir):
    """
    Compute MOT metrics using py-motmetrics
    Separate results by detector type (DPM, FRCNN, SDP)
    Uses compute_many() like BoxMOT for proper aggregation
    
    Args:
        data_root: root directory of MOT17 dataset (contains train folder with sequences)
        pred_dir: directory containing prediction files
    """
    data_root = Path(data_root)
    pred_dir = Path(pred_dir)
    
    # Get all prediction files (exclude summary.txt)
    pred_files = sorted([f for f in pred_dir.glob('*.txt') if f.name != 'summary.txt' and f.name != 'evaluation_summary.txt'])
    
    if len(pred_files) == 0:
        print(f"No prediction files found in {pred_dir}")
        return None
    
    print(f"\nFound {len(pred_files)} prediction files")
    
    # Group by detector type - store accumulators per sequence
    detector_metrics = defaultdict(list)
    
    for pred_file in pred_files:
        seq_name = pred_file.stem
        
        # Determine detector type
        det_type = None
        if 'DPM' in seq_name:
            det_type = 'DPM'
        elif 'FRCNN' in seq_name:
            det_type = 'FRCNN'
        elif 'SDP' in seq_name:
            det_type = 'SDP'
        else:
            continue
        
        gt_file = data_root / seq_name / 'gt' / 'gt.txt'
        
        if not gt_file.exists():
            print(f"Warning: Ground truth file not found for {seq_name}")
            continue
        
        print(f"  Processing: {seq_name}")
        
        # Create separate accumulator for this sequence
        acc = mm.MOTAccumulator(auto_id=True)
        
        # Load ground truth
        gt_data = defaultdict(list)
        with open(gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 8:
                    continue
                frame_id = int(parts[0])
                track_id = int(parts[1])
                x, y, w, h = map(float, parts[2:6])
                conf = float(parts[6])
                cls = int(parts[7]) if len(parts) > 7 else 1
                
                # Filter: only consider pedestrian class (cls==1) and confident annotations (conf==1)
                if cls == 1 and conf == 1:
                    gt_data[frame_id].append({
                        'id': track_id,
                        'bbox': [x, y, x+w, y+h]
                    })
        
        # Load predictions
        pred_data = defaultdict(list)
        with open(pred_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 6:
                    continue
                frame_id = int(parts[0])
                track_id = int(parts[1])
                x, y, w, h = map(float, parts[2:6])
                
                pred_data[frame_id].append({
                    'id': track_id,
                    'bbox': [x, y, x+w, y+h]
                })
        
        # Compute metrics per frame
        all_frames = sorted(set(list(gt_data.keys()) + list(pred_data.keys())))
        
        for frame_id in all_frames:
            gt_ids = [obj['id'] for obj in gt_data.get(frame_id, [])]
            gt_bboxes = [obj['bbox'] for obj in gt_data.get(frame_id, [])]
            
            pred_ids = [obj['id'] for obj in pred_data.get(frame_id, [])]
            pred_bboxes = [obj['bbox'] for obj in pred_data.get(frame_id, [])]
            
            # Compute IoU distance matrix
            if len(gt_bboxes) > 0 and len(pred_bboxes) > 0:
                # Compute IoU manually to match BoxMOT
                ious = np.zeros((len(gt_bboxes), len(pred_bboxes)))
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
                
                dists = 1 - ious
            else:
                dists = np.empty((len(gt_ids), len(pred_ids)))
            
            acc.update(gt_ids, pred_ids, dists)
        
        # Store accumulator with sequence name
        detector_metrics[det_type].append((seq_name, acc))
    
    # Store results for each detector
    all_results = {}
    
    # Compute metrics for each detector using compute_many
    for det_type in ['DPM', 'FRCNN', 'SDP']:
        if det_type not in detector_metrics or len(detector_metrics[det_type]) == 0:
            continue
        
        print(f"\n{'='*80}")
        print(f"Evaluating {det_type} sequences")
        print(f"{'='*80}")
        
        # Get accumulators and names
        accs = [acc for _, acc in detector_metrics[det_type]]
        names = [name for name, _ in detector_metrics[det_type]]
        
        # Compute metrics using compute_many
        mh = mm.metrics.create()
        summary = mh.compute_many(
            accs,
            metrics=['mota', 'motp', 'idf1', 'precision', 'recall',
                    'num_switches', 'num_fragmentations', 'num_false_positives', 'num_misses'],
            names=names
        )
        
        all_results[det_type] = summary
    
    return all_results


# ============================================================================
# Main Script
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='ByteTrack on MOT17 - Optimized to reduce ID switches and improve IDF1')
    parser.add_argument('--data_root', type=str, 
                       default=r'd:\Learn\Year4\KLTN\Dataset\MOT17',
                       help='Path to MOT17 dataset root directory')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'test'],
                       help='Dataset split to use')
    parser.add_argument('--output_dir', type=str, default='./results_bytetrack',
                       help='Directory to save tracking results')
    
    # Parameters optimized to match BoxMOT performance
    parser.add_argument('--det_conf_high', type=float, default=0.5,
                       help='High-score detection threshold (default: 0.5, like BoxMOT track_thresh)')
    parser.add_argument('--det_conf_low', type=float, default=0.1,
                       help='Low-score detection threshold (default: 0.1)')
    parser.add_argument('--new_track_thresh', type=float, default=0.6,
                       help='Threshold for creating new tracks (default: 0.6)')
    parser.add_argument('--match_thresh_high', type=float, default=0.8,
                       help='Cost threshold for first association (default: 0.8, IoU > 0.2, matching BoxMOT)')
    parser.add_argument('--match_thresh_low', type=float, default=0.5,
                       help='Cost threshold for second association (default: 0.5, IoU > 0.5)')
    parser.add_argument('--track_buffer', type=int, default=30,
                       help='Frames to keep lost tracks (default: 30, like BoxMOT)')
    parser.add_argument('--min_hits', type=int, default=1,
                       help='Minimum hits before track is confirmed (default: 1, matching BoxMOT)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("ByteTrack on MOT17 - Optimized for BoxMOT-level Performance")
    print("="*80)
    print(f"Data root: {args.data_root}")
    print(f"Split: {args.split}")
    print(f"Output dir: {args.output_dir}")
    print("\nParameters (matching BoxMOT):")
    print(f"  det_conf_high:     {args.det_conf_high} (high-score detection threshold)")
    print(f"  det_conf_low:      {args.det_conf_low} (low-score detection threshold)")
    print(f"  new_track_thresh:  {args.new_track_thresh} (new track creation threshold)")
    print(f"  match_thresh_high: {args.match_thresh_high} (cost thresh, IoU > {1-args.match_thresh_high:.1f})")
    print(f"  match_thresh_low:  {args.match_thresh_low} (cost thresh, IoU > {1-args.match_thresh_low:.1f})")
    print(f"  track_buffer:      {args.track_buffer} frames (like BoxMOT)")
    print(f"  min_hits:          {args.min_hits} detections (matching BoxMOT)")
    print("="*80)
    
    # Initialize dataset
    dataset = MOT17Dataset(args.data_root, args.split)
    print(f"\nFound {len(dataset.sequences)} sequences")
    
    # Initialize tracker with BoxMOT-optimized parameters
    tracker = BYTETracker(
        det_conf_high=args.det_conf_high,
        det_conf_low=args.det_conf_low,
        new_track_thresh=args.new_track_thresh,
        match_thresh_high=args.match_thresh_high,
        match_thresh_low=args.match_thresh_low,
        track_buffer=args.track_buffer,
        min_hits=args.min_hits
    )
    
    # Run tracking
    print("\nRunning ByteTrack with BoxMOT-optimized parameters...")
    results = evaluate_mot17(tracker, dataset, args.output_dir, args)
    
    print("\n" + "="*80)
    print("Tracking completed!")
    print(f"Results saved to: {args.output_dir}")
    print("="*80)
    
    # Compute metrics if ground truth available
    if args.split == 'train':
        print("\nComputing MOT metrics...")
        gt_dir = Path(args.data_root) / args.split
        
        try:
            all_results = compute_mot_metrics(gt_dir, Path(args.output_dir))
            
            if all_results is not None and len(all_results) > 0:
                print("\n" + "="*100)
                print(f"{'MOT17 EVALUATION RESULTS - ByteTrack 2-Phase Matching':^100}")
                print("="*100)
                
                # Prepare summary data
                summary_data = []
                
                for det_type in ['DPM', 'FRCNN', 'SDP']:
                    if det_type not in all_results:
                        continue
                    
                    summary = all_results[det_type]
                    
                    # Debug: Print summary structure
                    print(f"\n{det_type} summary index: {list(summary.index)}")
                    print(f"Summary shape: {summary.shape}")
                    
                    # Get overall metrics (last row is typically the overall summary)
                    if len(summary) > 0:
                        row = summary.iloc[-1]
                        summary_data.append({
                            'Detector': det_type,
                            'MOTA': row['mota'] * 100,
                            'IDF1': row['idf1'] * 100,
                            'MOTP': row['motp'],
                            'Precision': row['precision'] * 100,
                            'Recall': row['recall'] * 100,
                            'ID_Sw': int(row['num_switches']),
                            'Frag': int(row['num_fragmentations']),
                            'FP': int(row['num_false_positives']),
                            'FN': int(row['num_misses'])
                        })
                
                # Print formatted table
                print(f"\n{'Detector':<15} {'MOTA':<10} {'IDF1':<10} {'MOTP':<10} {'Precision':<12} {'Recall':<10} {'ID_Sw':<10} {'Frag':<10} {'FP':<10} {'FN':<10}")
                print("-" * 100)
                
                for data in summary_data:
                    print(f"{data['Detector']:<15} "
                          f"{data['MOTA']:>6.2f}%   "
                          f"{data['IDF1']:>6.2f}%   "
                          f"{data['MOTP']:>6.3f}    "
                          f"{data['Precision']:>8.2f}%   "
                          f"{data['Recall']:>6.2f}%   "
                          f"{data['ID_Sw']:>6}    "
                          f"{data['Frag']:>6}   "
                          f"{data['FP']:>6}   "
                          f"{data['FN']:>8}")
                
                # Calculate average
                if len(summary_data) > 0:
                    avg_data = {
                        'MOTA': np.mean([d['MOTA'] for d in summary_data]),
                        'IDF1': np.mean([d['IDF1'] for d in summary_data]),
                        'MOTP': np.mean([d['MOTP'] for d in summary_data]),
                        'Precision': np.mean([d['Precision'] for d in summary_data]),
                        'Recall': np.mean([d['Recall'] for d in summary_data]),
                        'ID_Sw': sum([d['ID_Sw'] for d in summary_data]),
                        'Frag': sum([d['Frag'] for d in summary_data]),
                        'FP': sum([d['FP'] for d in summary_data]),
                        'FN': sum([d['FN'] for d in summary_data]),
                    }
                    
                    print("-" * 100)
                    print(f"{'AVERAGE':<15} "
                          f"{avg_data['MOTA']:>6.2f}%   "
                          f"{avg_data['IDF1']:>6.2f}%   "
                          f"{avg_data['MOTP']:>6.3f}    "
                          f"{avg_data['Precision']:>9.2f}%   "
                          f"{avg_data['Recall']:>6.2f}%   "
                          f"{avg_data['ID_Sw']:>6}    "
                          f"{avg_data['Frag']:>6}   "
                          f"{avg_data['FP']:>6}   "
                          f"{avg_data['FN']:>8}")
                
                print("="*100)
                
                # Save detailed summary to file
                summary_file = Path(args.output_dir) / 'evaluation_summary.txt'
                with open(summary_file, 'w') as f:
                    f.write("="*100 + "\n")
                    f.write(f"{'MOT17 EVALUATION RESULTS - ByteTrack 2-Phase Matching':^100}\n")
                    f.write("="*100 + "\n\n")
                    
                    f.write("Configuration:\n")
                    f.write(f"  det_conf_high:     {args.det_conf_high}\n")
                    f.write(f"  det_conf_low:      {args.det_conf_low}\n")
                    f.write(f"  match_thresh_high: {args.match_thresh_high}\n")
                    f.write(f"  match_thresh_low:  {args.match_thresh_low}\n")
                    f.write(f"  track_buffer:      {args.track_buffer}\n\n")
                    
                    f.write("-"*100 + "\n")
                    f.write(f"{'Detector':<10} {'MOTA':>8} {'IDF1':>8} {'MOTP':>8} {'Precision':>10} {'Recall':>8} {'ID_Sw':>8} {'Frag':>8} {'FP':>8} {'FN':>8}\n")
                    f.write("-"*100 + "\n")
                    
                    for data in summary_data:
                        f.write(f"{data['Detector']:<10} "
                              f"{data['MOTA']:>7.2f}% "
                              f"{data['IDF1']:>7.2f}% "
                              f"{data['MOTP']:>8.3f} "
                              f"{data['Precision']:>9.2f}% "
                              f"{data['Recall']:>7.2f}% "
                              f"{data['ID_Sw']:>8d} "
                              f"{data['Frag']:>8d} "
                              f"{data['FP']:>8d} "
                              f"{data['FN']:>8d}\n")
                    
                    f.write("-"*100 + "\n")
                    f.write(f"{'AVERAGE':<10} "
                          f"{avg_data['MOTA']:>7.2f}% "
                          f"{avg_data['IDF1']:>7.2f}% "
                          f"{avg_data['MOTP']:>8.3f} "
                          f"{avg_data['Precision']:>9.2f}% "
                          f"{avg_data['Recall']:>7.2f}% "
                          f"{avg_data['ID_Sw']:>8d} "
                          f"{avg_data['Frag']:>8d} "
                          f"{avg_data['FP']:>8d} "
                          f"{avg_data['FN']:>8d}\n")
                    f.write("="*100 + "\n\n")
                    
                    # Write detailed results for each detector
                    for det_type, summary in all_results.items():
                        f.write(f"\n{det_type} Detailed Results:\n")
                        f.write("-"*100 + "\n")
                        f.write(summary.to_string())
                        f.write("\n\n")
                
                print(f"\nDetailed summary saved to: {summary_file}")
            
        except Exception as e:
            print(f"Warning: Could not compute metrics: {e}")
            import traceback
            traceback.print_exc()
            print("Make sure py-motmetrics is installed: pip install motmetrics")


if __name__ == '__main__':
    # Install required packages if not available
    print("Checking required packages...")
    try:
        import scipy.linalg
        import lap
        print("All packages available!")
    except ImportError as e:
        print(f"\nMissing package: {e}")
        print("Installing required packages...")
        import subprocess
        subprocess.check_call(['pip', 'install', 
                             'scipy', 'lap', 'motmetrics', 'opencv-python'])
        print("Packages installed successfully! Please run the script again.")
        exit(0)
    
    main()