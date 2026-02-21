#!/usr/bin/env python3
"""
ByteTrack GUI for Video Tracking with YOLO11s Traffic Model
GUI-only application for vehicle tracking using custom trained YOLO11s model

Usage:
    python bytetrack_test.py
"""

import os
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional
import scipy.linalg
import lap
import time
import sys


# ============================================================================
# ByteTrack Core Classes
# ============================================================================

class STrack:
    """Single target track with Kalman filter state"""
    
    shared_kalman = None
    track_id_count = 0
    
    def __init__(self, tlwh, score, class_id=None):
        # tlwh format: top-left width height
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        
        self.score = score
        self.tracklet_len = 0
        self.class_id = class_id  # Store class ID with track
        
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
        # ByteTrack: activate on frame 1, otherwise need confirmation
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
        # ByteTrack: reset tracklet_len to 0
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        # Keep the original class ID, don't change it
        # self.class_id remains unchanged
        
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
        # Keep the original class ID, don't change it
        # Only set class_id if it's not already set (for first update)
        if self.class_id is None:
            self.class_id = new_track.class_id
    
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
        # Format: [x1, y1, x2, y2, conf, class]
        if output_results.shape[1] == 5:
            # Format: [x1, y1, x2, y2, conf]
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
            classes = np.zeros(len(scores), dtype=np.int32)  # No class info
        else:
            # Format: [x1, y1, x2, y2, conf, class]
            # Use confidence score directly
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
            classes = output_results[:, 5].astype(np.int32)
        
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
        classes_high = classes[remain_inds_high]
        
        # D_low: Low-score detections (det_conf_low ≈ 0.1 < score ≤ det_conf_high)
        # These help recover tracks during occlusion/motion blur
        remain_inds_low = np.logical_and(scores > self.det_conf_low, scores <= self.det_conf_high)
        dets_low = bboxes[remain_inds_low]
        scores_low = scores[remain_inds_low]
        classes_low = classes[remain_inds_low]
        
        # Convert to STrack objects
        if len(dets_high) > 0:
            detections_high = [STrack(tlbr_to_tlwh(tlbr), s, c) for tlbr, s, c in zip(dets_high, scores_high, classes_high)]
        else:
            detections_high = []
            
        if len(dets_low) > 0:
            detections_low = [STrack(tlbr_to_tlwh(tlbr), s, c) for tlbr, s, c in zip(dets_low, scores_low, classes_low)]
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
        # First association with BOTH tracked AND lost tracks!
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
# ROI (Region of Interest) Utilities
# ============================================================================

class ROISelector:
    """Interactive ROI polygon selector"""
    
    def __init__(self):
        self.points = []
        self.drawing = False
        self.window_name = "Select ROI - Click to add points, 'c' to clear, 'q' to finish"
        
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for point selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            
    def select_roi(self, frame):
        """
        Interactive ROI selection
        
        Args:
            frame: First frame of video
            
        Returns:
            roi_polygon: List of (x, y) points defining ROI polygon
        """
        self.points = []
        clone = frame.copy()
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("\n" + "="*60)
        print("ROI Selection:")
        print("  - Click to add polygon points")
        print("  - Press 'c' to clear all points")
        print("  - Press 'q' to finish and continue")
        print("="*60 + "\n")
        
        while True:
            display = clone.copy()
            
            # Draw points
            for i, pt in enumerate(self.points):
                cv2.circle(display, pt, 5, (0, 255, 0), -1)
                if i > 0:
                    cv2.line(display, self.points[i-1], pt, (0, 255, 0), 2)
            
            # Draw closing line if we have points
            if len(self.points) > 2:
                cv2.line(display, self.points[-1], self.points[0], (0, 255, 0), 2)
                # Fill polygon with transparency
                overlay = display.copy()
                cv2.fillPoly(overlay, [np.array(self.points)], (0, 255, 0))
                cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)
            
            # Show instructions
            cv2.putText(display, f"Points: {len(self.points)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display, "Press 'c' to clear, 'q' to finish", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow(self.window_name, display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.points = []
                
        cv2.destroyWindow(self.window_name)
        
        if len(self.points) < 3:
            print(" No ROI selected (need at least 3 points). Using full frame.")
            return None
        
        print(f" ROI selected with {len(self.points)} points")
        return self.points


def create_roi_mask(frame_shape, roi_polygon):
    """
    Create binary mask from ROI polygon
    
    Args:
        frame_shape: (height, width) of frame
        roi_polygon: List of (x, y) points
        
    Returns:
        mask: Binary mask (255 inside ROI, 0 outside)
    """
    if roi_polygon is None or len(roi_polygon) < 3:
        return None
    
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(roi_polygon, dtype=np.int32)], 255)
    return mask


def filter_detections_by_roi(detections, roi_mask):
    """
    Filter detections to keep only those inside ROI
    
    Args:
        detections: Array [N, 6] of [x1, y1, x2, y2, conf, class]
        roi_mask: Binary mask
        
    Returns:
        filtered_detections: Detections inside ROI
    """
    if roi_mask is None or len(detections) == 0:
        return detections
    
    filtered = []
    for det in detections:
        x1, y1, x2, y2 = det[:4]
        # Check center point of bbox
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        # Keep detection if center is inside ROI
        if roi_mask[center_y, center_x] > 0:
            filtered.append(det)
    
    return np.array(filtered) if len(filtered) > 0 else np.empty((0, 6))


# ============================================================================
# YOLO Detector Integration
# ============================================================================

class YOLODetector:
    """YOLO detector wrapper for ByteTrack"""
    
    def __init__(self, model_path='yolo11n.pt', conf_threshold=0.1, device='cuda',
                 min_box_area=400, edge_margin=10):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model weights (.pt file)
            conf_threshold: Minimum confidence threshold for detections
            device: Device to run inference on ('cuda' or 'cpu')
            min_box_area: Minimum bounding box area (pixels) to keep detection (default: 400)
            edge_margin: Margin from frame edge (pixels) to filter out objects leaving scene (default: 10)
        """
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.conf_threshold = conf_threshold
            self.device = device
            self.min_box_area = min_box_area
            self.edge_margin = edge_margin
            
            # Load class names from model
            self.class_names = self.model.names  # Dict: {0: 'car', 1: 'truck', ...}
            
            print(f" YOLO model loaded: {model_path}")
            print(f"  Device: {device}")
            print(f"  Confidence threshold: {conf_threshold}")
            print(f"  Min box area: {min_box_area} pixels")
            print(f"  Edge margin: {edge_margin} pixels")
            print(f"  Classes: {list(self.class_names.values())}")
        except ImportError:
            print("Error: ultralytics package not found!")
            print("Install with: pip install ultralytics")
            raise
    
    def get_class_name(self, class_id):
        """Get class name from class ID"""
        return self.class_names.get(int(class_id), f"class_{int(class_id)}")
    
    def filter_detections(self, detections, frame_width, frame_height):
        """
        Filter detections by size and edge position
        
        Args:
            detections: numpy array [N, 6] with format [x1, y1, x2, y2, conf, class]
            frame_width: Width of the frame
            frame_height: Height of the frame
            
        Returns:
            filtered_detections: Detections passing the filters
        """
        if len(detections) == 0:
            return detections
        
        filtered = []
        for det in detections:
            x1, y1, x2, y2 = det[:4]
            
            # Calculate box area
            box_width = x2 - x1
            box_height = y2 - y1
            box_area = box_width * box_height
            
            # Filter 1: Remove very small boxes (likely false positives or too far away)
            if box_area < self.min_box_area:
                continue
            
            # Filter 2: Remove objects too close to frame edges (leaving scene)
            # Check if box center is near edge
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Allow some margin but filter out objects clearly leaving the frame
            if (center_x < self.edge_margin or 
                center_x > frame_width - self.edge_margin or
                center_y < self.edge_margin or 
                center_y > frame_height - self.edge_margin):
                # Additional check: only filter if box is also partially outside
                if (x1 < self.edge_margin or x2 > frame_width - self.edge_margin or
                    y1 < self.edge_margin or y2 > frame_height - self.edge_margin):
                    continue
            
            filtered.append(det)
        
        return np.array(filtered) if len(filtered) > 0 else np.empty((0, 6))
    
    def detect(self, frame):
        """
        Run detection on a frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            detections: numpy array of shape [N, 6] with format [x1, y1, x2, y2, conf, class]
        """
        frame_height, frame_width = frame.shape[:2]
        
        # Run inference
        results = self.model(frame, conf=self.conf_threshold, device=self.device, verbose=False)
        
        # Extract detections - USE LIST TO ACCUMULATE
        all_detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                # Get boxes in xyxy format
                xyxy = boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                conf = boxes.conf.cpu().numpy()  # confidence
                cls = boxes.cls.cpu().numpy()    # class
                
                # Use ALL detections from traffic model (vehicles, bikes, etc.)
                # No filtering by class - accept all vehicle types
                
                # Combine into detections array
                if len(xyxy) > 0:
                    batch_detections = np.concatenate([
                        xyxy,
                        conf.reshape(-1, 1),
                        cls.reshape(-1, 1)
                    ], axis=1)
                    all_detections.append(batch_detections)
        
        # Combine all detections from all results
        if len(all_detections) == 0:
            return np.empty((0, 6))
        
        # Concatenate all batches
        detections = np.vstack(all_detections) if len(all_detections) > 1 else all_detections[0]
        
        # Apply size and edge filters
        detections = self.filter_detections(detections, frame_width, frame_height)
        
        return detections


# ============================================================================
# Video Processor
# ============================================================================

class VideoProcessor:
    """Process video with ByteTrack + YOLO"""
    
    def __init__(self, video_path, tracker, detector, output_path=None, display=True):
        """
        Initialize video processor
        
        Args:
            video_path: Path to input video
            tracker: BYTETracker instance
            detector: YOLODetector instance
            output_path: Path to save output video (optional)
            display: Whether to display video while processing
        """
        self.video_path = video_path
        self.tracker = tracker
        self.detector = detector
        self.output_path = output_path
        self.display = display
        
        # Open video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n{'='*60}")
        print(f"Video Information:")
        print(f"  Path: {video_path}")
        print(f"  Resolution: {self.width}x{self.height}")
        print(f"  FPS: {self.fps}")
        print(f"  Total frames: {self.total_frames}")
        print(f"{'='*60}\n")
        
        # Setup video writer if output path is provided
        self.writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(output_path, fourcc, self.fps, 
                                         (self.width, self.height))
            print(f"Output will be saved to: {output_path}\n")
    
    def draw_tracks(self, frame, tracks, detections):
        """
        Draw tracking results on frame with class labels
        
        Args:
            frame: Input frame
            tracks: List of STrack objects
            detections: Detection array [x1, y1, x2, y2, conf, class]
            
        Returns:
            frame with drawings
        """
        # Generate colors for each track ID
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(1000, 3), dtype=np.uint8)
        
        # Draw tracks
        for track in tracks:
            if not track.is_activated:
                continue
            
            # Get bounding box
            tlbr = track.tlbr
            x1, y1, x2, y2 = map(int, tlbr)
            
            # Get track ID and color
            track_id = track.track_id
            color = colors[track_id % len(colors)].tolist()
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Get class name from track's stored class_id
            class_name = "unknown"
            if track.class_id is not None:
                class_name = self.detector.get_class_name(track.class_id)
            
            # Draw label with ID and class
            label = f"ID:{track_id} | {class_name}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Draw background for text
            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)
            
            # Draw text
            cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2)
        
        return frame
        
        return frame
    
    def process(self):
        """Process video with tracking"""
        frame_id = 0
        fps_list = []
        
        print("Processing video...")
        print(f"{'Frame':<10} {'Detections':<12} {'Tracks':<10} {'FPS':<10}")
        print("-" * 50)
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame_id += 1
                start_time = time.time()
                
                # Run detection
                detections = self.detector.detect(frame)
                
                # Run tracking
                img_shape = (self.height, self.width)
                online_tracks = self.tracker.update(detections, img_shape)
                
                # Calculate FPS
                elapsed = time.time() - start_time
                fps = 1.0 / elapsed if elapsed > 0 else 0
                fps_list.append(fps)
                
                # Print progress
                if frame_id % 30 == 0 or frame_id == 1:
                    print(f"{frame_id:<10} {len(detections):<12} {len(online_tracks):<10} {fps:<10.2f}")
                
                # Draw tracks on frame
                if self.display or self.writer:
                    frame_vis = self.draw_tracks(frame.copy(), online_tracks, detections)
                    
                    # Add info text
                    info_text = [
                        f"Frame: {frame_id}/{self.total_frames}",
                        f"Detections: {len(detections)}",
                        f"Tracks: {len(online_tracks)}",
                        f"FPS: {fps:.1f}"
                    ]
                    
                    y_offset = 30
                    for text in info_text:
                        cv2.putText(frame_vis, text, (10, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        y_offset += 30
                    
                    # Display
                    if self.display:
                        cv2.imshow('ByteTrack + YOLO', frame_vis)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("\nProcessing interrupted by user")
                            break
                    
                    # Save to video
                    if self.writer:
                        self.writer.write(frame_vis)
        
        finally:
            # Cleanup
            self.cap.release()
            if self.writer:
                self.writer.release()
            if self.display:
                cv2.destroyAllWindows()
            
            # Print summary
            print("\n" + "="*60)
            print("Processing completed!")
            print(f"  Total frames processed: {frame_id}")
            print(f"  Average FPS: {np.mean(fps_list):.2f}")
            print(f"  Total tracks created: {STrack.track_id_count}")
            if self.output_path:
                print(f"  Output saved to: {self.output_path}")
            print("="*60)


# ============================================================================
# GUI Components
# ============================================================================

def create_gui():
    """Create and run GUI application"""
    try:
        import tkinter as tk
        from tkinter import ttk, filedialog, scrolledtext, messagebox
        import threading
        import queue
        from PIL import Image, ImageTk
    except ImportError as e:
        print(f"Error: GUI requires additional packages: {e}")
        print("Install with: pip install pillow")
        sys.exit(1)
    
    class ByteTrackGUI:
        """GUI Application for ByteTrack + YOLO video tracking"""
        
        def __init__(self, root):
            self.root = root
            self.root.title("ByteTrack + YOLO Video Tracking")
            self.root.geometry("1400x900")
            
            # State variables
            self.video_path = None
            self.output_path = None
            self.is_processing = False
            self.should_stop = False
            self.cap = None
            self.tracker = None
            self.detector = None
            
            # Queue for thread communication
            self.log_queue = queue.Queue()
            self.frame_queue = queue.Queue(maxsize=2)
            
            # Create GUI
            self.create_widgets()
            
            # Start log update loop
            self.update_logs()
            
        def create_widgets(self):
            """Create all GUI widgets"""
            
            # Left Panel: Configuration with Scrollbar
            left_container = ttk.Frame(self.root)
            left_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            left_container.rowconfigure(0, weight=1)
            left_container.columnconfigure(0, weight=1)
            
            # Create canvas and scrollbar
            canvas = tk.Canvas(left_container, width=350, highlightthickness=0, bd=0)
            scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas, padding="5")
            
            # Bind scrollable_frame expand to canvas width
            canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            
            def on_canvas_configure(event):
                canvas.itemconfig(canvas_window, width=event.width)
            canvas.bind("<Configure>", on_canvas_configure)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.configure(yscrollcommand=scrollbar.set)
            
            # Grid layout (no spacing)
            canvas.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
            scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
            
            # Enable mouse wheel scrolling
            def _on_mousewheel(event):
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
            
            # Use scrollable_frame as left_frame
            left_frame = scrollable_frame
            
            # Title
            title_label = ttk.Label(left_frame, text="ByteTrack Configuration", 
                                   font=("Arial", 14, "bold"))
            title_label.grid(row=0, column=0, columnspan=2, pady=(0, 5))
            
            # Video Selection
            ttk.Label(left_frame, text="Video File:", font=("Arial", 10, "bold")).grid(
                row=1, column=0, sticky=tk.W, pady=2)
            
            video_frame = ttk.Frame(left_frame)
            video_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
            
            self.video_label = ttk.Label(video_frame, text="No video selected", 
                                         foreground="gray", wraplength=300)
            self.video_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            ttk.Button(video_frame, text="Browse...", command=self.browse_video).pack(
                side=tk.RIGHT, padx=(5, 0))
            
            # Model Path (fixed)
            ttk.Label(left_frame, text="YOLO Model:", font=("Arial", 10, "bold")).grid(
                row=3, column=0, sticky=tk.W, pady=2)
            
            model_frame = ttk.Frame(left_frame)
            model_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
            
            self.model_path = r"D:\Learn\Year4\KLTN\Dataset\traffic_yolo_v3\runs_part2\detect\traffic_yolo\yolo11s_traffic\weights\best.pt"
            model_label = ttk.Label(model_frame, text="YOLO11s Traffic (best.pt)", 
                                   foreground="blue", font=("Arial", 9))
            model_label.pack(side=tk.LEFT)
            
            # Device Selection
            ttk.Label(left_frame, text="Device:", font=("Arial", 10, "bold")).grid(
                row=5, column=0, sticky=tk.W, pady=2)
            
            self.device_var = tk.StringVar(value="cuda")
            device_frame = ttk.Frame(left_frame)
            device_frame.grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
            
            ttk.Radiobutton(device_frame, text="GPU (CUDA)", variable=self.device_var, 
                           value="cuda").pack(side=tk.LEFT, padx=(0, 10))
            ttk.Radiobutton(device_frame, text="CPU", variable=self.device_var, 
                           value="cpu").pack(side=tk.LEFT)
            
            # Separator
            ttk.Separator(left_frame, orient=tk.HORIZONTAL).grid(
                row=7, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
            
            # Detection Parameters
            ttk.Label(left_frame, text="Detection Parameters", 
                     font=("Arial", 11, "bold")).grid(row=8, column=0, columnspan=2, 
                                                      sticky=tk.W, pady=(2, 5))
            
            # YOLO Confidence
            ttk.Label(left_frame, text="YOLO Confidence:").grid(
                row=9, column=0, sticky=tk.W, pady=2)
            self.det_conf_var = tk.DoubleVar(value=0.01)
            det_conf_scale = ttk.Scale(left_frame, from_=0.0, to=1.0, 
                                       variable=self.det_conf_var, orient=tk.HORIZONTAL)
            det_conf_scale.grid(row=9, column=1, sticky=(tk.W, tk.E), pady=2)
            self.det_conf_label = ttk.Label(left_frame, text="0.01")
            self.det_conf_label.grid(row=10, column=1, sticky=tk.W)
            det_conf_scale.config(command=lambda v: self.det_conf_label.config(
                text=f"{float(v):.2f}"))
            
            # Separator
            ttk.Separator(left_frame, orient=tk.HORIZONTAL).grid(
                row=11, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
            
            # ByteTrack Parameters
            ttk.Label(left_frame, text="ByteTrack Parameters", 
                     font=("Arial", 11, "bold")).grid(row=12, column=0, columnspan=2, 
                                                      sticky=tk.W, pady=(2, 5))
            
            # High Confidence Threshold
            ttk.Label(left_frame, text="High Confidence:").grid(
                row=13, column=0, sticky=tk.W, pady=2)
            self.det_conf_high_var = tk.DoubleVar(value=0.5)
            high_scale = ttk.Scale(left_frame, from_=0.0, to=1.0, 
                                  variable=self.det_conf_high_var, orient=tk.HORIZONTAL)
            high_scale.grid(row=13, column=1, sticky=(tk.W, tk.E), pady=2)
            self.high_label = ttk.Label(left_frame, text="0.50")
            self.high_label.grid(row=14, column=1, sticky=tk.W)
            high_scale.config(command=lambda v: self.high_label.config(text=f"{float(v):.2f}"))
            
            # Low Confidence Threshold
            ttk.Label(left_frame, text="Low Confidence:").grid(
                row=15, column=0, sticky=tk.W, pady=2)
            self.det_conf_low_var = tk.DoubleVar(value=0.1)
            low_scale = ttk.Scale(left_frame, from_=0.0, to=1.0, 
                                 variable=self.det_conf_low_var, orient=tk.HORIZONTAL)
            low_scale.grid(row=15, column=1, sticky=(tk.W, tk.E), pady=2)
            self.low_label = ttk.Label(left_frame, text="0.10")
            self.low_label.grid(row=16, column=1, sticky=tk.W)
            low_scale.config(command=lambda v: self.low_label.config(text=f"{float(v):.2f}"))
            
            # New Track Threshold
            ttk.Label(left_frame, text="New Track Threshold:").grid(
                row=17, column=0, sticky=tk.W, pady=2)
            self.new_track_var = tk.DoubleVar(value=0.6)
            new_track_scale = ttk.Scale(left_frame, from_=0.0, to=1.0, 
                                        variable=self.new_track_var, orient=tk.HORIZONTAL)
            new_track_scale.grid(row=17, column=1, sticky=(tk.W, tk.E), pady=2)
            self.new_track_label = ttk.Label(left_frame, text="0.60")
            self.new_track_label.grid(row=18, column=1, sticky=tk.W)
            new_track_scale.config(command=lambda v: self.new_track_label.config(
                text=f"{float(v):.2f}"))
            
            # Match Threshold High
            ttk.Label(left_frame, text="Match Threshold (High):").grid(
                row=19, column=0, sticky=tk.W, pady=2)
            self.match_high_var = tk.DoubleVar(value=0.8)
            match_high_scale = ttk.Scale(left_frame, from_=0.0, to=1.0, 
                                         variable=self.match_high_var, orient=tk.HORIZONTAL)
            match_high_scale.grid(row=19, column=1, sticky=(tk.W, tk.E), pady=2)
            self.match_high_label = ttk.Label(left_frame, text="0.80 (IoU > 0.2)")
            self.match_high_label.grid(row=20, column=1, sticky=tk.W)
            match_high_scale.config(command=lambda v: self.match_high_label.config(
                text=f"{float(v):.2f} (IoU > {1-float(v):.1f})"))
            
            # Match Threshold Low
            ttk.Label(left_frame, text="Match Threshold (Low):").grid(
                row=21, column=0, sticky=tk.W, pady=2)
            self.match_low_var = tk.DoubleVar(value=0.5)
            match_low_scale = ttk.Scale(left_frame, from_=0.0, to=1.0, 
                                        variable=self.match_low_var, orient=tk.HORIZONTAL)
            match_low_scale.grid(row=21, column=1, sticky=(tk.W, tk.E), pady=2)
            self.match_low_label = ttk.Label(left_frame, text="0.50 (IoU > 0.5)")
            self.match_low_label.grid(row=22, column=1, sticky=tk.W)
            match_low_scale.config(command=lambda v: self.match_low_label.config(
                text=f"{float(v):.2f} (IoU > {1-float(v):.1f})"))
            
            # Track Buffer
            ttk.Label(left_frame, text="Track Buffer (frames):").grid(
                row=23, column=0, sticky=tk.W, pady=2)
            self.track_buffer_var = tk.IntVar(value=30)
            buffer_spinbox = ttk.Spinbox(left_frame, from_=1, to=100, 
                                         textvariable=self.track_buffer_var, width=10)
            buffer_spinbox.grid(row=23, column=1, sticky=tk.W, pady=2)
            
            # Min Hits
            ttk.Label(left_frame, text="Min Hits:").grid(
                row=24, column=0, sticky=tk.W, pady=2)
            self.min_hits_var = tk.IntVar(value=1)
            hits_spinbox = ttk.Spinbox(left_frame, from_=1, to=10, 
                                       textvariable=self.min_hits_var, width=10)
            hits_spinbox.grid(row=24, column=1, sticky=tk.W, pady=2)
            
            # Separator
            ttk.Separator(left_frame, orient=tk.HORIZONTAL).grid(
                row=25, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
            
            # Detection Filters
            ttk.Label(left_frame, text="Detection Filters", 
                     font=("Arial", 11, "bold")).grid(row=26, column=0, columnspan=2, 
                                                      sticky=tk.W, pady=(2, 5))
            
            # Min Box Area
            ttk.Label(left_frame, text="Min Box Area (pixels):").grid(
                row=27, column=0, sticky=tk.W, pady=2)
            self.min_box_area_var = tk.IntVar(value=400)
            area_spinbox = ttk.Spinbox(left_frame, from_=100, to=5000, increment=100,
                                       textvariable=self.min_box_area_var, width=10)
            area_spinbox.grid(row=27, column=1, sticky=tk.W, pady=2)
            ttk.Label(left_frame, text="(Filter tiny/distant boxes)", 
                     font=("Arial", 8), foreground="gray").grid(
                row=28, column=1, sticky=tk.W)
            
            # Edge Margin
            ttk.Label(left_frame, text="Edge Margin (pixels):").grid(
                row=29, column=0, sticky=tk.W, pady=2)
            self.edge_margin_var = tk.IntVar(value=10)
            margin_spinbox = ttk.Spinbox(left_frame, from_=0, to=100, increment=5,
                                         textvariable=self.edge_margin_var, width=10)
            margin_spinbox.grid(row=29, column=1, sticky=tk.W, pady=2)
            ttk.Label(left_frame, text="(Filter objects leaving frame)", 
                     font=("Arial", 8), foreground="gray").grid(
                row=30, column=1, sticky=tk.W)
            
            # Separator
            ttk.Separator(left_frame, orient=tk.HORIZONTAL).grid(
                row=31, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
            
            # Output Settings
            ttk.Label(left_frame, text="Output Settings", 
                     font=("Arial", 11, "bold")).grid(row=32, column=0, columnspan=2, 
                                                      sticky=tk.W, pady=(2, 5))
            
            self.save_output_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(left_frame, text="Save output video", 
                           variable=self.save_output_var,
                           command=self.toggle_output).grid(row=33, column=0, 
                                                            columnspan=2, sticky=tk.W, pady=2)
            
            # Control Buttons
            button_frame = ttk.Frame(left_frame)
            button_frame.grid(row=34, column=0, columnspan=2, pady=10)
            
            self.start_button = ttk.Button(button_frame, text=" Start Processing", 
                                           command=self.start_processing,
                                           width=20)
            self.start_button.pack(side=tk.LEFT, padx=5)
            
            self.stop_button = ttk.Button(button_frame, text=" Stop", 
                                          command=self.stop_processing, 
                                          state=tk.DISABLED, width=15)
            self.stop_button.pack(side=tk.LEFT, padx=5)
            
            # Configure grid weights for left frame
            left_frame.columnconfigure(1, weight=1)
            
            # Configure grid weights for main window
            self.root.columnconfigure(1, weight=1)
            self.root.rowconfigure(0, weight=1)
            left_container.rowconfigure(0, weight=1)
            
            # Right Panel: Video Display and Logs
            right_frame = ttk.Frame(self.root, padding="10")
            right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
            
            # Video Display
            video_label = ttk.Label(right_frame, text=" Video Preview", 
                                   font=("Arial", 12, "bold"))
            video_label.pack(pady=(0, 5))
            
            self.canvas = tk.Canvas(right_frame, width=800, height=450, bg="black")
            self.canvas.pack(pady=(0, 10))
            
            # Progress Bar
            self.progress_var = tk.DoubleVar()
            self.progress_bar = ttk.Progressbar(right_frame, variable=self.progress_var, 
                                               maximum=100, mode='determinate')
            self.progress_bar.pack(fill=tk.X, pady=(0, 5))
            
            # Status Label
            self.status_label = ttk.Label(right_frame, text="Ready to process", 
                                          font=("Arial", 10))
            self.status_label.pack(pady=(0, 10))
            
            # Logs
            log_label = ttk.Label(right_frame, text=" Processing Logs", 
                                 font=("Arial", 11, "bold"))
            log_label.pack(pady=(0, 5))
            
            self.log_text = scrolledtext.ScrolledText(right_frame, height=15, 
                                                      width=90, state=tk.DISABLED,
                                                      font=("Consolas", 9))
            self.log_text.pack(fill=tk.BOTH, expand=True)
            
        def browse_video(self):
            """Open file dialog to select video"""
            filename = filedialog.askopenfilename(
                title="Select Video File",
                filetypes=[
                    ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                    ("All files", "*.*")
                ]
            )
            
            if filename:
                self.video_path = filename
                self.video_label.config(text=Path(filename).name, foreground="black")
                self.log(f"✓ Video selected: {filename}")
                
                # Suggest output path
                if self.save_output_var.get():
                    output_name = Path(filename).stem + "_tracked.mp4"
                    self.output_path = str(Path(filename).parent / output_name)
                    
        def toggle_output(self):
            """Toggle output video saving"""
            if self.save_output_var.get() and self.video_path:
                output_name = Path(self.video_path).stem + "_tracked.mp4"
                self.output_path = str(Path(self.video_path).parent / output_name)
            else:
                self.output_path = None
                
        def log(self, message):
            """Add message to log"""
            self.log_queue.put(message)
            
        def update_logs(self):
            """Update log text widget from queue"""
            try:
                while True:
                    message = self.log_queue.get_nowait()
                    self.log_text.config(state=tk.NORMAL)
                    self.log_text.insert(tk.END, message + "\n")
                    self.log_text.see(tk.END)
                    self.log_text.config(state=tk.DISABLED)
            except queue.Empty:
                pass
            
            # Schedule next update
            self.root.after(100, self.update_logs)
            
        def start_processing(self):
            """Start video processing in separate thread"""
            if not self.video_path:
                messagebox.showerror("Error", "Please select a video file first!")
                return
            
            if not Path(self.video_path).exists():
                messagebox.showerror("Error", f"Video file not found: {self.video_path}")
                return
            
            # Check if model exists
            if not Path(self.model_path).exists():
                messagebox.showerror("Error", 
                    f"YOLO model not found!\n\n"
                    f"Expected path:\n{self.model_path}\n\n"
                    "Please make sure the trained YOLO11s model exists.")
                return
            
            # Disable controls
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.is_processing = True
            self.should_stop = False
            
            # Reset track counter
            STrack.track_id_count = 0
            
            # Start processing thread
            thread = threading.Thread(target=self.process_video, daemon=True)
            thread.start()
            
            # Start frame display update
            self.update_display()
            
        def stop_processing(self):
            """Stop video processing"""
            self.should_stop = True
            self.log("⚠ Stopping processing...")
            
        def process_video(self):
            """Process video with ByteTrack (runs in separate thread)"""
            try:
                # Initialize detector
                self.log("="*60)
                self.log("Initializing YOLO11s Traffic detector...")
                self.detector = YOLODetector(
                    model_path=self.model_path,
                    conf_threshold=self.det_conf_var.get(),
                    device=self.device_var.get(),
                    min_box_area=self.min_box_area_var.get(),
                    edge_margin=self.edge_margin_var.get()
                )
                self.log(f"✓ YOLO11s Traffic model loaded")
                self.log(f"  Model: {Path(self.model_path).name}")
                self.log(f"  Min box area: {self.min_box_area_var.get()} pixels")
                self.log(f"  Edge margin: {self.edge_margin_var.get()} pixels")
                
                # Initialize tracker
                self.log("Initializing ByteTrack...")
                self.tracker = BYTETracker(
                    det_conf_high=self.det_conf_high_var.get(),
                    det_conf_low=self.det_conf_low_var.get(),
                    new_track_thresh=self.new_track_var.get(),
                    match_thresh_high=self.match_high_var.get(),
                    match_thresh_low=self.match_low_var.get(),
                    track_buffer=self.track_buffer_var.get(),
                    min_hits=self.min_hits_var.get()
                )
                self.log("✓ ByteTrack initialized")
                
                # Open video
                self.log(f"Opening video: {Path(self.video_path).name}")
                self.cap = cv2.VideoCapture(self.video_path)
                
                if not self.cap.isOpened():
                    self.log("❌ ERROR: Cannot open video file!")
                    return
                
                # Get video properties
                fps = int(self.cap.get(cv2.CAP_PROP_FPS))
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                self.log(f"Video info: {width}x{height} @ {fps} FPS, {total_frames} frames")
                
                # Setup video writer if needed
                writer = None
                if self.save_output_var.get() and self.output_path:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
                    self.log(f"✓ Output will be saved to: {Path(self.output_path).name}")
                
                self.log("="*60)
                self.log("🚀 Processing started...")
                self.log(f"{'Frame':<10} {'Detections':<12} {'Tracks':<10} {'FPS':<10}")
                self.log("-"*50)
                
                # Process video
                frame_id = 0
                fps_list = []
                
                while not self.should_stop:
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    
                    frame_id += 1
                    start_time = time.time()
                    
                    # Run detection
                    detections = self.detector.detect(frame)
                    
                    # Run tracking
                    img_shape = (height, width)
                    online_tracks = self.tracker.update(detections, img_shape)
                    
                    # Calculate FPS
                    elapsed = time.time() - start_time
                    fps_val = 1.0 / elapsed if elapsed > 0 else 0
                    fps_list.append(fps_val)
                    
                    # Log progress
                    if frame_id % 30 == 0 or frame_id == 1:
                        self.log(f"{frame_id:<10} {len(detections):<12} "
                               f"{len(online_tracks):<10} {fps_val:<10.2f}")
                    
                    # Draw tracks
                    frame_vis = self.draw_tracks(frame, online_tracks, detections)
                    
                    # Add info text
                    info_text = [
                        f"Frame: {frame_id}/{total_frames}",
                        f"Detections: {len(detections)}",
                        f"Tracks: {len(online_tracks)}",
                        f"FPS: {fps_val:.1f}"
                    ]
                    
                    y_offset = 30
                    for text in info_text:
                        cv2.putText(frame_vis, text, (10, y_offset),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        y_offset += 30
                    
                    # Update progress
                    progress = (frame_id / total_frames) * 100
                    self.progress_var.set(progress)
                    self.status_label.config(
                        text=f"Processing: Frame {frame_id}/{total_frames} "
                             f"({progress:.1f}%) - FPS: {fps_val:.1f}")
                    
                    # Put frame in queue for display
                    if not self.frame_queue.full():
                        try:
                            self.frame_queue.put_nowait(frame_vis)
                        except queue.Full:
                            pass
                    
                    # Save frame
                    if writer:
                        writer.write(frame_vis)
                
                # Cleanup
                self.cap.release()
                if writer:
                    writer.release()
                
                # Summary
                self.log("\n" + "="*60)
                if self.should_stop:
                    self.log(" Processing stopped by user")
                else:
                    self.log(" Processing completed!")
                self.log(f"  Total frames processed: {frame_id}")
                if len(fps_list) > 0:
                    self.log(f"  Average FPS: {np.mean(fps_list):.2f}")
                self.log(f"  Total tracks created: {STrack.track_id_count}")
                if self.output_path and writer:
                    self.log(f"  Output saved to: {self.output_path}")
                self.log("="*60)
                
                self.status_label.config(text="✅ Processing completed!")
                self.progress_var.set(100)
                
                if not self.should_stop:
                    messagebox.showinfo("Success", 
                                       f"Video processing completed!\n\n"
                                       f"Frames processed: {frame_id}\n"
                                       f"Average FPS: {np.mean(fps_list):.1f}\n"
                                       f"Total tracks: {STrack.track_id_count}")
                
            except Exception as e:
                self.log(f" ERROR: {str(e)}")
                import traceback
                self.log(traceback.format_exc())
                messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
                
            finally:
                # Re-enable controls
                self.is_processing = False
                self.start_button.config(state=tk.NORMAL)
                self.stop_button.config(state=tk.DISABLED)
                
        def draw_tracks(self, frame, tracks, detections):
            """Draw tracking results on frame with class labels"""
            np.random.seed(42)
            colors = np.random.randint(0, 255, size=(1000, 3), dtype=np.uint8)
            
            # Draw tracks
            for track in tracks:
                if not track.is_activated:
                    continue
                
                tlbr = track.tlbr
                x1, y1, x2, y2 = map(int, tlbr)
                track_id = track.track_id
                color = colors[track_id % len(colors)].tolist()
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Get class name from track's stored class_id
                class_name = "unknown"
                if track.class_id is not None:
                    class_name = self.detector.get_class_name(track.class_id)
                
                # Draw label: "ID:1 | car"
                label = f"ID:{track_id} | {class_name}"
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)
                cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                           0.6, (255, 255, 255), 2)
            
            return frame
        
        def update_display(self):
            """Update canvas with latest frame"""
            if self.is_processing:
                try:
                    frame = self.frame_queue.get_nowait()
                    
                    # Resize frame to fit canvas
                    canvas_width = self.canvas.winfo_width()
                    canvas_height = self.canvas.winfo_height()
                    
                    if canvas_width > 1 and canvas_height > 1:
                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Resize
                        h, w = frame_rgb.shape[:2]
                        scale = min(canvas_width / w, canvas_height / h)
                        new_w, new_h = int(w * scale), int(h * scale)
                        frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
                        
                        # Convert to PhotoImage
                        img = Image.fromarray(frame_resized)
                        photo = ImageTk.PhotoImage(image=img)
                        
                        # Update canvas
                        self.canvas.delete("all")
                        self.canvas.create_image(canvas_width // 2, canvas_height // 2,
                                               image=photo, anchor=tk.CENTER)
                        self.canvas.image = photo  # Keep reference
                        
                except queue.Empty:
                    pass
                
                # Schedule next update
                self.root.after(30, self.update_display)
    
    # Create and run GUI
    root = tk.Tk()
    
    # Set style
    style = ttk.Style()
    try:
        style.theme_use('clam')
    except:
        pass  # Use default theme if clam not available
    
    app = ByteTrackGUI(root)
    
    print("=" * 60)
    print("ByteTrack GUI Started")
    print("=" * 60)
    
    root.mainloop()


# ============================================================================
# Remove old MOT17 related code
# ============================================================================


# ============================================================================
# Main Script - GUI Only
# ============================================================================

def main():
    """Launch GUI application"""
    print("=" * 60)
    print("ByteTrack + YOLO11s Traffic Tracking")
    print("GUI Mode")
    print("=" * 60)
    print("\nLaunching GUI...")
    create_gui()


if __name__ == '__main__':
    # Check required packages
    print("=" * 60)
    print("ByteTrack + YOLO11s Traffic - Vehicle Tracking")
    print("=" * 60)
    print("Checking required packages...")
    
    missing_packages = []
    
    try:
        import scipy.linalg
        import lap
        from ultralytics import YOLO
        print(" Core packages available (scipy, lap, ultralytics)")
    except ImportError as e:
        print(f" Missing core package: {e}")
        missing_packages.extend(["scipy", "lap", "ultralytics", "opencv-python"])
    
    # Check GUI packages (required)
    try:
        import tkinter
        from PIL import Image, ImageTk
        print(" GUI packages available (tkinter, pillow)")
    except ImportError as e:
        print(f" Missing GUI package: {e}")
        missing_packages.append("pillow")
    
    if missing_packages:
        print(f"\n Missing required packages!")
        print("\n Please install required packages:")
        print(f"  pip install {' '.join(set(missing_packages))}")
        sys.exit(1)
    
    # Check model exists
    model_path = Path(r"D:\Learn\Year4\KLTN\Dataset\traffic_yolo_v3\runs_part1\detect\traffic_yolo\yolo11s_traffic\weights\best.pt")
    if not model_path.exists():
        print(f"\n Warning: YOLO11s model not found!")
        print(f"Expected path: {model_path}")
        print("\nPlease make sure the trained model exists.")
        print("You can still launch the GUI, but processing will fail without the model.")
        input("\nPress Enter to continue anyway...")
    else:
        print(f" YOLO11s Traffic model found: {model_path.name}")
    
    print("\n" + "=" * 60)
    print()
    
    main()