#!/usr/bin/env python3
"""
BoT-SORT GUI for Video Tracking with YOLO11s Traffic Model
GUI application for vehicle tracking using custom trained YOLO11s model with BoT-SORT improvements

BoT-SORT = ByteTrack + Improved Kalman Filter + Camera Motion Compensation
- Improved Kalman Filter: Better covariance tuning for more stable predictions
- Camera Motion Compensation (GMC): Handle camera pan/tilt/zoom using ORB/SIFT features

Usage:
    python botsort_test.py
"""

import os
import sys
import numpy as np
import cv2
import time
import threading
import queue
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional
import scipy.linalg
import lap


# ============================================================================
# BoT-SORT Core Classes
# ============================================================================

class STrack:
    """Single target track with Kalman filter state (BoT-SORT enhanced)"""
    
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
        
        # BoT-SORT: Track prediction for lost tracks
        self.predicted_bbox = None
        
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
        
        # BoT-SORT: Store predicted bbox
        self.predicted_bbox = self.tlwh.copy()
        
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
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        
    def re_activate(self, new_track, frame_id, new_id=False):
        """Reactivate a lost track"""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        
    def update(self, new_track, frame_id, min_hits=1):
        """Update a matched track"""
        self.frame_id = frame_id
        self.tracklet_len += 1
        
        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh)
        )
        
        if self.tracklet_len >= min_hits:
            self.is_activated = True
            self.state = TrackState.Tracked
        
        self.score = new_track.score
        
        # Keep original class ID
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
    
    def camera_update(self, warp):
        """
        BoT-SORT: Apply camera motion compensation
        Compensate track position for camera movement
        """
        if warp is None or self.mean is None:
            return
        
        # Get center point of track
        x, y = self.mean[0], self.mean[1]
        
        # Apply affine transformation
        x_new = warp[0, 0] * x + warp[0, 1] * y + warp[0, 2]
        y_new = warp[1, 0] * x + warp[1, 1] * y + warp[1, 2]
        
        # Update mean position
        self.mean[0] = x_new
        self.mean[1] = y_new
    
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
    """
    Kalman filter for track state estimation (BoT-SORT improved version)
    
    Key improvements over ByteTrack:
    1. Better initial covariance (more confident in detections)
    2. Reduced process noise (smoother predictions)
    3. Better observation noise (trust measurements more)
    """
    
    def __init__(self):
        ndim, dt = 4, 1.
        
        # Create Kalman filter model matrices
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        
        # BoT-SORT: Improved uncertainty weights (more confident)
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160
        
    def initiate(self, measurement):
        """
        Create track from unassociated measurement
        BoT-SORT: Reduced initial covariance for more stable initialization
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        
        # BoT-SORT: Lower initial uncertainty (more confident in detection)
        std = [
            0.5 * self._std_weight_position * measurement[3],
            0.5 * self._std_weight_position * measurement[3],
            1e-4,
            0.5 * self._std_weight_position * measurement[3],
            2 * self._std_weight_velocity * measurement[3],
            2 * self._std_weight_velocity * measurement[3],
            1e-6,
            2 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance
    
    def predict(self, mean, covariance):
        """
        Run Kalman filter prediction step
        BoT-SORT: Reduced process noise for smoother predictions
        """
        # BoT-SORT: Lower process noise (more stable predictions)
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-4,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-6,
            self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        
        return mean, covariance
    
    def project(self, mean, covariance):
        """
        Project state distribution to measurement space
        BoT-SORT: Lower measurement noise (trust detections more)
        """
        # BoT-SORT: Lower measurement uncertainty
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-3,
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
        """
        Run prediction step for multiple tracks
        BoT-SORT: Same improvements as single predict
        """
        # BoT-SORT: Lower process noise
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-4 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-6 * np.ones_like(mean[:, 3]),
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


class GMC:
    """
    Generalized Motion Compensation (GMC) for BoT-SORT
    Handles camera motion (pan, tilt, zoom) to improve tracking stability
    """
    
    def __init__(self, method='orb', downscale=2):
        """
        Args:
            method: Feature detection method ('orb', 'sift', 'ecc')
            downscale: Downscale factor for faster processing
        """
        self.method = method
        self.downscale = max(1, downscale)
        self.prevFrame = None
        self.prevKeyPoints = None
        self.prevDescriptors = None
        self.initializedFirstFrame = False
        
        # Create feature detector
        if method == 'orb':
            self.detector = cv2.ORB_create(1000)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        elif method == 'sift':
            self.detector = cv2.SIFT_create(1000)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        else:
            self.method = 'none'
    
    def apply(self, raw_frame, detections=None):
        """
        Estimate camera motion between frames
        
        Args:
            raw_frame: Current frame (BGR image)
            detections: Optional detections to mask
            
        Returns:
            H: 2x3 affine transformation matrix (or None if estimation fails)
        """
        if self.method == 'none':
            return np.eye(2, 3, dtype=np.float32)
        
        # Downscale frame
        height, width = raw_frame.shape[:2]
        frame = cv2.resize(raw_frame, (width // self.downscale, height // self.downscale))
        
        # Convert to grayscale
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        H = np.eye(2, 3, dtype=np.float32)
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.detector.detectAndCompute(frame, None)
        
        # First frame initialization
        if not self.initializedFirstFrame:
            self.prevFrame = frame.copy()
            self.prevKeyPoints = keypoints
            self.prevDescriptors = descriptors
            self.initializedFirstFrame = True
            return H
        
        # Match features with previous frame
        if descriptors is not None and self.prevDescriptors is not None:
            matches = self.matcher.knnMatch(self.prevDescriptors, descriptors, k=2)
            
            # Filter matches using ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            
            # Need at least 4 matches for affine transform
            if len(good_matches) >= 4:
                # Get matched keypoints
                src_pts = np.float32([self.prevKeyPoints[m.queryIdx].pt for m in good_matches])
                dst_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches])
                
                # Estimate affine transformation
                H, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
                
                # Scale back to original resolution
                if H is not None:
                    H[:, 2] *= self.downscale
                else:
                    H = np.eye(2, 3, dtype=np.float32)
        
        # Update previous frame
        self.prevFrame = frame.copy()
        self.prevKeyPoints = keypoints
        self.prevDescriptors = descriptors
        
        return H


class BoTSORT:
    """
    BoT-SORT multi-object tracker with improvements over ByteTrack:
    
    1. Improved Kalman Filter (lower covariance, more stable predictions)
    2. Camera Motion Compensation (GMC) - handle camera pan/tilt/zoom
    3. Track Prediction - predict bbox for short-term lost tracks
    """
    
    def __init__(self, det_conf_high=0.5, det_conf_low=0.1, new_track_thresh=0.6,
                 match_thresh_high=0.8, match_thresh_low=0.5, track_buffer=30, min_hits=1,
                 use_cmc=True, cmc_method='orb'):
        """
        Args:
            det_conf_high: Confidence threshold for high-score detections
            det_conf_low: Confidence threshold for low-score detections
            new_track_thresh: Threshold for creating new tracks
            match_thresh_high: Cost threshold for first association
            match_thresh_low: Cost threshold for second association
            track_buffer: Number of frames to keep lost tracks
            min_hits: Minimum hits before track is confirmed
            use_cmc: Enable Camera Motion Compensation
            cmc_method: CMC method ('orb', 'sift', 'none')
        """
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        
        self.frame_id = 0
        
        # Thresholds
        self.det_conf_high = det_conf_high
        self.det_conf_low = det_conf_low
        self.new_track_thresh = new_track_thresh
        self.match_thresh_high = match_thresh_high
        self.match_thresh_low = match_thresh_low
        self.max_time_lost = track_buffer
        self.min_hits = min_hits
        
        # BoT-SORT: Camera Motion Compensation
        self.use_cmc = use_cmc
        if use_cmc and cmc_method != 'none':
            self.gmc = GMC(method=cmc_method, downscale=2)
        else:
            self.gmc = None
        
        self.kalman_filter = KalmanFilter()
        STrack.shared_kalman = self.kalman_filter
        
    def update(self, output_results, img_shape, img=None):
        """
        Update tracker with new detections using BoT-SORT algorithm
        
        Args:
            output_results: numpy array [N, 6] of [x1, y1, x2, y2, conf, class]
            img_shape: tuple of (height, width)
            img: Optional raw frame for CMC (BGR image)
            
        Returns:
            list of active tracks
        """
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        
        # Parse detection results
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
            classes = np.zeros(len(scores), dtype=int)
        else:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
            classes = output_results[:, 5].astype(int)
        
        # Split detections by score threshold
        remain_inds_high = scores > self.det_conf_high
        dets_high = bboxes[remain_inds_high]
        scores_high = scores[remain_inds_high]
        classes_high = classes[remain_inds_high]
        
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
            
        # Separate confirmed and unconfirmed tracks
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        
        # BoT-SORT: Camera Motion Compensation (GMC)
        if self.gmc is not None and img is not None:
            warp = self.gmc.apply(img, detections_high)
            if warp is not None:
                # Apply camera motion compensation to all tracks
                for track in tracked_stracks:
                    track.camera_update(warp)
                for track in self.lost_stracks:
                    track.camera_update(warp)
        
        # First association with BOTH tracked AND lost tracks
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        
        # Predict ALL tracks with Kalman Filter
        STrack.multi_predict(strack_pool)
        
        # First Association: Match with high-score detections
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
        
        # Get remaining detections and TRACKED tracks
        dets_remain_high = [detections_high[i] for i in u_detection_high]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        
        # Second Association: Match remaining tracked tracks with low-score detections
        dists_low = iou_distance(r_tracked_stracks, detections_low)
        matches_low, u_track_remain, u_detection_low = linear_assignment(dists_low, thresh=self.match_thresh_low)
        
        # Update tracks matched with low-score detections
        for itracked, idet in matches_low:
            track = r_tracked_stracks[itracked]
            det = detections_low[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id, self.min_hits)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        
        # Mark unmatched TRACKED tracks as lost
        for it in u_track_remain:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        
        # Handle unconfirmed tracks
        detections = [detections_high[i] for i in u_detection_high]
        dists = iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.5)
        
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id, self.min_hits)
            activated_stracks.append(unconfirmed[itracked])
        
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)
        
        # Initialize new tracks
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)
            
        # Remove lost tracks that exceeded max_time_lost
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
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        
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
    """Compute cost based on IoU between tracks"""
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
    """Perform linear assignment using Hungarian algorithm"""
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
        """Interactive ROI selection"""
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
                cv2.putText(display, str(i+1), (pt[0]+10, pt[1]+10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw polygon
            if len(self.points) > 1:
                cv2.polylines(display, [np.array(self.points)], True, (0, 255, 0), 2)
                
            # Draw filled polygon preview
            if len(self.points) >= 3:
                overlay = display.copy()
                cv2.fillPoly(overlay, [np.array(self.points)], (0, 255, 0))
                cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)
            
            cv2.imshow(self.window_name, display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.points = []
                
        cv2.destroyWindow(self.window_name)
        
        if len(self.points) < 3:
            print("⚠ Warning: Need at least 3 points for ROI. ROI disabled.")
            return None
        
        print(f"✓ ROI selected with {len(self.points)} points")
        return self.points


def create_roi_mask(frame_shape, roi_polygon):
    """Create binary mask from ROI polygon"""
    if roi_polygon is None or len(roi_polygon) < 3:
        return None
    
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(roi_polygon, dtype=np.int32)], 255)
    return mask


def filter_detections_by_roi(detections, roi_mask):
    """Filter detections to keep only those inside ROI"""
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
    """YOLO detector wrapper for BoT-SORT"""
    
    def __init__(self, model_path='yolo11n.pt', conf_threshold=0.1, device='cuda',
                 min_box_area=400, edge_margin=10):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model weights
            conf_threshold: Minimum confidence threshold
            device: Device to run inference on
            min_box_area: Minimum bounding box area (pixels)
            edge_margin: Margin from frame edge (pixels)
        """
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.conf_threshold = conf_threshold
            self.device = device
            self.min_box_area = min_box_area
            self.edge_margin = edge_margin
            
            # Get class names from model
            self.class_names = self.model.names if hasattr(self.model, 'names') else {}
            
            print(f"✓ YOLO model loaded: {model_path}")
            print(f"  Classes: {list(self.class_names.values())}")
            print(f"  Device: {device}")
            print(f"  Confidence threshold: {conf_threshold}")
            
        except ImportError:
            print("ERROR: ultralytics package not found!")
            print("Install with: pip install ultralytics")
            sys.exit(1)
    
    def get_class_name(self, class_id):
        """Get class name from class ID"""
        return self.class_names.get(int(class_id), f"class_{int(class_id)}")
    
    def filter_detections(self, detections, frame_width, frame_height):
        """
        Filter detections by area and edge proximity
        
        Args:
            detections: Array [N, 6] of [x1, y1, x2, y2, conf, class]
            frame_width: Frame width
            frame_height: Frame height
            
        Returns:
            filtered_detections: Array of filtered detections
        """
        if len(detections) == 0:
            return detections
        
        filtered = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            
            # Calculate box area
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            # Filter by minimum area
            if area < self.min_box_area:
                continue
            
            # Filter objects too close to edge (likely leaving scene)
            if (x1 < self.edge_margin or 
                y1 < self.edge_margin or 
                x2 > frame_width - self.edge_margin or 
                y2 > frame_height - self.edge_margin):
                continue
            
            filtered.append(det)
        
        return np.array(filtered) if len(filtered) > 0 else np.empty((0, 6))
    
    def detect(self, frame):
        """
        Run detection on frame
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            detections: Array [N, 6] of [x1, y1, x2, y2, conf, class]
        """
        results = self.model(frame, conf=self.conf_threshold, device=self.device, verbose=False)
        
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                detections.append([x1, y1, x2, y2, conf, cls])
        
        detections = np.array(detections) if len(detections) > 0 else np.empty((0, 6))
        
        # Filter detections
        detections = self.filter_detections(detections, frame.shape[1], frame.shape[0])
        
        return detections


# ============================================================================
# Video Processor
# ============================================================================

class VideoProcessor:
    """Process video with BoT-SORT + YOLO"""
    
    def __init__(self, video_path, tracker, detector, output_path=None, display=True, roi_polygon=None):
        """
        Initialize video processor
        
        Args:
            video_path: Path to input video
            tracker: BoTSORT instance
            detector: YOLODetector instance
            output_path: Path to output video (optional)
            display: Whether to display video during processing
            roi_polygon: ROI polygon points (optional)
        """
        self.video_path = video_path
        self.tracker = tracker
        self.detector = detector
        self.output_path = output_path
        self.display = display
        self.roi_polygon = roi_polygon
        
        # Open video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create ROI mask if provided
        self.roi_mask = create_roi_mask((self.height, self.width), roi_polygon)
        
        # Setup video writer if output path provided
        self.writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
        # Statistics
        self.frame_count = 0
        self.track_count = {}  # Track ID -> frame count
        
        print(f"\n{'='*60}")
        print(f"Video Properties:")
        print(f"  Resolution: {self.width}x{self.height}")
        print(f"  FPS: {self.fps}")
        print(f"  Total frames: {self.total_frames}")
        print(f"  ROI: {'Enabled' if self.roi_mask is not None else 'Disabled'}")
        print(f"{'='*60}\n")
    
    def draw_tracks(self, frame, tracks, detections):
        """Draw tracks and detections on frame"""
        # Draw ROI polygon if enabled
        if self.roi_polygon is not None:
            overlay = frame.copy()
            cv2.polylines(overlay, [np.array(self.roi_polygon)], True, (0, 255, 0), 2)
            cv2.fillPoly(overlay, [np.array(self.roi_polygon)], (0, 255, 0))
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        
        # Color palette for different classes
        colors = [
            (255, 0, 0),    # Blue
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        
        # Draw tracks
        for track in tracks:
            x1, y1, x2, y2 = track.tlbr.astype(int)
            track_id = track.track_id
            class_id = track.class_id if track.class_id is not None else 0
            
            # Get color based on class
            color = colors[class_id % len(colors)]
            
            # Draw bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            class_name = self.detector.get_class_name(class_id)
            label = f"ID:{track_id} {class_name}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Draw label background
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Update track statistics
            if track_id not in self.track_count:
                self.track_count[track_id] = 0
            self.track_count[track_id] += 1
        
        # Draw info
        info_text = f"Frame: {self.frame_count}/{self.total_frames} | Tracks: {len(tracks)} | Total IDs: {len(self.track_count)}"
        cv2.putText(frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return frame
    
    def process(self):
        """Process video"""
        print("Processing video...")
        print("Press 'q' to quit, 'p' to pause/resume")
        
        paused = False
        
        try:
            while True:
                if not paused:
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    
                    self.frame_count += 1
                    
                    # Detect objects
                    detections = self.detector.detect(frame)
                    
                    # Filter by ROI if enabled
                    if self.roi_mask is not None:
                        detections = filter_detections_by_roi(detections, self.roi_mask)
                    
                    # Update tracker (pass raw frame for CMC)
                    tracks = self.tracker.update(detections, (self.height, self.width), img=frame)
                    
                    # Draw tracks
                    vis_frame = self.draw_tracks(frame.copy(), tracks, detections)
                    
                    # Write to output
                    if self.writer is not None:
                        self.writer.write(vis_frame)
                    
                    # Display
                    if self.display:
                        cv2.imshow('BoT-SORT Tracking', vis_frame)
                    
                    # Progress
                    if self.frame_count % 30 == 0:
                        progress = (self.frame_count / self.total_frames) * 100
                        print(f"Progress: {progress:.1f}% ({self.frame_count}/{self.total_frames}) | Tracks: {len(tracks)}")
                
                # Handle keys
                if self.display:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\nStopped by user")
                        break
                    elif key == ord('p'):
                        paused = not paused
                        print(f"{'Paused' if paused else 'Resumed'}")
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.cap.release()
            if self.writer is not None:
                self.writer.release()
            cv2.destroyAllWindows()
            
            print(f"\n{'='*60}")
            print("Processing complete!")
            print(f"  Total frames processed: {self.frame_count}")
            print(f"  Total unique tracks: {len(self.track_count)}")
            if self.output_path:
                print(f"  Output saved to: {self.output_path}")
            print(f"{'='*60}\n")


# ============================================================================
# GUI Components
# ============================================================================

def create_gui():
    """Create and run GUI application"""
    try:
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
        import threading
    except ImportError as e:
        print(f"ERROR: GUI packages not found: {e}")
        print("Install with: pip install tk")
        sys.exit(1)
    
    class BoTSORTGUI:
        def __init__(self, root):
            self.root = root
            self.root.title("BoT-SORT + YOLO Video Tracking")
            self.root.geometry("1400x900")
            
            # State variables
            self.video_path = None
            self.output_path = None
            self.is_processing = False
            self.should_stop = False
            self.cap = None
            self.tracker = None
            self.detector = None
            
            # Model path (fixed)
            self.model_path = r"D:\Learn\Year4\KLTN\Dataset\traffic_yolo\runs\yolo11s_traffic2\weights\best.pt"
            
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
            title_label = ttk.Label(left_frame, text="BoT-SORT Configuration", 
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
            
            # BoT-SORT Parameters
            ttk.Label(left_frame, text="BoT-SORT Parameters", 
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
            
            # CMC Parameters (BoT-SORT specific)
            ttk.Label(left_frame, text="Camera Motion Compensation", 
                     font=("Arial", 11, "bold")).grid(row=26, column=0, columnspan=2, 
                                                      sticky=tk.W, pady=(2, 5))
            
            self.use_cmc_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(left_frame, text="Enable CMC", 
                           variable=self.use_cmc_var).grid(row=27, column=0, 
                                                           columnspan=2, sticky=tk.W, pady=2)
            
            ttk.Label(left_frame, text="CMC Method:").grid(
                row=28, column=0, sticky=tk.W, pady=2)
            
            self.cmc_method_var = tk.StringVar(value="orb")
            cmc_method_frame = ttk.Frame(left_frame)
            cmc_method_frame.grid(row=28, column=1, sticky=tk.W, pady=2)
            
            ttk.Radiobutton(cmc_method_frame, text="ORB", variable=self.cmc_method_var, 
                           value="orb").pack(side=tk.LEFT, padx=(0, 10))
            ttk.Radiobutton(cmc_method_frame, text="SIFT", variable=self.cmc_method_var, 
                           value="sift").pack(side=tk.LEFT)
            
            ttk.Label(left_frame, text="(Handles camera pan/tilt/zoom)", 
                     font=("Arial", 8), foreground="gray").grid(
                row=29, column=1, sticky=tk.W)
            
            # Separator
            ttk.Separator(left_frame, orient=tk.HORIZONTAL).grid(
                row=30, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
            
            # Detection Filters
            ttk.Label(left_frame, text="Detection Filters", 
                     font=("Arial", 11, "bold")).grid(row=31, column=0, columnspan=2, 
                                                      sticky=tk.W, pady=(2, 5))
            
            # Min Box Area
            ttk.Label(left_frame, text="Min Box Area (pixels):").grid(
                row=32, column=0, sticky=tk.W, pady=2)
            self.min_box_area_var = tk.IntVar(value=400)
            area_spinbox = ttk.Spinbox(left_frame, from_=100, to=5000, increment=100,
                                       textvariable=self.min_box_area_var, width=10)
            area_spinbox.grid(row=32, column=1, sticky=tk.W, pady=2)
            ttk.Label(left_frame, text="(Filter tiny/distant boxes)", 
                     font=("Arial", 8), foreground="gray").grid(
                row=33, column=1, sticky=tk.W)
            
            # Edge Margin
            ttk.Label(left_frame, text="Edge Margin (pixels):").grid(
                row=34, column=0, sticky=tk.W, pady=2)
            self.edge_margin_var = tk.IntVar(value=10)
            margin_spinbox = ttk.Spinbox(left_frame, from_=0, to=100, increment=5,
                                         textvariable=self.edge_margin_var, width=10)
            margin_spinbox.grid(row=34, column=1, sticky=tk.W, pady=2)
            ttk.Label(left_frame, text="(Filter objects leaving frame)", 
                     font=("Arial", 8), foreground="gray").grid(
                row=35, column=1, sticky=tk.W)
            
            # Separator
            ttk.Separator(left_frame, orient=tk.HORIZONTAL).grid(
                row=36, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
            
            # Output Settings
            ttk.Label(left_frame, text="Output Settings", 
                     font=("Arial", 11, "bold")).grid(row=37, column=0, columnspan=2, 
                                                      sticky=tk.W, pady=(2, 5))
            
            self.save_output_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(left_frame, text="Save output video", 
                           variable=self.save_output_var,
                           command=self.toggle_output).grid(row=38, column=0, 
                                                            columnspan=2, sticky=tk.W, pady=2)
            
            # Control Buttons
            button_frame = ttk.Frame(left_frame)
            button_frame.grid(row=39, column=0, columnspan=2, pady=10)
            
            self.start_button = ttk.Button(button_frame, text="▶ Start Processing", 
                                           command=self.start_processing,
                                           width=20)
            self.start_button.pack(side=tk.LEFT, padx=5)
            
            self.stop_button = ttk.Button(button_frame, text="■ Stop", 
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
            video_label = ttk.Label(right_frame, text="📹 Video Preview", 
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
            log_label = ttk.Label(right_frame, text="📄 Processing Logs", 
                                 font=("Arial", 11, "bold"))
            log_label.pack(pady=(0, 5))
            
            try:
                from tkinter import scrolledtext
                self.log_text = scrolledtext.ScrolledText(right_frame, height=15, 
                                                          width=90, state=tk.DISABLED,
                                                          font=("Consolas", 9))
                self.log_text.pack(fill=tk.BOTH, expand=True)
            except:
                # Fallback without scrolledtext
                self.log_text = tk.Text(right_frame, height=15, width=90, 
                                       state=tk.DISABLED, font=("Consolas", 9))
                self.log_text.pack(fill=tk.BOTH, expand=True)
        
        def browse_video(self):
            """Open file dialog to select video"""
            try:
                from tkinter import filedialog
            except:
                return
            
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
                    output_name = Path(filename).stem + "_botsort_tracked.mp4"
                    self.output_path = str(Path(filename).parent / output_name)
                    
        def toggle_output(self):
            """Toggle output video saving"""
            if self.save_output_var.get() and self.video_path:
                output_name = Path(self.video_path).stem + "_botsort_tracked.mp4"
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
            from tkinter import messagebox
            
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
            """Process video with BoT-SORT (runs in separate thread)"""
            try:
                from tkinter import messagebox
                
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
                self.log("Initializing BoT-SORT...")
                cmc_method = self.cmc_method_var.get() if self.use_cmc_var.get() else 'none'
                self.tracker = BoTSORT(
                    det_conf_high=self.det_conf_high_var.get(),
                    det_conf_low=self.det_conf_low_var.get(),
                    new_track_thresh=self.new_track_var.get(),
                    match_thresh_high=self.match_high_var.get(),
                    match_thresh_low=self.match_low_var.get(),
                    track_buffer=self.track_buffer_var.get(),
                    min_hits=self.min_hits_var.get(),
                    use_cmc=self.use_cmc_var.get(),
                    cmc_method=cmc_method
                )
                self.log("✓ BoT-SORT initialized")
                if self.use_cmc_var.get():
                    self.log(f"  CMC enabled: {cmc_method.upper()} feature matching")
                
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
                    
                    # Run tracking with BoT-SORT (passes raw frame for CMC)
                    img_shape = (height, width)
                    online_tracks = self.tracker.update(detections, img_shape, img=frame)
                    
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
                        f"FPS: {fps_val:.1f}",
                        f"CMC: {'ON' if self.use_cmc_var.get() else 'OFF'}"
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
                    self.log("⏹ Processing stopped by user")
                else:
                    self.log("✅ Processing completed!")
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
                self.log(f"❌ ERROR: {str(e)}")
                import traceback
                self.log(traceback.format_exc())
                try:
                    from tkinter import messagebox
                    messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
                except:
                    pass
                
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
                        try:
                            from PIL import Image, ImageTk
                            img = Image.fromarray(frame_resized)
                            photo = ImageTk.PhotoImage(image=img)
                            
                            # Update canvas
                            self.canvas.delete("all")
                            self.canvas.create_image(canvas_width // 2, canvas_height // 2,
                                                   image=photo, anchor=tk.CENTER)
                            self.canvas.image = photo  # Keep reference
                        except:
                            pass
                        
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
        pass
    
    app = BoTSORTGUI(root)
    
    print("=" * 60)
    print("BoT-SORT GUI Started")
    print("=" * 60)
    
    root.mainloop()


# ============================================================================
# Main Script - GUI Only
# ============================================================================

def main():
    """Launch GUI application"""
    print("=" * 60)
    print("BoT-SORT + YOLO11s Traffic Tracking")
    print("GUI Mode")
    print("=" * 60)
    print("\nLaunching GUI...")
    create_gui()


if __name__ == '__main__':
    # Check required packages
    print("=" * 60)
    print("BoT-SORT + YOLO11s Traffic - Vehicle Tracking")
    print("=" * 60)
    print("Checking required packages...")
    
    missing_packages = []
    
    try:
        import scipy.linalg
        import lap
        print("✓ Core packages available")
    except ImportError as e:
        missing_packages.append(str(e))
    
    # Check GUI packages
    try:
        import tkinter
        print("✓ GUI packages available")
    except ImportError as e:
        missing_packages.append(str(e))
    
    if missing_packages:
        print("\n⚠ Missing packages:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\nInstalling required packages...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'scipy', 'lap', 'tk'])
        print("\nPackages installed! Please run the script again.")
        sys.exit(0)
    
    # Check model exists
    model_path = Path(r"D:\Learn\Year4\KLTN\Dataset\traffic_yolo\runs\yolo11s_traffic2\weights\best.pt")
    if not model_path.exists():
        print(f"\n⚠ Warning: Default model not found at:")
        print(f"  {model_path}")
        print("  You can select a different model in the GUI.")
    else:
        print(f"✓ Default model found")
    
    print("\n" + "=" * 60)
    print()
    
    main()
