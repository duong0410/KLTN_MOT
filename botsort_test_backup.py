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
            self.root.title("BoT-SORT Vehicle Tracking - YOLO11s Traffic Model")
            self.root.geometry("800x850")
            
            # Variables
            self.video_path = tk.StringVar()
            self.output_path = tk.StringVar()
            self.model_path = tk.StringVar(value=r"D:\Learn\Year4\KLTN\Dataset\traffic_yolo\runs\yolo11s_traffic2\weights\best.pt")
            
            # Tracker parameters
            self.det_conf_high = tk.DoubleVar(value=0.5)
            self.det_conf_low = tk.DoubleVar(value=0.1)
            self.new_track_thresh = tk.DoubleVar(value=0.6)
            self.match_thresh_high = tk.DoubleVar(value=0.8)
            self.match_thresh_low = tk.DoubleVar(value=0.5)
            self.track_buffer = tk.IntVar(value=30)
            self.min_hits = tk.IntVar(value=1)
            
            # Detector parameters
            self.det_conf_threshold = tk.DoubleVar(value=0.3)
            self.min_box_area = tk.IntVar(value=400)
            self.edge_margin = tk.IntVar(value=10)
            
            # CMC parameters
            self.use_cmc = tk.BooleanVar(value=True)
            self.cmc_method = tk.StringVar(value='orb')
            
            # Device
            self.device = tk.StringVar(value='cuda')
            
            # ROI
            self.use_roi = tk.BooleanVar(value=False)
            self.roi_polygon = None
            
            self.processing = False
            
            self.create_widgets()
        
        def create_widgets(self):
            """Create GUI widgets"""
            # Title
            title_frame = ttk.Frame(self.root)
            title_frame.pack(pady=10, fill='x')
            
            title_label = ttk.Label(title_frame, text="BoT-SORT Vehicle Tracking", 
                                   font=('Arial', 16, 'bold'))
            title_label.pack()
            
            subtitle_label = ttk.Label(title_frame, text="Improved Kalman Filter + Camera Motion Compensation",
                                      font=('Arial', 10))
            subtitle_label.pack()
            
            # Create main container with scrollbar
            main_canvas = tk.Canvas(self.root)
            scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=main_canvas.yview)
            scrollable_frame = ttk.Frame(main_canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
            )
            
            main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            main_canvas.configure(yscrollcommand=scrollbar.set)
            
            # Pack scrollbar and canvas
            scrollbar.pack(side="right", fill="y")
            main_canvas.pack(side="left", fill="both", expand=True, padx=10)
            
            # File Selection
            self.create_file_section(scrollable_frame)
            
            # BoT-SORT Parameters
            self.create_tracker_section(scrollable_frame)
            
            # YOLO Detector Parameters
            self.create_detector_section(scrollable_frame)
            
            # CMC Parameters
            self.create_cmc_section(scrollable_frame)
            
            # ROI
            self.create_roi_section(scrollable_frame)
            
            # Control buttons
            control_frame = ttk.Frame(self.root)
            control_frame.pack(pady=10, fill='x', padx=10)
            
            self.start_button = ttk.Button(control_frame, text="Start Tracking", 
                                          command=self.start_tracking, width=20)
            self.start_button.pack(side='left', padx=5)
            
            ttk.Button(control_frame, text="Exit", command=self.root.quit, width=20).pack(side='right', padx=5)
            
            # Status bar
            self.status_var = tk.StringVar(value="Ready")
            status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                                  relief='sunken', anchor='w')
            status_bar.pack(side='bottom', fill='x')
        
        def create_file_section(self, parent):
            """Create file selection section"""
            # Model path
            model_frame = ttk.LabelFrame(parent, text="YOLO Model", padding=10)
            model_frame.pack(pady=5, fill='x')
            
            ttk.Entry(model_frame, textvariable=self.model_path, width=70).pack(side='left', padx=5)
            ttk.Button(model_frame, text="Browse", command=self.browse_model).pack(side='left')
            
            # Video input
            video_frame = ttk.LabelFrame(parent, text="Input Video", padding=10)
            video_frame.pack(pady=5, fill='x')
            
            ttk.Entry(video_frame, textvariable=self.video_path, width=70).pack(side='left', padx=5)
            ttk.Button(video_frame, text="Browse", command=self.browse_video).pack(side='left')
            
            # Output video
            output_frame = ttk.LabelFrame(parent, text="Output Video (optional)", padding=10)
            output_frame.pack(pady=5, fill='x')
            
            ttk.Entry(output_frame, textvariable=self.output_path, width=70).pack(side='left', padx=5)
            ttk.Button(output_frame, text="Browse", command=self.browse_output).pack(side='left')
        
        def create_tracker_section(self, parent):
            """Create tracker parameters section"""
            tracker_frame = ttk.LabelFrame(parent, text="BoT-SORT Parameters", padding=10)
            tracker_frame.pack(pady=5, fill='x')
            
            self.create_slider(tracker_frame, "High Score Threshold:", self.det_conf_high, 0.0, 1.0, 0.01)
            self.create_slider(tracker_frame, "Low Score Threshold:", self.det_conf_low, 0.0, 1.0, 0.01)
            self.create_slider(tracker_frame, "New Track Threshold:", self.new_track_thresh, 0.0, 1.0, 0.01)
            self.create_slider(tracker_frame, "High Match Threshold:", self.match_thresh_high, 0.0, 1.0, 0.01)
            self.create_slider(tracker_frame, "Low Match Threshold:", self.match_thresh_low, 0.0, 1.0, 0.01)
            self.create_slider(tracker_frame, "Track Buffer (frames):", self.track_buffer, 1, 100, 1)
            self.create_slider(tracker_frame, "Minimum Hits:", self.min_hits, 1, 10, 1)
        
        def create_detector_section(self, parent):
            """Create detector parameters section"""
            det_frame = ttk.LabelFrame(parent, text="YOLO Detector Parameters", padding=10)
            det_frame.pack(pady=5, fill='x')
            
            self.create_slider(det_frame, "Confidence Threshold:", self.det_conf_threshold, 0.0, 1.0, 0.01)
            self.create_slider(det_frame, "Min Box Area (pixels):", self.min_box_area, 0, 2000, 50)
            self.create_slider(det_frame, "Edge Margin (pixels):", self.edge_margin, 0, 100, 5)
            
            # Device selection
            device_row = ttk.Frame(det_frame)
            device_row.pack(fill='x', pady=5)
            ttk.Label(device_row, text="Device:", width=25).pack(side='left')
            ttk.Radiobutton(device_row, text="CUDA (GPU)", variable=self.device, 
                           value='cuda').pack(side='left', padx=10)
            ttk.Radiobutton(device_row, text="CPU", variable=self.device, 
                           value='cpu').pack(side='left')
        
        def create_cmc_section(self, parent):
            """Create CMC parameters section"""
            cmc_frame = ttk.LabelFrame(parent, text="Camera Motion Compensation (CMC) - BoT-SORT Feature", padding=10)
            cmc_frame.pack(pady=5, fill='x')
            
            ttk.Checkbutton(cmc_frame, text="Enable CMC (handle camera pan/tilt/zoom)", 
                           variable=self.use_cmc).pack(anchor='w', pady=5)
            
            method_row = ttk.Frame(cmc_frame)
            method_row.pack(fill='x', pady=5)
            
            ttk.Label(method_row, text="CMC Method:", width=25).pack(side='left')
            ttk.Radiobutton(method_row, text="ORB (fast)", variable=self.cmc_method, 
                           value='orb').pack(side='left', padx=10)
            ttk.Radiobutton(method_row, text="SIFT (accurate)", variable=self.cmc_method, 
                           value='sift').pack(side='left')
            
            # Info label
            info_label = ttk.Label(cmc_frame, text="ℹ CMC compensates for camera movement using feature matching", 
                                  foreground='blue', font=('Arial', 9, 'italic'))
            info_label.pack(anchor='w', pady=5)
        
        def create_roi_section(self, parent):
            """Create ROI section"""
            roi_frame = ttk.LabelFrame(parent, text="Region of Interest (ROI)", padding=10)
            roi_frame.pack(pady=5, fill='x')
            
            ttk.Checkbutton(roi_frame, text="Enable ROI (select on first frame)", 
                           variable=self.use_roi).pack(anchor='w')
        
        def create_slider(self, parent, label, variable, from_, to, resolution):
            """Create a labeled slider"""
            frame = ttk.Frame(parent)
            frame.pack(fill='x', pady=3)
            
            label_widget = ttk.Label(frame, text=label, width=25)
            label_widget.pack(side='left')
            
            value_label = ttk.Label(frame, text=f"{variable.get():.2f}", width=8)
            value_label.pack(side='right')
            
            def update_label(val):
                if isinstance(variable.get(), int):
                    value_label.config(text=f"{int(float(val))}")
                else:
                    value_label.config(text=f"{float(val):.2f}")
            
            slider = ttk.Scale(frame, from_=from_, to=to, variable=variable, 
                              orient='horizontal', command=update_label)
            slider.pack(side='left', fill='x', expand=True, padx=5)
        
        def browse_model(self):
            """Browse for model file"""
            path = filedialog.askopenfilename(
                title="Select YOLO Model",
                filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")]
            )
            if path:
                self.model_path.set(path)
        
        def browse_video(self):
            """Browse for input video"""
            path = filedialog.askopenfilename(
                title="Select Input Video",
                filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
            )
            if path:
                self.video_path.set(path)
        
        def browse_output(self):
            """Browse for output video"""
            path = filedialog.asksaveasfilename(
                title="Save Output Video",
                defaultextension=".mp4",
                filetypes=[("MP4 Video", "*.mp4"), ("AVI Video", "*.avi"), ("All Files", "*.*")]
            )
            if path:
                self.output_path.set(path)
        
        def start_tracking(self):
            """Start tracking process"""
            # Validate inputs
            if not self.video_path.get():
                messagebox.showerror("Error", "Please select an input video")
                return
            
            if not Path(self.model_path.get()).exists():
                messagebox.showerror("Error", "Model file not found")
                return
            
            if not Path(self.video_path.get()).exists():
                messagebox.showerror("Error", "Video file not found")
                return
            
            # Run in separate thread
            self.processing = True
            self.start_button.config(state='disabled')
            self.stop_button.config(state='normal')
            
            thread = threading.Thread(target=self.run_tracking)
            thread.daemon = True
            thread.start()
        

        
        def run_tracking(self):
            """Run tracking (in separate thread)"""
            try:
                self.status_var.set("Initializing...")
                
                # Initialize detector
                detector = YOLODetector(
                    model_path=self.model_path.get(),
                    conf_threshold=self.det_conf_threshold.get(),
                    device=self.device.get(),
                    min_box_area=self.min_box_area.get(),
                    edge_margin=self.edge_margin.get()
                )
                
                # Initialize tracker
                cmc_method = self.cmc_method.get() if self.use_cmc.get() else 'none'
                tracker = BoTSORT(
                    det_conf_high=self.det_conf_high.get(),
                    det_conf_low=self.det_conf_low.get(),
                    new_track_thresh=self.new_track_thresh.get(),
                    match_thresh_high=self.match_thresh_high.get(),
                    match_thresh_low=self.match_thresh_low.get(),
                    track_buffer=self.track_buffer.get(),
                    min_hits=self.min_hits.get(),
                    use_cmc=self.use_cmc.get(),
                    cmc_method=cmc_method
                )
                
                # ROI selection
                roi_polygon = None
                if self.use_roi.get():
                    cap = cv2.VideoCapture(self.video_path.get())
                    ret, first_frame = cap.read()
                    cap.release()
                    
                    if ret:
                        roi_selector = ROISelector()
                        roi_polygon = roi_selector.select_roi(first_frame)
                        self.roi_polygon = roi_polygon
                
                # Initialize processor
                output_path = self.output_path.get() if self.output_path.get() else None
                processor = VideoProcessor(
                    video_path=self.video_path.get(),
                    tracker=tracker,
                    detector=detector,
                    output_path=output_path,
                    display=True,
                    roi_polygon=roi_polygon
                )
                
                self.status_var.set("Processing video...")
                processor.process()
                
                self.status_var.set("Complete!")
                messagebox.showinfo("Success", "Tracking completed successfully!")
                
            except Exception as e:
                self.status_var.set(f"Error: {str(e)}")
                messagebox.showerror("Error", f"Tracking failed:\n{str(e)}")
            
            finally:
                self.processing = False
                self.start_button.config(state='normal')
                self.stop_button.config(state='disabled')
    
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
