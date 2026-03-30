"""
Traffic violation detection module
Detects traffic violations based on lane rules and vehicle behavior
"""

import numpy as np
import cv2
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional, Dict


class ViolationType(Enum):
    """Types of traffic violations"""
    NONE = 0
    WRONG_LANE = 1          # Xe vào lane không hợp lệ
    WRONG_DIRECTION = 2     # Đi sai hướng
    WRONG_VEHICLE_TYPE = 3  # Loại xe không hợp lệ cho lane này


@dataclass
class TrafficLane:
    """Represents a traffic lane with allowed vehicle types and direction"""
    lane_id: int
    name: str
    polygon: List[Tuple[int, int]]  # ROI polygon points
    allowed_classes: List[int]       # Class IDs allowed (e.g., [2,3,5] for cars, buses)
    direction_vector: Tuple[float, float] = (1.0, 0.0)  # Expected movement direction (normalized)
    
    def point_in_lane(self, x: int, y: int) -> bool:
        """Check if point is inside lane polygon"""
        if len(self.polygon) < 3:
            return False
        point = np.array([x, y], dtype=np.float32)
        polygon = np.array(self.polygon, dtype=np.int32)
        return cv2.pointPolygonTest(polygon, tuple(point), False) >= 0
    
    def bbox_in_lane(self, tlbr: Tuple[int, int, int, int]) -> bool:
        """Check if bbox center is inside lane"""
        x1, y1, x2, y2 = tlbr
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        return self.point_in_lane(cx, cy)


@dataclass
class TrackViolation:
    """Track violation information"""
    track_id: int
    violation_type: ViolationType
    lane_id: Optional[int] = None
    confidence: float = 1.0
    frame_detected: int = 0


class ViolationDetector:
    """Detect traffic violations"""
    
    def __init__(self):
        self.lanes: Dict[int, TrafficLane] = {}
        self.track_violations: Dict[int, List[TrackViolation]] = {}
        self.previous_positions: Dict[int, Tuple[int, int]] = {}
        self.persistent_violations: Dict[int, ViolationType] = {}  # Track violations that persist even after leaving lane
        
    def add_lane(self, lane: TrafficLane):
        """Register a new traffic lane"""
        self.lanes[lane.lane_id] = lane
    
    def detect_violations(self, tracks: List, frame_id: int) -> Dict[int, ViolationType]:
        """
        Detect violations for all tracks.
        
        Violations are PERSISTENT: Once a track is marked with a violation inside a lane,
        it remains flagged even if the vehicle exits the lane area.
        
        Returns:
            Dictionary of {track_id: violation_type}
        """
        violations = {}
        active_track_ids = set()
        
        for track in tracks:
            if not track.is_activated:
                continue
            
            track_id = track.track_id
            active_track_ids.add(track_id)
            tlbr = track.tlbr
            class_id = track.class_id
            x1, y1, x2, y2 = map(int, tlbr)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Start with existing persistent violation (if any)
            violation = self.persistent_violations.get(track_id, ViolationType.NONE)
            
            # Check for new violations only when vehicle is INSIDE a lane
            for lane_id, lane in self.lanes.items():
                if lane.bbox_in_lane(tlbr):
                    # Vehicle is INSIDE this lane - check for violations
                    new_violation = ViolationType.NONE
                    
                    # Check violation: Wrong vehicle type for this lane
                    if class_id is not None and class_id not in lane.allowed_classes:
                        new_violation = ViolationType.WRONG_VEHICLE_TYPE
                    
                    # Update violation if a new one is detected
                    if new_violation != ViolationType.NONE:
                        violation = new_violation
                        # Store in persistent violations
                        self.persistent_violations[track_id] = violation
                    
                    break
            
            # Report violation (either existing persistent or newly detected)
            if violation != ViolationType.NONE:
                violations[track_id] = violation
                track.violation_type = violation  # Update track's internal violation state
        
        # Clean up: Remove old violations for tracks that are no longer active
        removed_tracks = set(self.persistent_violations.keys()) - active_track_ids
        for track_id in removed_tracks:
            del self.persistent_violations[track_id]
            if track_id in self.previous_positions:
                del self.previous_positions[track_id]
        
        return violations
