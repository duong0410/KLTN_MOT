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
    NONE = 0
    WRONG_LANE = 1
    WRONG_DIRECTION = 2
    WRONG_VEHICLE_TYPE = 3
    NO_PARKING = 4


@dataclass
class TrafficLane:
    lane_id: int
    name: str
    polygon: List[Tuple[int, int]]
    allowed_classes: List[int]
    direction_vector: Tuple[float, float] = (1.0, 0.0)

    def point_in_lane(self, x: int, y: int) -> bool:
        if len(self.polygon) < 3:
            return False
        point = np.array([x, y], dtype=np.float32)
        polygon = np.array(self.polygon, dtype=np.int32)
        return cv2.pointPolygonTest(polygon, tuple(point), False) >= 0

    def bbox_in_lane(self, tlbr: Tuple[int, int, int, int]) -> bool:
        x1, y1, x2, y2 = tlbr
        return self.point_in_lane((x1 + x2) // 2, (y1 + y2) // 2)


@dataclass
class NoParkingZone:
    zone_id: int
    name: str
    polygon: List[Tuple[int, int]]
    parking_frame_threshold: int = 60      # ~2 giây ở 30fps — số frame đứng yên liên tục để bị flag
    movement_threshold: float = 3.0        # < 3 pixel/frame → đứng yên
    movement_reset_threshold: float = 8.0  # > 8 pixel/frame → di chuyển rõ ràng, reset streak
    max_total_displacement: float = 15.0   # Tổng displacement tối đa so với entry point
    clear_frame_threshold: int = 15        # Số frame di chuyển liên tục để clear flag
    clear_movement_threshold: float = 25.0 # Pixel từ flag_center để clear flag ngay lập tức

    def point_in_zone(self, x: int, y: int) -> bool:
        if len(self.polygon) < 3:
            return False
        point = np.array([x, y], dtype=np.float32)
        polygon = np.array(self.polygon, dtype=np.int32)
        return cv2.pointPolygonTest(polygon, tuple(point), False) >= 0

    def bbox_in_zone(self, tlbr: Tuple[int, int, int, int]) -> bool:
        x1, y1, x2, y2 = tlbr
        return self.point_in_zone((x1 + x2) // 2, (y1 + y2) // 2)


@dataclass
class TrackViolation:
    track_id: int
    violation_type: ViolationType
    lane_id: Optional[int] = None
    confidence: float = 1.0
    frame_detected: int = 0


class ViolationDetector:

    def __init__(self):
        self.lanes: Dict[int, TrafficLane] = {}
        self.no_parking_zones: Dict[int, NoParkingZone] = {}
        self.track_violations: Dict[int, List[TrackViolation]] = {}
        self.previous_positions: Dict[int, Tuple[int, int]] = {}
        self.persistent_violations: Dict[int, ViolationType] = {}
        self.no_parking_state: Dict[Tuple[int, int], Dict] = {}

    def add_lane(self, lane: TrafficLane):
        self.lanes[lane.lane_id] = lane

    def add_no_parking_zone(self, zone: NoParkingZone):
        self.no_parking_zones[zone.zone_id] = zone

    def _get_stable_center(self, track, x1, y1, x2, y2):
        """Dùng Kalman position để tránh bbox jitter."""
        if hasattr(track, 'mean') and track.mean is not None:
            return int(track.mean[0]), int(track.mean[1])
        return (x1 + x2) // 2, (y1 + y2) // 2

    def detect_violations(self, tracks: List, frame_id: int) -> Dict[int, List]:
        """
        Detect violations for all tracks.
        
        Returns:
            Dictionary of {track_id: [List of ViolationType]}
            Mỗi track có thể bị cả WRONG_VEHICLE_TYPE lẫn NO_PARKING cùng lúc.
        """
        violations = {}
        active_track_ids = set()

        for track in tracks:
            if not track.is_activated:
                continue

            track_id = track.track_id
            active_track_ids.add(track_id)
            tlbr = track.tlbr
            class_id = getattr(track, 'class_id', None)
            x1, y1, x2, y2 = map(int, tlbr)
            cx, cy = self._get_stable_center(track, x1, y1, x2, y2)

            # ================================================================
            # Bước 1: Lấy violations persist (WRONG_VEHICLE_TYPE persist)
            # ================================================================
            violations_list = []
            
            if track_id in self.persistent_violations:
                persisted = self.persistent_violations[track_id]
                # Lấy WRONG_VEHICLE_TYPE từ persistent
                if persisted == ViolationType.WRONG_VEHICLE_TYPE:
                    violations_list.append(ViolationType.WRONG_VEHICLE_TYPE)

            # ================================================================
            # Bước 2: Check lane violations (WRONG_VEHICLE_TYPE)
            # ================================================================
            for lane_id, lane in self.lanes.items():
                if lane.bbox_in_lane(tlbr):
                    if class_id is not None and class_id not in lane.allowed_classes:
                        if ViolationType.WRONG_VEHICLE_TYPE not in violations_list:
                            violations_list.append(ViolationType.WRONG_VEHICLE_TYPE)
                            self.persistent_violations[track_id] = ViolationType.WRONG_VEHICLE_TYPE
                    break

            # ================================================================
            # Bước 3: Check no-parking violations
            # Vào ngay cả nếu đã bị WRONG_VEHICLE_TYPE (có thể bị cả 2)
            #
            # Logic phát hiện (dùng stationary_streak):
            # - Đếm số frame đứng yên LIÊN TỤC (stationary_streak).
            # - Bất kỳ frame nào xe nhúc nhích → streak về 0.
            # - Xe di chuyển nhanh → reset entry_center và streak.
            # - Trigger khi streak >= parking_frame_threshold.
            #
            # Logic clear violation:
            # - Khi xe đang bị flag, theo dõi movement liên tục.
            # - Clear khi di chuyển >= clear_frame_threshold frame
            #   HOẶC di chuyển xa hơn clear_movement_threshold pixel từ flag_center.
            # - Xe dừng lại → reset moving_frames_after_flag về 0.
            # ================================================================
            is_parking = False
            
            for zone_id, zone in self.no_parking_zones.items():
                    key = (track_id, zone_id)

                    if zone.bbox_in_zone(tlbr):
                        state = self.no_parking_state.get(key)

                        if state is None:
                            # Khởi tạo state — frame đầu chưa có previous position
                            # nên movement = 0, streak bắt đầu từ 1.
                            state = {
                                'entry_center': (cx, cy),
                                'last_center': (cx, cy),
                                'frames_in_zone': 1,
                                'stationary_streak': 1,  # frame đứng yên liên tiếp
                                'moving_frames_after_flag': 0,
                                'is_flagged': False,
                                'flag_center': None,
                            }
                            self.no_parking_state[key] = state

                        else:
                            state['frames_in_zone'] += 1
                            prev_cx, prev_cy = state['last_center']
                            movement = float(np.hypot(cx - prev_cx, cy - prev_cy))

                            if state['is_flagged']:
                                # -----------------------------------------------
                                # Xe đang bị flag → theo dõi để clear
                                # -----------------------------------------------
                                flag_cx, flag_cy = state['flag_center']
                                dist_from_flag = float(np.hypot(
                                    cx - flag_cx, cy - flag_cy
                                ))

                                if movement > zone.movement_threshold:
                                    state['moving_frames_after_flag'] += 1
                                else:
                                    # Xe dừng lại → phải di chuyển liên tục mới clear
                                    state['moving_frames_after_flag'] = 0

                                # Clear flag khi di chuyển đủ lâu hoặc đủ xa
                                if (state['moving_frames_after_flag'] >= zone.clear_frame_threshold
                                        or dist_from_flag > zone.clear_movement_threshold):
                                    state['is_flagged'] = False
                                    state['flag_center'] = None
                                    state['moving_frames_after_flag'] = 0
                                    state['entry_center'] = (cx, cy)
                                    state['stationary_streak'] = 0
                                    state['frames_in_zone'] = 0
                                    # is_parking vẫn False → violation được clear
                                else:
                                    # Chưa đủ điều kiện clear → vẫn flag
                                    is_parking = True
                                    violation = ViolationType.NO_PARKING

                            else:
                                # -----------------------------------------------
                                # Xe chưa bị flag → tích lũy stationary_streak
                                # -----------------------------------------------
                                entry_cx, entry_cy = state['entry_center']
                                total_displacement = float(np.hypot(
                                    cx - entry_cx, cy - entry_cy
                                ))

                                if movement > zone.movement_reset_threshold:
                                    # Di chuyển nhanh → reset hoàn toàn
                                    state['entry_center'] = (cx, cy)
                                    state['stationary_streak'] = 0
                                else:
                                    # Đứng yên khi movement nhỏ VÀ chưa trôi xa khỏi entry
                                    is_still = (
                                        movement <= zone.movement_threshold
                                        and total_displacement <= zone.max_total_displacement
                                    )

                                    if is_still:
                                        state['stationary_streak'] += 1
                                    else:
                                        # Nhúc nhích nhẹ hoặc trôi dần → reset streak
                                        # nhưng không reset entry_center
                                        state['stationary_streak'] = 0

                                # Trigger khi streak đủ dài
                                if state['stationary_streak'] >= zone.parking_frame_threshold:
                                    state['is_flagged'] = True
                                    state['flag_center'] = (cx, cy)
                                    state['moving_frames_after_flag'] = 0
                                    is_parking = True
                                    violation = ViolationType.NO_PARKING

                            state['last_center'] = (cx, cy)

                        if is_parking:
                            break

                    else:
                        # Xe rời zone → xóa state hoàn toàn
                        if key in self.no_parking_state:
                            del self.no_parking_state[key]

            # ================================================================
            # Bước 4: Cập nhật persistent violations
            # ================================================================
            if is_parking:
                self.persistent_violations[track_id] = ViolationType.NO_PARKING
                if ViolationType.NO_PARKING not in violations_list:
                    violations_list.append(ViolationType.NO_PARKING)
            elif (track_id in self.persistent_violations
                  and self.persistent_violations[track_id] == ViolationType.NO_PARKING):
                del self.persistent_violations[track_id]

            # ================================================================
            # Bước 5: Ghi nhận violations
            # ================================================================
            if violations_list:
                violations[track_id] = violations_list
                # Lưu vào track object (chỉ lưu type đầu tiên là primary, UI có thể dùng cả list)
                if track.violation_type is not None:
                    pass  # Sẽ set dưới
            else:
                if hasattr(track, 'violation_type'):
                    track.violation_type = ViolationType.NONE

        # ================================================================
        # Cleanup: xóa state của các track không còn active
        # ================================================================
        removed_tracks = set(self.persistent_violations.keys()) - active_track_ids
        for track_id in removed_tracks:
            del self.persistent_violations[track_id]
            self.previous_positions.pop(track_id, None)

        stale_keys = [k for k in self.no_parking_state if k[0] not in active_track_ids]
        for key in stale_keys:
            del self.no_parking_state[key]

        return violations