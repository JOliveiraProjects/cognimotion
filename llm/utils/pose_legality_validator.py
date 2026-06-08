from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from data.pose_frame import PoseFrame


@dataclass
class JointConstraint:
    bone_index: int
    min_angle_deg: float
    max_angle_deg: float
    axis: np.ndarray = None

    def __post_init__(self):
        if self.axis is None:
            self.axis = np.array([0., 1., 0.], dtype=np.float32)


@dataclass
class LegalityReport:
    is_legal: bool
    violations: List[str]
    corrected_transforms: Optional[List[np.ndarray]]
    confidence_penalty: float

    @staticmethod
    def legal() -> "LegalityReport":
        return LegalityReport(is_legal=True, violations=[], corrected_transforms=None, confidence_penalty=0.0)


class PoseLegalityValidator:
    def __init__(self, constraints: Optional[List[JointConstraint]] = None):
        self.constraints = constraints or self._default_constraints()
        self._max_velocity_cm_s = 800.0
        self._max_angular_vel_rad_s = 15.0
        self._foot_ground_threshold = 10.0

    def validate(self, frame: PoseFrame) -> LegalityReport:
        violations: List[str] = []
        penalty = 0.0

        velocity_norm = float(np.linalg.norm(frame.linear_velocity))
        if velocity_norm > self._max_velocity_cm_s:
            violations.append(
                f"Linear velocity {velocity_norm:.1f} exceeds max {self._max_velocity_cm_s:.0f} cm/s")
            penalty += 0.2

        angular_norm = float(np.linalg.norm(frame.angular_velocity))
        if angular_norm > self._max_angular_vel_rad_s:
            violations.append(
                f"Angular velocity {angular_norm:.2f} rad/s exceeds max {self._max_angular_vel_rad_s:.1f}")
            penalty += 0.1

        if not np.isfinite(frame.root_location).all():
            violations.append("Root location contains NaN/Inf")
            penalty += 1.0

        if not np.isfinite(frame.root_rotation).all():
            violations.append("Root rotation contains NaN/Inf")
            penalty += 1.0

        rot_norm = float(np.linalg.norm(frame.root_rotation))
        if abs(rot_norm - 1.0) > 0.05:
            violations.append(f"Root rotation quaternion not normalized: norm={rot_norm:.3f}")
            penalty += 0.1

        for i, bt in enumerate(frame.bone_transforms):
            if not np.isfinite(bt).all():
                violations.append(f"Bone {i} transform contains NaN/Inf")
                penalty += 0.3
                break

        for constraint in self.constraints:
            if constraint.bone_index >= len(frame.bone_transforms):
                continue
            bt = frame.bone_transforms[constraint.bone_index]
            bt_arr = np.asarray(bt, dtype=np.float32)
            if not np.isfinite(bt_arr).all():
                continue
            if bt_arr.shape == (4, 4):
                rot_col = bt_arr[:3, :3]
            elif bt_arr.ndim == 1 and bt_arr.shape[0] >= 9:
                rot_col = bt_arr[:9].reshape(3, 3)
            else:
                continue
            axis_n = constraint.axis / (np.linalg.norm(constraint.axis) + 1e-8)
            forward_n = rot_col @ axis_n
            cross_norm = float(np.linalg.norm(np.cross(axis_n, forward_n)))
            dot_val    = float(np.clip(np.dot(axis_n, forward_n), -1.0, 1.0))
            angle_deg  = float(np.degrees(np.arctan2(cross_norm, dot_val)))
            if angle_deg < constraint.min_angle_deg or angle_deg > constraint.max_angle_deg:
                violations.append(
                    f"Bone {constraint.bone_index} angle {angle_deg:.1f}° "
                    f"outside [{constraint.min_angle_deg:.0f}, {constraint.max_angle_deg:.0f}]"
                )
                penalty += 0.15

        penalty = min(1.0, penalty)
        return LegalityReport(
            is_legal=len(violations) == 0,
            violations=violations,
            corrected_transforms=None if len(violations) == 0 else self._correct(frame),
            confidence_penalty=penalty,
        )

    def _correct(self, frame: PoseFrame) -> List[np.ndarray]:
        corrected = []
        for bt in frame.bone_transforms:
            if not np.isfinite(bt).all():
                corrected.append(np.eye(4, dtype=np.float32))
            else:
                corrected.append(bt)
        return corrected

    def validate_transition(
        self,
        prev_frame: PoseFrame,
        curr_frame: PoseFrame,
        dt: float,
    ) -> Tuple[bool, float]:
        if dt < 1e-6:
            return True, 0.0

        pos_delta = np.linalg.norm(curr_frame.root_location - prev_frame.root_location)
        max_pos_delta = self._max_velocity_cm_s * dt

        if pos_delta > max_pos_delta * 1.5:
            return False, 0.5

        return True, 0.0

    @staticmethod
    def _default_constraints() -> List[JointConstraint]:
        return [
            JointConstraint(bone_index=4, min_angle_deg=-45.0, max_angle_deg=135.0),
            JointConstraint(bone_index=5, min_angle_deg=-45.0, max_angle_deg=135.0),
            JointConstraint(bone_index=6, min_angle_deg=-90.0, max_angle_deg=90.0),
            JointConstraint(bone_index=7, min_angle_deg=-90.0, max_angle_deg=90.0),
        ]
