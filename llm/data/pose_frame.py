from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class TrajectorySample:
    position: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    facing: np.ndarray
    time_in_seconds: float
    speed: float

    @staticmethod
    def zero() -> "TrajectorySample":
        return TrajectorySample(
            position=np.zeros(3, dtype=np.float32),
            linear_velocity=np.zeros(3, dtype=np.float32),
            angular_velocity=np.zeros(3, dtype=np.float32),
            facing=np.array([1., 0., 0., 0.], dtype=np.float32),
            time_in_seconds=0.0,
            speed=0.0,
        )

    def to_numpy(self) -> np.ndarray:
        return np.concatenate([
            self.position,
            self.linear_velocity,
            self.angular_velocity,
            self.facing,
            [self.time_in_seconds, self.speed],
        ]).astype(np.float32)


@dataclass
class Trajectory:
    samples: List[TrajectorySample] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.samples) > 0

    def to_numpy(self, n_samples: int = 6) -> np.ndarray:
        if not self.samples:
            return np.zeros((n_samples, 12), dtype=np.float32)
        padded = self.samples[:n_samples]
        while len(padded) < n_samples:
            padded.append(TrajectorySample.zero())
        return np.stack([s.to_numpy()[:12] for s in padded], axis=0).astype(np.float32)

    @staticmethod
    def idle(n_samples: int = 6, dt: float = 0.05) -> "Trajectory":
        samples = []
        for i in range(n_samples):
            s = TrajectorySample.zero()
            s.time_in_seconds = (i + 1) * dt
            samples.append(s)
        return Trajectory(samples=samples)


@dataclass
class PoseFrame:
    timestamp: float
    frame_index: int
    root_location: np.ndarray
    root_rotation: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    bone_transforms: List[np.ndarray]
    past_trajectory: Trajectory
    future_trajectory: Trajectory
    curve_values: Dict[str, float]
    tags: List[str]
    movement_mode: int
    motion_style: int

    @staticmethod
    def zero(bone_count: int = 9) -> "PoseFrame":
        return PoseFrame(
            timestamp=0.0,
            frame_index=0,
            root_location=np.zeros(3, dtype=np.float32),
            root_rotation=np.array([0., 0., 0., 1.], dtype=np.float32),
            linear_velocity=np.zeros(3, dtype=np.float32),
            angular_velocity=np.zeros(3, dtype=np.float32),
            bone_transforms=[np.eye(4, dtype=np.float32) for _ in range(bone_count)],
            past_trajectory=Trajectory(),
            future_trajectory=Trajectory.idle(),
            curve_values={},
            tags=[],
            movement_mode=0,
            motion_style=0,
        )

    @staticmethod
    def _rot_matrix_to_quat(m: np.ndarray) -> np.ndarray:
        """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
        trace = m[0, 0] + m[1, 1] + m[2, 2]
        if trace > 0.0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (m[2, 1] - m[1, 2]) * s
            y = (m[0, 2] - m[2, 0]) * s
            z = (m[1, 0] - m[0, 1]) * s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
        return np.array([w, x, y, z], dtype=np.float32)

    def to_observation_vector(self) -> np.ndarray:
        # Root features: 3+4+3+3 = 13 floats  (matches root_input_dim in PoseEncoder)
        parts = [
            self.root_location,    # 3
            self.root_rotation,    # 4
            self.linear_velocity,  # 3
            self.angular_velocity, # 3
        ]

        # Bone features: bone_dim=10 per bone = loc(3) + quat(4) + scale(3)
        # This matches BoneEncoder(bone_dim=10) and encode_frame slice [13 : 13+bone_count*10]
        for bt in self.bone_transforms:
            if bt.shape == (4, 4):
                loc   = bt[:3, 3]                                             # (3,)
                quat  = self._rot_matrix_to_quat(bt[:3, :3])                  # (4,)
                scale = np.array([
                    np.linalg.norm(bt[:3, 0]),
                    np.linalg.norm(bt[:3, 1]),
                    np.linalg.norm(bt[:3, 2]),
                ], dtype=np.float32)                                           # (3,)
                parts.append(loc)    # 3
                parts.append(quat)   # 4
                parts.append(scale)  # 3  → 10 per bone ✓
            else:
                # Flat bone vector — pad/clip to exactly bone_dim=10
                flat = bt.flatten().astype(np.float32)
                if len(flat) >= 10:
                    parts.append(flat[:10])
                else:
                    padded = np.zeros(10, dtype=np.float32)
                    padded[:len(flat)] = flat
                    parts.append(padded)

        # Trajectory features: 6 past + 6 future samples × 12 floats = 144
        parts.append(self.past_trajectory.to_numpy(6).flatten())
        parts.append(self.future_trajectory.to_numpy(6).flatten())

        # Misc: movement_mode + motion_style = 2
        parts.append(np.array([self.movement_mode, self.motion_style], dtype=np.float32))

        # Total: 13 + 9*10 + 144 + 2 = 249 floats
        return np.concatenate(parts).astype(np.float32)
