from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import numpy as np
from data.pose_frame import TrajectorySample, Trajectory


@dataclass
class TrajectoryFrame:
    past: Trajectory
    future: Trajectory
    session_id: str
    timestamp: float
    npc_state: int
    motion_style: int

    @staticmethod
    def from_samples(
        past_samples: List[TrajectorySample],
        future_samples: List[TrajectorySample],
        session_id: str = "",
        timestamp: float = 0.0,
        npc_state: int = 0,
        motion_style: int = 0,
    ) -> "TrajectoryFrame":
        return TrajectoryFrame(
            past=Trajectory(samples=past_samples),
            future=Trajectory(samples=future_samples),
            session_id=session_id,
            timestamp=timestamp,
            npc_state=npc_state,
            motion_style=motion_style,
        )

    def to_numpy(self, n_past: int = 6, n_future: int = 6) -> np.ndarray:
        return np.concatenate([
            self.past.to_numpy(n_past).flatten(),
            self.future.to_numpy(n_future).flatten(),
            np.array([self.npc_state, self.motion_style], dtype=np.float32),
        ]).astype(np.float32)

    @property
    def is_valid(self) -> bool:
        return self.past.is_valid or self.future.is_valid

    @staticmethod
    def idle(dt: float = 0.05, n_samples: int = 6) -> "TrajectoryFrame":
        import time
        idle_traj = Trajectory.idle(n_samples=n_samples, dt=dt)
        return TrajectoryFrame(
            past=idle_traj,
            future=idle_traj,
            session_id="",
            timestamp=time.time(),
            npc_state=0,
            motion_style=0,
        )
