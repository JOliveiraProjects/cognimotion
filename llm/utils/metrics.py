from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import time


@dataclass
class MotionQualityMetrics:
    foot_sliding: float = 0.0
    smoothness: float = 1.0
    imitation_score: float = 0.0
    trajectory_error: float = 0.0
    velocity_error: float = 0.0
    prediction_error: float = 0.0
    confidence: float = 0.0
    latency_ms: float = 0.0

    @property
    def overall_score(self) -> float:
        foot_ok    = max(0.0, 1.0 - self.foot_sliding * 2.0)
        smooth_ok  = self.smoothness
        imitate_ok = self.imitation_score
        traj_ok    = max(0.0, 1.0 - self.trajectory_error)
        return (foot_ok * 0.3 + smooth_ok * 0.2 + imitate_ok * 0.3 + traj_ok * 0.2)

    def to_dict(self) -> Dict[str, float]:
        return {
            "foot_sliding": self.foot_sliding,
            "smoothness": self.smoothness,
            "imitation_score": self.imitation_score,
            "trajectory_error": self.trajectory_error,
            "velocity_error": self.velocity_error,
            "prediction_error": self.prediction_error,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "overall_score": self.overall_score,
        }


class MetricsAggregator:
    def __init__(self, window_size: int = 200):
        self._window  = window_size
        self._history: List[MotionQualityMetrics] = []
        self._latency_history: List[float] = []
        self._step_count = 0
        self._start_time = time.time()

    def record(self, metrics: MotionQualityMetrics) -> None:
        self._history.append(metrics)
        self._latency_history.append(metrics.latency_ms)
        if len(self._history) > self._window:
            self._history.pop(0)
            self._latency_history.pop(0)
        self._step_count += 1

    def compute_window_stats(self) -> Dict[str, float]:
        if not self._history:
            return {"steps": self._step_count, "uptime_s": time.time() - self._start_time}
        field_names = ["foot_sliding", "smoothness", "imitation_score",
                       "trajectory_error", "velocity_error", "confidence", "latency_ms"]
        stats: Dict[str, float] = {}
        for field in field_names:
            vals = np.array([getattr(m, field) for m in self._history], dtype=np.float64)
            if not np.isfinite(vals).all():
                vals = vals[np.isfinite(vals)]
            stats[f"mean_{field}"] = float(np.mean(vals)) if len(vals) > 0 else 0.0
            stats[f"max_{field}"]  = float(np.max(vals))  if len(vals) > 0 else 0.0
        overall = np.array([m.overall_score for m in self._history], dtype=np.float64)
        stats["overall_mean"] = float(np.mean(overall)) if len(overall) > 0 else 0.0
        if self._latency_history:
            stats["p99_latency"] = float(np.percentile(
                np.array(self._latency_history, dtype=np.float64), 99))
        else:
            stats["p99_latency"] = 0.0
        stats["steps"]    = float(self._step_count)
        stats["uptime_s"] = time.time() - self._start_time
        return stats


def compute_foot_sliding(
    left_foot_pos: np.ndarray,
    right_foot_pos: np.ndarray,
    prev_left: np.ndarray,
    prev_right: np.ndarray,
    character_speed: float,
    dt: float,
) -> float:
    if dt < 1e-6 or character_speed < 1.0:
        return 0.0
    left_slide  = float(np.linalg.norm(left_foot_pos[:2]  - prev_left[:2]))
    right_slide = float(np.linalg.norm(right_foot_pos[:2] - prev_right[:2]))
    expected    = character_speed * dt
    return (left_slide + right_slide) * 0.5 / max(expected, 1e-6)


def compute_smoothness(
    velocity_sequence: List[np.ndarray],
) -> float:
    if len(velocity_sequence) < 3:
        return 1.0
    jerk_sum = 0.0
    for i in range(2, len(velocity_sequence)):
        accel_now  = velocity_sequence[i]   - velocity_sequence[i - 1]
        accel_prev = velocity_sequence[i-1] - velocity_sequence[i - 2]
        jerk_sum  += float(np.linalg.norm(accel_now - accel_prev))
    return float(np.exp(-jerk_sum / max(len(velocity_sequence), 1)))


def compute_trajectory_error(
    predicted: np.ndarray,
    actual: np.ndarray,
    n_samples: int = 6,
) -> float:
    if predicted.shape != actual.shape:
        return 1.0
    return float(np.mean(np.linalg.norm(predicted[:n_samples, :3] - actual[:n_samples, :3], axis=-1)))
