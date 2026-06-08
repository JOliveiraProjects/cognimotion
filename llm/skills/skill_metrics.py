from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np



import logging
logger = logging.getLogger(__name__)

DEFAULT_MAX_SPEED_CMS: float = 600.0


@dataclass
class SkillFrameMetrics:
    skill_name: str = ""
    context: str = "open_area"
    action_idx: int = 0
    frame: int = -1

    reward: float = 0.0

    velocity_reward: float = 0.0
    speed_reward: float = 0.0
    facing_reward: float = 0.0

    trajectory_error: float = 0.0
    velocity_error: float = 0.0
    facing_error: float = 0.0
    facing_error_deg: float = 0.0

    pose_cost: float = 0.0
    foot_sliding: float = 0.0

    prediction_error: float = 0.0

    jitter: float = 0.0

    is_falling: bool = False
    movement_mode: str = "walking"
    actual_speed: float = 0.0
    desired_speed: float = 0.0

    @property
    def is_valid(self) -> bool:
        return self.skill_name != "" and self.frame >= 0

    @property
    def composite_degradation(self) -> float:
        return (
            self.trajectory_error * 0.30
            + self.velocity_error * 0.20
            + self.foot_sliding * 0.20
            + min(self.facing_error / math.pi, 1.0) * 0.15
            + min(self.pose_cost / 10.0, 1.0) * 0.15
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "context": self.context,
            "action_idx": self.action_idx,
            "frame": self.frame,
            "reward": round(self.reward, 4),
            "trajectory_error": round(self.trajectory_error, 4),
            "velocity_error": round(self.velocity_error, 4),
            "facing_error_deg": round(self.facing_error_deg, 2),
            "pose_cost": round(self.pose_cost, 4),
            "foot_sliding": round(self.foot_sliding, 4),
            "prediction_error": round(self.prediction_error, 4),
            "jitter": round(self.jitter, 4),
            "is_falling": self.is_falling,
            "actual_speed": round(self.actual_speed, 2),
            "desired_speed": round(self.desired_speed, 2),
        }


class SkillMetrics:

    def __init__(self, max_speed_cms: float = DEFAULT_MAX_SPEED_CMS):
        self.max_speed_cms = max_speed_cms

    def compute(
        self,
        skill_name: str,
        context: str,
        action_idx: int,
        observation: Dict[str, Any],
        frame: int = -1,
        prev_observation: Optional[Dict[str, Any]] = None,
        predicted_obs: Optional[np.ndarray] = None,
    ) -> SkillFrameMetrics:

        env_state = observation.get("env_state", {})
        mm = env_state.get("motion_metrics", {})

        result = SkillFrameMetrics(
            skill_name=skill_name,
            context=context,
            action_idx=action_idx,
            frame=frame,
        )

        actual_vel = self._safe_vec3(mm.get("actual_velocity", env_state.get("actual_velocity")))
        desired_vel = self._safe_vec3(mm.get("desired_velocity", env_state.get("desired_velocity")))
        actual_spd = self._safe_float(mm.get("actual_speed", mm.get("speed", 0.0)))
        desired_spd = self._safe_float(mm.get("desired_speed", env_state.get("desired_speed", actual_spd)))
        result.actual_speed = actual_spd
        result.desired_speed = desired_spd

        movement_mode = env_state.get("movement_mode", mm.get("movement_mode", "walking"))
        result.movement_mode = str(movement_mode)
        result.is_falling = str(movement_mode).lower() in {"falling", "fall", "inair"}

        vel_norm_actual = self._normalize(actual_vel)
        vel_norm_desired = self._normalize(desired_vel)
        velocity_reward = float(np.clip(np.dot(vel_norm_actual, vel_norm_desired), -1.0, 1.0))
        result.velocity_reward = velocity_reward

        result.trajectory_error = float(np.clip((1.0 - velocity_reward) / 2.0, 0.0, 1.0))

        max_spd = max(self.max_speed_cms, 1e-8)
        speed_error = abs(actual_spd - desired_spd) / max_spd
        result.velocity_error = float(np.clip(speed_error, 0.0, 1.0))
        result.speed_reward = 1.0 - result.velocity_error

        actor_fwd = self._safe_vec3(mm.get("actor_forward", env_state.get("actor_forward")))
        desired_facing = self._safe_vec3(mm.get("desired_facing", env_state.get("desired_facing")))
        fwd_norm = self._normalize(actor_fwd)
        face_norm = self._normalize(desired_facing)
        facing_dot = float(np.clip(np.dot(fwd_norm, face_norm), -1.0, 1.0))
        result.facing_reward = facing_dot

        result.facing_error = float(np.arccos(np.clip(facing_dot, -1.0, 1.0)))
        result.facing_error_deg = math.degrees(result.facing_error)

        pose_cost = self._safe_float(mm.get("pose_cost", 0.0))
        foot_sliding = self._safe_float(mm.get("foot_sliding", 0.0))
        result.pose_cost = float(max(0.0, pose_cost))
        result.foot_sliding = float(np.clip(foot_sliding, 0.0, 1.0))

        if prev_observation is not None:
            prev_env = prev_observation.get("env_state", {})
            prev_dir = self._safe_vec3(prev_env.get("move_direction"))
            curr_dir = self._safe_vec3(env_state.get("move_direction"))
            delta = float(np.linalg.norm(curr_dir - prev_dir))
            result.jitter = float(np.clip(delta, 0.0, 1.0))

        if predicted_obs is not None:
            obs_raw = observation.get("obs")
            if obs_raw is not None:
                try:
                    obs_arr = np.array(obs_raw, dtype=np.float32)
                    pred_arr = np.array(predicted_obs, dtype=np.float32)
                    if obs_arr.shape == pred_arr.shape:
                        result.prediction_error = float(
                            np.mean((obs_arr - pred_arr) ** 2)
                        )
                except Exception:
                    pass

        result.reward = self._compute_reward(
            velocity_reward=velocity_reward,
            speed_reward=result.speed_reward,
            facing_reward=result.facing_reward,
            foot_sliding=result.foot_sliding,
            pose_cost=result.pose_cost,
            jitter_penalty=result.jitter,
            is_falling=result.is_falling,
        )

        return result

    @staticmethod
    def _compute_reward(
        velocity_reward: float,
        speed_reward: float,
        facing_reward: float,
        foot_sliding: float,
        pose_cost: float,
        jitter_penalty: float,
        is_falling: bool,
    ) -> float:

        pose_cost_norm = float(min(pose_cost / 10.0, 1.0)) if pose_cost > 0 else 0.0

        r = (
            0.35 * velocity_reward
            + 0.20 * speed_reward
            + 0.15 * facing_reward
            - 0.15 * foot_sliding
            - 0.10 * pose_cost_norm
            - 0.10 * jitter_penalty
        )
        if is_falling:
            r -= 0.50
        return float(np.clip(r, -1.0, 1.0))

    @staticmethod
    def _safe_float(v: Any, default: float = 0.0) -> float:
        try:
            f = float(v)
            return f if math.isfinite(f) else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_vec3(v: Any) -> np.ndarray:
        if not isinstance(v, (list, tuple, np.ndarray)) or len(v) < 3:
            return np.zeros(3, dtype=np.float64)
        try:
            arr = np.array([float(x) for x in v[:3]], dtype=np.float64)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            return arr
        except (TypeError, ValueError):
            return np.zeros(3, dtype=np.float64)

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        return v / n if n >= 1e-8 else np.zeros_like(v)


def aggregate_window(
    frames: List[SkillFrameMetrics],
) -> Optional[Dict[str, float]]:
    if not frames:
        return None

    def stats(values: List[float]) -> Tuple[float, float]:
        if not values:
            return 0.0, 0.0
        arr = np.array(values, dtype=np.float64)
        return float(np.mean(arr)), float(np.std(arr))

    rewards = [f.reward for f in frames]
    traj_errors = [f.trajectory_error for f in frames]
    vel_errors = [f.velocity_error for f in frames]
    pose_costs = [f.pose_cost for f in frames]
    foot_slidings = [f.foot_sliding for f in frames]
    facing_errors = [f.facing_error_deg for f in frames]
    pred_errors = [f.prediction_error for f in frames]

    mean_r, std_r = stats(rewards)
    mean_te, std_te = stats(traj_errors)
    mean_ve, _ = stats(vel_errors)
    mean_pc, _ = stats(pose_costs)
    mean_fs, _ = stats(foot_slidings)
    mean_fe, _ = stats(facing_errors)
    mean_pe, _ = stats(pred_errors)

    return {
        "sample_count": len(frames),
        "mean_reward": mean_r,
        "std_reward": std_r,
        "mean_trajectory_error": mean_te,
        "std_trajectory_error": std_te,
        "mean_velocity_error": mean_ve,
        "mean_pose_cost": mean_pc,
        "mean_foot_sliding": mean_fs,
        "mean_facing_error": mean_fe,
        "mean_prediction_error": mean_pe,
    }
