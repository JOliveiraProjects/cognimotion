"""
production/policy_drift_monitor.py
====================================
PolicyDriftMonitor — detecta drift entre versões de política via KL + reward drop.
Adaptado de training_brain.zip/production/policy_drift_monitor.py.
"""
from __future__ import annotations

import logging
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DriftSeverity(str, Enum):
    NONE   = "none"
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


@dataclass
class PolicySnapshot:
    version:     int
    saved_at:    float = field(default_factory=time.time)
    mean_logits: Optional[list] = None
    mean_reward: float = 0.0


@dataclass
class DriftResult:
    is_drifted:          bool = False
    severity:            DriftSeverity = DriftSeverity.NONE
    kl_divergence:       float = 0.0
    reward_drop_fraction: float = 0.0
    from_version:        int = 0
    to_version:          int = 0
    message:             str = ""


class PolicyDriftMonitor:
    """
    Monitora drift entre versões de política:
      - KL-divergência via obs sintéticas
      - Queda de reward via janela EMA
    Thread-safe.
    """

    def __init__(
        self,
        kl_threshold:           float = 0.5,
        reward_drop_threshold:  float = 0.20,
        window_size:            int   = 100,
        obs_dim:                int   = 256,   # embedding_dim do PoseEncoder
        n_synthetic_obs:        int   = 32,
        device:                 str   = "cpu",
    ) -> None:
        self.kl_threshold           = kl_threshold
        self.reward_drop_threshold  = reward_drop_threshold
        self.window_size            = window_size
        self.obs_dim                = obs_dim
        self.n_synthetic_obs        = n_synthetic_obs
        self.device                 = device

        self._lock:           threading.Lock = threading.Lock()
        self._snapshots:      List[PolicySnapshot] = []
        self._reward_window:  Deque[float] = deque(maxlen=window_size)
        self._drift_history:  List[DriftResult] = []
        self._synthetic_obs               = None

        logger.info(
            f"PolicyDriftMonitor | kl_thr={kl_threshold} "
            f"| rw_drop_thr={reward_drop_threshold} | window={window_size}"
        )

    # ──────────────────────────────────────────────────────────────────────────

    def record_snapshot(self, model, version: int, mean_reward: float = 0.0) -> None:
        logits = self._compute_mean_logits(model)
        snap   = PolicySnapshot(version=version, mean_logits=logits, mean_reward=mean_reward)
        with self._lock:
            self._snapshots.append(snap)
            if len(self._snapshots) > 5:
                self._snapshots.pop(0)
        logger.debug(f"PolicyDriftMonitor | snapshot v{version} | reward={mean_reward:.4f}")

    def record_reward(self, reward: float) -> None:
        with self._lock:
            self._reward_window.append(reward)

    def check(self, model, current_mean_reward: float) -> DriftResult:
        with self._lock:
            if len(self._snapshots) < 2:
                return DriftResult(message="Snapshots insuficientes")
            prev = self._snapshots[-2]
            curr = self._snapshots[-1]

        kl         = self._compute_kl(model, prev)
        rew_drop   = 0.0
        if abs(prev.mean_reward) > 1e-6:
            rew_drop = max(0.0, (prev.mean_reward - current_mean_reward) / abs(prev.mean_reward))

        severity, drifted = self._classify(kl, rew_drop)
        result = DriftResult(
            is_drifted=drifted, severity=severity,
            kl_divergence=kl, reward_drop_fraction=rew_drop,
            from_version=prev.version, to_version=curr.version,
            message=f"KL={kl:.4f} | rw_drop={rew_drop:.3f}",
        )
        if drifted:
            with self._lock:
                self._drift_history.append(result)
            logger.warning(f"PolicyDriftMonitor | DRIFT {severity.value.upper()} | {result.message}")

        return result

    # ──────────────────────────────────────────────────────────────────────────

    def _classify(self, kl: float, rw_drop: float) -> Tuple[DriftSeverity, bool]:
        if kl > self.kl_threshold or rw_drop > self.reward_drop_threshold:
            return DriftSeverity.HIGH, True
        if kl > self.kl_threshold * 0.4 or rw_drop > self.reward_drop_threshold * 0.5:
            return DriftSeverity.MEDIUM, False
        if kl > self.kl_threshold * 0.2 or rw_drop > self.reward_drop_threshold * 0.25:
            return DriftSeverity.LOW, False
        return DriftSeverity.NONE, False

    def _compute_kl(self, model, prev: PolicySnapshot) -> float:
        if prev.mean_logits is None:
            return 0.0
        try:
            import torch
            import torch.nn.functional as F
            obs = self._get_synthetic_obs()
            with torch.no_grad():
                out = model(obs)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                if not isinstance(out, torch.Tensor):
                    return 0.0
                curr_logits = out.float().mean(dim=0)
                prev_logits = torch.tensor(prev.mean_logits, dtype=torch.float32, device=self.device)
                p = F.softmax(prev_logits,  dim=-1).clamp(min=1e-8)
                q = F.softmax(curr_logits,  dim=-1).clamp(min=1e-8)
                return float(max(0.0, (p * (p / q).log()).sum().item()))
        except Exception as exc:
            logger.debug(f"PolicyDriftMonitor | KL falhou: {exc}")
            return 0.0

    def _compute_mean_logits(self, model) -> Optional[list]:
        try:
            import torch
            obs = self._get_synthetic_obs()
            with torch.no_grad():
                out = model(obs)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                return out.float().mean(dim=0).tolist() if isinstance(out, torch.Tensor) else None
        except Exception:
            return None

    def _get_synthetic_obs(self):
        import torch
        if self._synthetic_obs is None:
            g = torch.Generator(); g.manual_seed(42)
            self._synthetic_obs = torch.randn(
                self.n_synthetic_obs, self.obs_dim, generator=g, device=self.device
            )
        return self._synthetic_obs

    def get_current_reward_stats(self) -> Dict[str, float]:
        with self._lock:
            rewards = list(self._reward_window)
        if not rewards:
            return {"mean": 0.0, "std": 0.0, "n": 0}
        return {
            "mean": sum(rewards) / len(rewards),
            "std":  statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
            "n":    len(rewards),
        }

    def get_diagnostics(self) -> Dict[str, Any]:
        with self._lock:
            n_snap   = len(self._snapshots)
            n_drift  = len(self._drift_history)
            n_high   = sum(1 for d in self._drift_history if d.severity == DriftSeverity.HIGH)
        return {
            "snapshots":         n_snap,
            "drift_events":      n_drift,
            "high_drift_events": n_high,
            "kl_threshold":      self.kl_threshold,
            "reward_threshold":  self.reward_drop_threshold,
            **self.get_current_reward_stats(),
        }
