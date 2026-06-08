"""
planning/uncertainty_controller.py
====================================
UncertaintyController — controle de ação baseado em entropia do ator.

Adaptado de realtime_brain.zip/realtime/uncertainty_controller.py:
  - Remove core.logger → logging padrão
  - Mantém UncertaintyMode, UncertaintyState, ActionModification
  - Integra com Policy.entropy()
  - Usado pelo SessionAgent para modular speed/aggressiveness por incerteza
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
CAUTIOUS_THRESHOLD:             float = 0.50
FALLBACK_THRESHOLD:             float = 0.999
CAUTIOUS_AGGRESSIVENESS_FACTOR: float = 0.70
CAUTIOUS_STOP_INTENT_DELTA:     float = 0.20
FALLBACK_SAFE_ACTIONS:          List[int] = [0, 1]
SMOOTHING_WINDOW:               int = 10
SAFE_ACTION_IDLE     = 0
SAFE_ACTION_WALK_FWD = 1


# ── Enums / Dataclasses ───────────────────────────────────────────────────────

class UncertaintyMode(str, Enum):
    NORMAL       = "normal"
    CAUTIOUS     = "cautious"
    FALLBACK_BIAS = "fallback_bias"


@dataclass
class UncertaintyState:
    mode:               UncertaintyMode
    raw_entropy:        float
    normalized_entropy: float
    smoothed_entropy:   float
    dominant_factor:    str
    factor_entropies:   Dict[str, float] = field(default_factory=dict)
    timestamp:          float = field(default_factory=time.time)

    @property
    def is_normal(self)   -> bool: return self.mode == UncertaintyMode.NORMAL
    @property
    def is_cautious(self) -> bool: return self.mode == UncertaintyMode.CAUTIOUS
    @property
    def is_fallback(self) -> bool: return self.mode == UncertaintyMode.FALLBACK_BIAS

    def to_dict(self) -> dict:
        return {
            "mode":               self.mode.value,
            "raw_entropy":        round(self.raw_entropy, 4),
            "normalized_entropy": round(self.normalized_entropy, 4),
            "smoothed_entropy":   round(self.smoothed_entropy, 4),
            "dominant_factor":    self.dominant_factor,
        }


@dataclass
class ActionModification:
    """Modificação aplicada a uma ação com base na incerteza."""
    original_action:       int
    modified_action:       int
    speed_multiplier:      float
    was_overridden:        bool
    mode:                  UncertaintyMode
    aggressiveness_factor: float = 1.0
    stop_intent_delta:     float = 0.0
    reason:                str   = ""

    def apply(
        self,
        action_idx:    int,
        speed:         float,
        move_dir:      List[float],
        aggressiveness: float = 0.5,
        stop_intent:   float = 0.0,
    ) -> Tuple[int, float, List[float], float, float]:
        new_action = self.modified_action if self.was_overridden else action_idx
        new_speed  = float(max(0.0, min(1.0, speed * self.speed_multiplier)))
        new_dir    = move_dir if not self.was_overridden else [0.0, 0.0, 0.0]
        new_agg    = float(max(0.0, min(1.0, aggressiveness * self.aggressiveness_factor)))
        new_stop   = float(max(0.0, min(1.0, stop_intent + self.stop_intent_delta)))
        return new_action, new_speed, new_dir, new_agg, new_stop


# ── Main controller ───────────────────────────────────────────────────────────

class UncertaintyController:
    """
    Classifica o modo de operação do NPC com base na entropia da política:
      NORMAL        → operação normal
      CAUTIOUS      → reduz velocidade/agressividade
      FALLBACK_BIAS → força ação segura (idle/walk)

    Thread-safe (chamado dentro do asyncio executor).
    """

    def __init__(
        self,
        cautious_threshold:             float = CAUTIOUS_THRESHOLD,
        fallback_threshold:             float = FALLBACK_THRESHOLD,
        cautious_aggressiveness_factor: float = CAUTIOUS_AGGRESSIVENESS_FACTOR,
        cautious_stop_intent_delta:     float = CAUTIOUS_STOP_INTENT_DELTA,
        smoothing_window:               int   = SMOOTHING_WINDOW,
        max_entropy_nats:               float = 3.0,
        enabled:                        bool  = True,
    ) -> None:
        self.cautious_threshold             = cautious_threshold
        self.fallback_threshold             = fallback_threshold
        self.cautious_aggressiveness_factor = cautious_aggressiveness_factor
        self.cautious_stop_intent_delta     = cautious_stop_intent_delta
        self.smoothing_window               = smoothing_window
        self.max_entropy_nats               = max_entropy_nats
        self.enabled                        = enabled

        self._current_mode:  UncertaintyMode           = UncertaintyMode.NORMAL
        self._current_state: Optional[UncertaintyState] = None
        self._entropy_window: Deque[float]              = deque(maxlen=smoothing_window)
        self._mode_counts: Dict[str, int]               = {m.value: 0 for m in UncertaintyMode}
        self._total_updates: int                        = 0
        self._recent_rewards: Deque[float]              = deque(maxlen=50)

        logger.info(
            f"UncertaintyController | cautious_th={cautious_threshold:.2f} "
            f"| fallback_th={fallback_threshold:.3f} | window={smoothing_window}"
        )

    def update(
        self,
        entropy_nats:      float,
        reward:            Optional[float] = None,
        factor_entropies:  Optional[Dict[str, float]] = None,
    ) -> UncertaintyState:
        """
        Atualiza modo com base na entropia atual (em nats).

        Args:
            entropy_nats:    Entropia da distribuição de ação (bits ou nats)
            reward:          Reward recente (opcional, para ajuste do proxy)
            factor_entropies: Entropias por fator cognitivo (opcional)
        """
        self._total_updates += 1

        if reward is not None:
            self._recent_rewards.append(float(reward))

        normalized = float(
            np.clip(entropy_nats / max(self.max_entropy_nats, 1e-6), 0.0, 1.0)
        )
        self._entropy_window.append(normalized)
        smoothed   = float(np.mean(self._entropy_window))
        mode       = self._classify(smoothed)

        if mode != self._current_mode:
            self._on_mode_change(self._current_mode, mode, smoothed)
            self._current_mode = mode

        fe = factor_entropies or {"policy": entropy_nats}
        dominant = max(fe, key=fe.get) if fe else "unknown"

        state = UncertaintyState(
            mode=mode,
            raw_entropy=entropy_nats,
            normalized_entropy=normalized,
            smoothed_entropy=smoothed,
            dominant_factor=dominant,
            factor_entropies=fe,
        )
        self._current_state = state
        self._mode_counts[mode.value] = self._mode_counts.get(mode.value, 0) + 1
        return state

    def get_action_modification(
        self,
        action_idx: int,
        speed:      float = 1.0,
        move_dir:   Optional[List[float]] = None,
    ) -> ActionModification:
        move_dir = move_dir or [0.0, 0.0, 0.0]
        mode     = self.get_mode()

        if mode == UncertaintyMode.NORMAL or not self.enabled:
            return ActionModification(action_idx, action_idx, 1.0, False, mode, reason="normal")

        if mode == UncertaintyMode.CAUTIOUS:
            return ActionModification(
                action_idx, action_idx, 1.0, False, mode,
                aggressiveness_factor=self.cautious_aggressiveness_factor,
                stop_intent_delta=self.cautious_stop_intent_delta,
                reason="cautious",
            )

        safe = SAFE_ACTION_IDLE if speed < 0.3 else SAFE_ACTION_WALK_FWD
        return ActionModification(
            action_idx, safe, 0.5,
            was_overridden=(action_idx not in FALLBACK_SAFE_ACTIONS),
            mode=mode,
            aggressiveness_factor=0.0,
            stop_intent_delta=0.5,
            reason=f"fallback: {action_idx}→{safe}",
        )

    def get_mode(self) -> UncertaintyMode:
        if not self.enabled:
            return UncertaintyMode.NORMAL
        return self._current_mode

    def should_dream(self) -> bool:
        return self.enabled and self._current_mode != UncertaintyMode.FALLBACK_BIAS

    def get_diagnostics(self) -> dict:
        return {
            "enabled":         self.enabled,
            "current_mode":    self._current_mode.value,
            "total_updates":   self._total_updates,
            "mode_counts":     dict(self._mode_counts),
            "smoothed_entropy": round(
                float(np.mean(self._entropy_window)) if self._entropy_window else 0.0, 4
            ),
        }

    def reset(self) -> None:
        self._current_mode = UncertaintyMode.NORMAL
        self._entropy_window.clear()
        self._current_state = None

    def _classify(self, smoothed: float) -> UncertaintyMode:
        if smoothed >= self.fallback_threshold:
            return UncertaintyMode.FALLBACK_BIAS
        if smoothed >= self.cautious_threshold:
            return UncertaintyMode.CAUTIOUS
        return UncertaintyMode.NORMAL

    def _on_mode_change(
        self, old: UncertaintyMode, new: UncertaintyMode, entropy: float
    ) -> None:
        if new == UncertaintyMode.FALLBACK_BIAS:
            logger.warning(
                f"UncertaintyController | FALLBACK_BIAS | entropy={entropy:.3f}"
            )
        elif new == UncertaintyMode.CAUTIOUS:
            logger.info(
                f"UncertaintyController | CAUTIOUS | entropy={entropy:.3f}"
            )
        else:
            logger.info(
                f"UncertaintyController | NORMAL restaurado | entropy={entropy:.3f}"
            )
