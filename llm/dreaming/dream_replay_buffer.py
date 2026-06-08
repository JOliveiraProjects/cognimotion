"""
dreaming/dream_replay_buffer.py
=================================
DreamReplayBuffer — controla a mistura de dados reais e imaginados no treinamento.

Adaptado de worldmodel_dreaming.zip/dreaming/dream_replay_buffer.py:
  - Remove core.logger → logging padrão
  - Mantém DreamBatch, add, should_use_dream, adjust_weight
  - Integrado com ImaginationEngine e DreamerTrainer
"""
from __future__ import annotations

import logging
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

RSSM_ERROR_THRESHOLD:  float = 1.0
INITIAL_DREAM_WEIGHT:  float = 0.3
MIN_DREAM_WEIGHT:      float = 0.05
MAX_DREAM_WEIGHT:      float = 0.5
MIN_REAL_SAMPLES:      int   = 500
MAX_DREAM_BATCH_SIZE:  int   = 10_000


@dataclass
class DreamBatch:
    skill_name:  str
    context:     str
    n_steps:     int
    rssm_error:  float
    confidence:  float
    created_at:  float = field(default_factory=time.time)
    rewards:     List[float] = field(default_factory=list)
    actions:     List[int]   = field(default_factory=list)
    dones:       List[bool]  = field(default_factory=list)

    @property
    def mean_reward(self) -> float:
        return float(np.mean(self.rewards)) if self.rewards else 0.0

    @property
    def is_usable(self) -> bool:
        return self.confidence >= 0.3 and len(self.rewards) > 0

    def to_dict(self) -> dict:
        return {
            "skill_name": self.skill_name,
            "n_steps":    self.n_steps,
            "rssm_error": round(self.rssm_error, 4),
            "confidence": round(self.confidence, 4),
            "mean_reward": round(self.mean_reward, 4),
            "is_usable":  self.is_usable,
        }


class DreamReplayBuffer:
    """
    Buffer que armazena batches de sonho (imaginados) e controla o peso
    de mistura com dados reais durante o treinamento.

    Regra de ajuste de peso:
      - Se RSSM error alto → reduz dream_weight (sonhos pouco confiáveis)
      - Se RSSM error baixo → mantém ou aumenta dream_weight
    """

    def __init__(
        self,
        min_real_samples:    int   = MIN_REAL_SAMPLES,
        initial_dream_weight: float = INITIAL_DREAM_WEIGHT,
        min_dream_weight:    float = MIN_DREAM_WEIGHT,
        max_dream_weight:    float = MAX_DREAM_WEIGHT,
        rssm_error_threshold: float = RSSM_ERROR_THRESHOLD,
        max_batches:         int   = 200,
        max_steps_stored:    int   = MAX_DREAM_BATCH_SIZE,
    ) -> None:
        self.min_real_samples     = min_real_samples
        self.dream_weight         = initial_dream_weight
        self.min_dream_weight     = min_dream_weight
        self.max_dream_weight     = max_dream_weight
        self.rssm_error_threshold = rssm_error_threshold
        self.max_batches          = max_batches
        self.max_steps_stored     = max_steps_stored

        self._batches:              Deque[DreamBatch] = deque(maxlen=max_batches)
        self._total_steps_stored:   int  = 0
        self._total_batches_added:  int  = 0
        self._weight_reductions:    int  = 0
        self._lock = threading.Lock()

        logger.info(
            f"DreamReplayBuffer | initial_weight={initial_dream_weight:.2f} "
            f"| min={min_dream_weight:.2f} | max={max_dream_weight:.2f} "
            f"| rssm_thr={rssm_error_threshold:.2f}"
        )

    # ──────────────────────────────────────────────────────────────────────────

    def add(self, batch: DreamBatch, real_buffer_size: int) -> bool:
        """Adiciona batch ao buffer. Retorna False se rejeitado."""
        if not batch.is_usable:
            logger.debug("DreamReplayBuffer | batch rejeitado (confiança baixa)")
            return False

        if real_buffer_size < self.min_real_samples:
            logger.debug(
                f"DreamReplayBuffer | real buffer insuficiente "
                f"({real_buffer_size} < {self.min_real_samples})"
            )
            return False

        with self._lock:
            self._batches.append(batch)
            self._total_steps_stored += batch.n_steps
            self._total_batches_added += 1

        # Ajusta peso com base no erro do RSSM
        self._adjust_weight(batch.rssm_error)

        logger.debug(
            f"DreamReplayBuffer | add | skill={batch.skill_name} "
            f"| rssm_err={batch.rssm_error:.4f} | "
            f"weight={self.dream_weight:.2f}"
        )
        return True

    def should_use_dream(
        self,
        real_buffer_size: int,
        current_rssm_error: float = 0.0,
    ) -> bool:
        """Decide se deve usar dados de sonho neste step."""
        if real_buffer_size < self.min_real_samples:
            return False
        if current_rssm_error > self.rssm_error_threshold:
            return False
        with self._lock:
            return len(self._batches) > 0

    def sample_dream_batch(self) -> Optional[DreamBatch]:
        """Retorna um batch aleatório de sonho."""
        import random
        with self._lock:
            if not self._batches:
                return None
            return random.choice(list(self._batches))

    def _adjust_weight(self, rssm_error: float) -> None:
        """Ajusta dream_weight com base na qualidade do RSSM."""
        if rssm_error > self.rssm_error_threshold:
            new_w = max(self.min_dream_weight, self.dream_weight * 0.95)
            if new_w < self.dream_weight:
                self._weight_reductions += 1
            self.dream_weight = new_w
        else:
            # RSSM preciso — pode aumentar levemente o peso
            self.dream_weight = min(self.max_dream_weight, self.dream_weight * 1.001)

    # ──────────────────────────────────────────────────────────────────────────

    def get_effective_dream_weight(
        self, real_buffer_size: int, current_rssm_error: float = 0.0
    ) -> float:
        """Retorna dream_weight efetivo (0 se condições não satisfeitas)."""
        if not self.should_use_dream(real_buffer_size, current_rssm_error):
            return 0.0
        return self.dream_weight

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_batches":   self._total_batches_added,
                "stored_batches":  len(self._batches),
                "total_steps":     self._total_steps_stored,
                "dream_weight":    round(self.dream_weight, 4),
                "weight_reductions": self._weight_reductions,
            }
