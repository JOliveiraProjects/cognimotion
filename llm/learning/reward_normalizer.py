"""
reward_normalizer.py
====================
Normalização online de rewards via Welford running statistics.
Adaptado de itens.zip — sem dependências externas além de stdlib.
Thread-safe.
"""
from __future__ import annotations

import math
import threading
from typing import Dict


class RewardNormalizer:
    """
    Normaliza rewards online usando média e variância de Welford.
    Clip para evitar explosão de gradiente.
    """

    def __init__(
        self,
        clip: float = 10.0,
        epsilon: float = 1e-8,
        warmup_steps: int = 100,
    ) -> None:
        if clip <= 0:
            raise ValueError(f"clip deve ser > 0, recebeu {clip}")
        self._clip = clip
        self._epsilon = epsilon
        self._warmup_steps = max(warmup_steps, 2)
        self._lock = threading.Lock()

        self._count: int = 0
        self._mean: float = 0.0
        self._m2: float = 0.0

    # ──────────────────────────────────────────────────────────────────────────

    def update(self, reward: float) -> None:
        """Atualiza estatísticas com um novo reward (Welford online)."""
        r = float(reward)
        with self._lock:
            self._count += 1
            delta = r - self._mean
            self._mean += delta / self._count
            delta2 = r - self._mean
            self._m2 += delta * delta2

    def normalize(self, reward: float) -> float:
        """Normaliza um reward usando as estatísticas acumuladas."""
        r = float(reward)
        with self._lock:
            if self._count < self._warmup_steps:
                return r
            variance = self._m2 / (self._count - 1)
            std = math.sqrt(max(variance, 0.0)) + self._epsilon

        normed = (r - self._mean) / std
        return float(max(-self._clip, min(self._clip, normed)))

    def update_and_normalize(self, reward: float) -> float:
        """Combina update + normalize em uma chamada."""
        self.update(reward)
        return self.normalize(reward)

    # ──────────────────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def mean(self) -> float:
        with self._lock:
            return self._mean

    @property
    def std(self) -> float:
        with self._lock:
            if self._count < 2:
                return 1.0
            return math.sqrt(self._m2 / (self._count - 1)) + self._epsilon

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    @property
    def is_warmed_up(self) -> bool:
        with self._lock:
            return self._count >= self._warmup_steps

    # ──────────────────────────────────────────────────────────────────────────
    # Serialização
    # ──────────────────────────────────────────────────────────────────────────

    def state_dict(self) -> Dict[str, float]:
        with self._lock:
            return {
                "count": float(self._count),
                "mean": self._mean,
                "m2": self._m2,
            }

    def load_state_dict(self, d: Dict[str, float]) -> None:
        with self._lock:
            self._count = int(d["count"])
            self._mean = float(d["mean"])
            self._m2 = float(d["m2"])
