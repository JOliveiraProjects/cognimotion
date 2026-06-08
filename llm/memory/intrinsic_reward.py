"""
memory/intrinsic_reward.py
===========================
IntrinsicRewardModule — recompensa intrínseca por curiosidade via EpisodicMemory.

Implementa Item 7 do documento de integração:
  "Use a memória para recompensa de curiosidade: comparar o estado latente
   atual com episódios similares na memória (quanto mais 'novo', maior a recompensa)."

Estratégia:
  - Distância no espaço latente (z embedding) ao vizinho mais próximo na memória
  - Quanto maior a distância → mais "nova" a situação → maior recompensa intrínseca
  - Beta decai com o tempo para equilibrar exploração vs exploração

Thread-safe.
"""
from __future__ import annotations

import logging
import math
import threading
from collections import deque
from typing import Deque, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class IntrinsicRewardModule:
    """
    Calcula recompensa intrínseca baseada em novidade do estado latente.

    Uso no serviço:
        intrinsic = IntrinsicRewardModule(embedding_dim=1024, beta=0.1)
        r_int = intrinsic.compute(z_embedding)
        total_reward = r_ext + r_int
    """

    def __init__(
        self,
        embedding_dim:   int   = 1024,   # RSSM.stochastic_dim
        beta:            float = 0.1,    # escala da recompensa intrínseca
        beta_decay:      float = 0.9999, # decaimento exponencial por step
        beta_min:        float = 0.01,
        k_neighbors:     int   = 5,      # vizinhos mais próximos na memória
        memory_capacity: int   = 10_000, # max embeddings na memória de curiosidade
        novelty_kernel:  str   = "gaussian",  # "gaussian" | "knn_distance"
    ) -> None:
        self.embedding_dim   = embedding_dim
        self.beta            = beta
        self.beta_decay      = beta_decay
        self.beta_min        = beta_min
        self.k               = k_neighbors
        self.capacity        = memory_capacity
        self.novelty_kernel  = novelty_kernel

        self._memory:        Deque[np.ndarray] = deque(maxlen=memory_capacity)
        self._step:          int   = 0
        self._current_beta:  float = beta
        self._lock = threading.Lock()

        # Histórico para normalização
        self._reward_history: Deque[float] = deque(maxlen=1000)

        logger.info(
            f"IntrinsicRewardModule | embedding_dim={embedding_dim} "
            f"| beta={beta} | k={k_neighbors} | capacity={memory_capacity}"
        )

    # ──────────────────────────────────────────────────────────────────────────

    def compute(self, z_embedding: np.ndarray) -> float:
        """
        Computa recompensa intrínseca para um embedding latente.

        Args:
            z_embedding: vetor numpy (stochastic_dim,)

        Returns:
            r_intrinsic: float ≥ 0
        """
        with self._lock:
            r_int = self._compute_novelty(z_embedding)
            self._memory.append(z_embedding.copy())
            self._step += 1
            self._current_beta = max(
                self.beta_min,
                self._current_beta * self.beta_decay,
            )
            self._reward_history.append(r_int)

        return float(r_int * self._current_beta)

    def compute_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Batch version: embeddings (N, embedding_dim) → r_intrinsic (N,)
        """
        results = np.zeros(embeddings.shape[0], dtype=np.float32)
        for i in range(embeddings.shape[0]):
            results[i] = self.compute(embeddings[i])
        return results

    # ──────────────────────────────────────────────────────────────────────────

    def _compute_novelty(self, z: np.ndarray) -> float:
        if len(self._memory) < self.k:
            return 1.0   # buffer vazio → máxima novidade

        mem = np.stack(list(self._memory), axis=0)   # (N, D)
        z_n = z / (np.linalg.norm(z) + 1e-8)
        m_n = mem / (np.linalg.norm(mem, axis=1, keepdims=True) + 1e-8)

        # Distâncias euclidianas no espaço normalizado
        diffs    = m_n - z_n[None, :]
        dists    = np.linalg.norm(diffs, axis=1)

        # Vizinhos mais próximos
        k        = min(self.k, len(dists))
        knn_dist = np.partition(dists, k - 1)[:k]
        mean_knn = float(np.mean(knn_dist))

        if self.novelty_kernel == "gaussian":
            # Recompensa gaussiana: mais distante = mais novo
            sigma  = max(self._running_std(), 1e-4)
            novelty = 1.0 - math.exp(-(mean_knn ** 2) / (2 * sigma ** 2))
        else:
            # Distância direta normalizada
            novelty = min(mean_knn / 2.0, 1.0)

        return float(novelty)

    def _running_std(self) -> float:
        if len(self._reward_history) < 10:
            return 1.0
        hist = np.array(list(self._reward_history), dtype=np.float32)
        return float(np.std(hist) + 1e-8)

    # ──────────────────────────────────────────────────────────────────────────

    def add_to_memory(self, z_embedding: np.ndarray) -> None:
        """Adiciona embedding à memória sem calcular recompensa."""
        with self._lock:
            self._memory.append(z_embedding.copy())

    def reset_memory(self) -> None:
        with self._lock:
            self._memory.clear()
            self._step = 0
            self._current_beta = self.beta
        logger.info("IntrinsicRewardModule | memória resetada")

    def get_diagnostics(self) -> dict:
        with self._lock:
            hist = list(self._reward_history)
        return {
            "memory_size":   len(self._memory),
            "step":          self._step,
            "current_beta":  round(self._current_beta, 5),
            "mean_novelty":  round(float(np.mean(hist)) if hist else 0.0, 4),
            "std_novelty":   round(float(np.std(hist))  if hist else 0.0, 4),
        }
