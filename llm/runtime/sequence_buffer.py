"""
runtime/sequence_buffer.py
===========================
SequenceBuffer — buffer de sequências para treinar o RSSM.

Cada sequência é composta por transições consecutivas:
  (obs_enc, action, reward, done)

Suporta:
  - Indexação por group_key (leader_id, follower_id) para aprendizado personalizado
  - sample_sequence() para o WorldModelTrainer
  - add_sequence() para o handler de MSG_LEADER_SEQUENCE
  - Thread-safe
"""
from __future__ import annotations

import logging
import random
import threading
from collections import defaultdict, deque
from typing import Dict, Deque, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class SequenceBuffer:
    """
    Buffer de sequências de embeddings (obs_enc) para treinamento do RSSM.

    Estrutura interna:
      _episodes[group_key] = deque de episódios
      Cada episódio = lista de transições (obs_enc, action, reward, done)
    """

    def __init__(
        self,
        capacity:   int = 200_000,   # total de transições
        obs_dim:    int = 256,
        action_dim: int = 9,
        seq_len:    int = 16,
    ) -> None:
        self.capacity   = capacity
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.seq_len    = seq_len

        self._episodes: Dict[str, Deque[list]] = defaultdict(lambda: deque(maxlen=2000))
        self._total_transitions: int = 0
        self._lock = threading.RLock()

        logger.info(
            f"SequenceBuffer | capacity={capacity} | obs_dim={obs_dim} "
            f"| action_dim={action_dim} | seq_len={seq_len}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Inserção
    # ──────────────────────────────────────────────────────────────────────────

    def add_sequence(
        self,
        obs_seq:    np.ndarray,       # (T, obs_dim)
        action_seq: Optional[np.ndarray] = None,  # (T, action_dim) | None → zeros
        reward_seq: Optional[np.ndarray] = None,  # (T,) | None → zeros
        done_seq:   Optional[np.ndarray] = None,  # (T,) bool | None → [False, ..., True]
        group_key:  str = "global",
        pose_seq:   Optional[np.ndarray] = None,  # (T, num_bones*7) | None → zeros (alvo do PoseDecoder)
    ) -> None:
        """
        Adiciona uma sequência de frames ao buffer.
        Se action/reward/done forem None, preenche com defaults.
        """
        T = obs_seq.shape[0]
        if T < 2:
            return

        if action_seq is None:
            action_seq = np.zeros((T, self.action_dim), dtype=np.float32)
        if reward_seq is None:
            reward_seq = np.zeros(T, dtype=np.float32)
        if done_seq is None:
            done_seq = np.zeros(T, dtype=bool)
            done_seq[-1] = True

        # Normaliza shapes
        obs_seq    = np.asarray(obs_seq, dtype=np.float32)
        action_seq = np.asarray(action_seq, dtype=np.float32)
        reward_seq = np.asarray(reward_seq, dtype=np.float32)
        done_seq   = np.asarray(done_seq, dtype=bool)

        # Clip or pad action dimension to match buffer config — prevents RSSM input_size mismatch
        if action_seq.ndim == 2 and action_seq.shape[1] != self.action_dim:
            if action_seq.shape[1] > self.action_dim:
                action_seq = action_seq[:, :self.action_dim]
            else:
                pad = np.zeros((action_seq.shape[0], self.action_dim - action_seq.shape[1]), dtype=np.float32)
                action_seq = np.concatenate([action_seq, pad], axis=1)

        if pose_seq is None:
            pose_seq = np.zeros((T, 1), dtype=np.float32)  # placeholder; mask via shape
        else:
            pose_seq = np.asarray(pose_seq, dtype=np.float32)

        episode = list(zip(obs_seq, action_seq, reward_seq, done_seq, pose_seq))

        with self._lock:
            self._episodes[group_key].append(episode)
            self._total_transitions += T

            # Evita estouro de capacidade — remove episódios antigos do grupo mais cheio
            if self._total_transitions > self.capacity:
                self._evict()

    def add_embedding_sequence(
        self,
        embeddings: List[np.ndarray],
        group_key:  str = "global",
    ) -> None:
        """
        Adiciona sequência de embeddings puros (sem ação/reward).
        Usado pelo handler de MSG_LEADER_SEQUENCE.
        """
        if len(embeddings) < 2:
            return

        obs_seq    = np.stack(embeddings, axis=0)
        action_seq = np.zeros((len(embeddings), self.action_dim), dtype=np.float32)
        reward_seq = np.zeros(len(embeddings), dtype=np.float32)
        done_seq   = np.zeros(len(embeddings), dtype=bool)
        done_seq[-1] = True

        self.add_sequence(obs_seq, action_seq, reward_seq, done_seq, group_key)

    # ──────────────────────────────────────────────────────────────────────────
    # Amostragem
    # ──────────────────────────────────────────────────────────────────────────

    def sample_sequence(
        self,
        batch_size: int,
        seq_len:    Optional[int] = None,
        group_key:  Optional[str] = None,  # None → todos os grupos
    ) -> Optional[dict]:
        """
        Amostra batch_size sequências de comprimento seq_len.

        Returns dict:
            obs:    (B, T, obs_dim)
            action: (B, T, action_dim)
            reward: (B, T)
            done:   (B, T)
            mask:   (B, T) — 1 para passos válidos
        """
        T = seq_len or self.seq_len

        with self._lock:
            if group_key is not None:
                pool = list(self._episodes.get(group_key, []))
            else:
                pool = [ep for eps in self._episodes.values() for ep in eps]

        # Filtra episódios com comprimento suficiente
        valid = [ep for ep in pool if len(ep) >= 1]
        # Treina desde que haja ao menos 1 episódio válido: reamostra com
        # reposição para formar o batch. Isso permite começar a aprender cedo,
        # sem exigir batch_size episódios DISTINTOS (que travava o treino).
        if len(valid) < 1:
            return None

        episodes = random.choices(valid, k=batch_size)
        obs_b, act_b, rew_b, done_b, mask_b, pose_b = [], [], [], [], [], []

        for ep in episodes:
            actual = min(T, len(ep))
            start  = random.randint(0, len(ep) - actual)
            seq    = ep[start: start + actual]

            obs_s    = np.stack([t[0] for t in seq], axis=0)
            act_s    = np.stack([t[1] for t in seq], axis=0)
            rew_s    = np.array([t[2] for t in seq], dtype=np.float32)
            done_s   = np.array([t[3] for t in seq], dtype=np.float32)
            pose_s   = np.stack([t[4] for t in seq], axis=0)
            mask_s   = np.ones(actual, dtype=np.float32)

            # Padding para T
            if actual < T:
                pad = T - actual
                obs_s  = np.pad(obs_s,  ((0, pad), (0, 0)), constant_values=0.0)
                act_s  = np.pad(act_s,  ((0, pad), (0, 0)), constant_values=0.0)
                pose_s = np.pad(pose_s, ((0, pad), (0, 0)), constant_values=0.0)
                rew_s  = np.pad(rew_s,  (0, pad),           constant_values=0.0)
                done_s = np.pad(done_s, (0, pad),           constant_values=1.0)
                mask_s = np.pad(mask_s, (0, pad),           constant_values=0.0)

            obs_b.append(obs_s); act_b.append(act_s)
            rew_b.append(rew_s); done_b.append(done_s); mask_b.append(mask_s)
            pose_b.append(pose_s)

        return {
            "obs":    torch.from_numpy(np.stack(obs_b)).float(),
            "action": torch.from_numpy(np.stack(act_b)).float(),
            "reward": torch.from_numpy(np.stack(rew_b)).float(),
            "done":   torch.from_numpy(np.stack(done_b)).float(),
            "mask":   torch.from_numpy(np.stack(mask_b)).float(),
            "pose":   torch.from_numpy(np.stack(pose_b)).float(),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Utilitários
    # ──────────────────────────────────────────────────────────────────────────

    def ready_sequence(self, batch_size: int) -> bool:
        # Pronto para treinar quando há pelo menos um episódio válido e um
        # mínimo de transições acumuladas (evita treinar com quase nada).
        # Não exige batch_size episódios DISTINTOS — o sample reamostra.
        with self._lock:
            n_episodes = sum(len(eps) for eps in self._episodes.values())
            total = self._total_transitions
            min_transitions = max(16, batch_size // 2)
            return n_episodes >= 1 and total >= min_transitions

    def total_transitions(self) -> int:
        with self._lock:
            return self._total_transitions

    def episode_count(self, group_key: Optional[str] = None) -> int:
        with self._lock:
            if group_key:
                return len(self._episodes.get(group_key, []))
            return sum(len(eps) for eps in self._episodes.values())

    def group_keys(self) -> List[str]:
        with self._lock:
            return list(self._episodes.keys())

    def summary(self) -> dict:
        with self._lock:
            return {
                "total_transitions":  self._total_transitions,
                "total_episodes":     sum(len(eps) for eps in self._episodes.values()),
                "groups":             len(self._episodes),
                "capacity":           self.capacity,
            }

    def _evict(self) -> None:
        """Remove episódios antigos quando capacity é excedida."""
        # Encontra o grupo com mais episódios e remove o mais antigo
        if not self._episodes:
            return
        biggest_key = max(self._episodes, key=lambda k: len(self._episodes[k]))
        if self._episodes[biggest_key]:
            evicted = self._episodes[biggest_key].popleft()
            self._total_transitions -= len(evicted)
