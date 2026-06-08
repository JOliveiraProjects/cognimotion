"""
unified_buffer.py
=================
Replay buffer unificado com Prioritized Experience Replay (PER),
suporte a demonstrações e amostragem de sequências para LSTM/Transformer.

Adaptado de itens.zip — remove dependência de core.logger.
"""
from __future__ import annotations

import logging
import random
import threading
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Sum-Tree (estrutura de dados para PER)
# ──────────────────────────────────────────────────────────────────────────────

class SumTree:
    """Árvore de soma para amostragem proporcional em O(log n)."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self._write = 0
        self.n_entries = 0

    def _propagate(self, idx: int, delta: float) -> None:
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent:
            self._propagate(parent, delta)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        if left >= len(self.tree):
            return idx
        return (
            self._retrieve(left, s)
            if s <= self.tree[left]
            else self._retrieve(left + 1, s - self.tree[left])
        )

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, data) -> int:
        idx = self._write + self.capacity - 1
        self.data[self._write] = data
        self.update(idx, priority)
        self._write = (self._write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
        return idx

    def update(self, idx: int, priority: float) -> None:
        self._propagate(idx, priority - self.tree[idx])
        self.tree[idx] = priority

    def get(self, s: float) -> Tuple[int, float, object]:
        idx = self._retrieve(0, min(s, self.total))
        data_idx = idx - self.capacity + 1
        return idx, float(self.tree[idx]), self.data[data_idx]

    def __len__(self) -> int:
        return self.n_entries


# ──────────────────────────────────────────────────────────────────────────────
# UnifiedReplayBuffer
# ──────────────────────────────────────────────────────────────────────────────

class UnifiedReplayBuffer:
    """
    Buffer de replay unificado com PER, suporte a demos e sequências.

    Transition tuple: (obs, action, reward, done, next_obs, is_demo)
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        epsilon: float = 1e-6,
        total_steps: int = 1_000_000,
        use_per: bool = True,
        action_dim: int = 9,
        demo_priority_boost: float = 2.0,
    ) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.epsilon = epsilon
        self.total_steps = total_steps
        self.use_per = use_per
        self.action_dim = action_dim
        self.demo_priority_boost = demo_priority_boost

        self._tree = SumTree(capacity)
        self._max_priority = 1.0
        self._step = 0
        self._lock = threading.RLock()

        self._episodes: List[list] = []
        self._current_ep: list = []
        self._ep_max = max(capacity // 100, 1)

        self._demo_count = 0

        logger.info(
            f"UnifiedReplayBuffer | capacity={capacity} "
            f"| PER={'on' if use_per else 'off'} | action_dim={action_dim}"
        )

    @property
    def beta(self) -> float:
        progress = min(self._step / max(self.total_steps, 1), 1.0)
        return self.beta_start + progress * (self.beta_end - self.beta_start)

    # ──────────────────────────────────────────────────────────────────────────
    # Inserção
    # ──────────────────────────────────────────────────────────────────────────

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        done: bool,
        next_obs: torch.Tensor,
        priority: Optional[float] = None,
        is_demo: bool = False,
    ) -> None:
        # Converte action para one-hot se escalar
        if isinstance(action, torch.Tensor):
            if action.dim() == 0 or (action.dim() == 1 and action.size(0) == 1):
                action = F.one_hot(action.long().squeeze(), self.action_dim).float()
            action = action.detach().cpu()
        obs = obs.detach().cpu() if isinstance(obs, torch.Tensor) else obs
        next_obs = next_obs.detach().cpu() if isinstance(next_obs, torch.Tensor) else next_obs

        transition = (obs, action, float(reward), bool(done), next_obs, is_demo)
        p = (priority if priority is not None else self._max_priority) ** self.alpha
        if is_demo:
            p *= self.demo_priority_boost
            self._demo_count += 1

        with self._lock:
            self._tree.add(p, transition)
            self._current_ep.append(transition)
            if done:
                self._episodes.append(list(self._current_ep))
                self._current_ep = []
                if len(self._episodes) > self._ep_max:
                    self._episodes.pop(0)

        self._step += 1

    # ──────────────────────────────────────────────────────────────────────────
    # Atualização de prioridades
    # ──────────────────────────────────────────────────────────────────────────

    def update_priorities(self, indices: np.ndarray, errors: np.ndarray) -> None:
        with self._lock:
            for idx, err in zip(indices, errors):
                p = (float(abs(err)) + self.epsilon) ** self.alpha
                data_idx = idx - self._tree.capacity + 1
                if (
                    0 <= data_idx < len(self._tree.data)
                    and self._tree.data[data_idx] is not None
                    and self._tree.data[data_idx][5]  # is_demo
                ):
                    p *= self.demo_priority_boost
                self._tree.update(int(idx), p)
                self._max_priority = max(self._max_priority, p)

    # ──────────────────────────────────────────────────────────────────────────
    # Amostragem
    # ──────────────────────────────────────────────────────────────────────────

    def sample(self, batch_size: int, demo_ratio: float = 0.2) -> Optional[dict]:
        with self._lock:
            if len(self._tree) < batch_size:
                return None
            total = self._tree.total
            if total <= 0:
                return None

            n_demo = int(batch_size * demo_ratio) if self._demo_count > 0 else 0
            n_normal = batch_size - n_demo

            indices, priorities, transitions = [], [], []

            # Amostras de demonstração
            if n_demo > 0:
                demo_transitions = [
                    t for t in self._tree.data
                    if t is not None and t[5]
                ]
                if demo_transitions:
                    for t in random.choices(demo_transitions, k=n_demo):
                        for i, d in enumerate(self._tree.data):
                            if d is t:
                                tree_idx = i + self._tree.capacity - 1
                                p = self._tree.tree[tree_idx]
                                indices.append(tree_idx)
                                priorities.append(max(p, self.epsilon))
                                transitions.append(t)
                                break

            # Amostras normais por segmento
            segment = total / max(n_normal, 1)
            for i in range(n_normal):
                s = random.uniform(segment * i, segment * (i + 1))
                idx, p, data = self._tree.get(s)
                if data is None:
                    continue
                indices.append(idx)
                priorities.append(max(p, self.epsilon))
                transitions.append(data)

        if len(transitions) < batch_size:
            return None

        probs = np.array(priorities) / max(self._tree.total, 1e-8)
        weights = (max(len(self._tree), 1) * probs) ** (-self.beta)
        weights /= weights.max()

        obs = torch.stack([t[0].squeeze(0) if t[0].dim() > 3 else t[0] for t in transitions])
        actions = torch.stack([t[1] for t in transitions])
        rewards = torch.tensor([t[2] for t in transitions], dtype=torch.float32)
        dones = torch.tensor([t[3] for t in transitions], dtype=torch.float32)
        next_obs = torch.stack([t[4].squeeze(0) if t[4].dim() > 3 else t[4] for t in transitions])
        is_demo = torch.tensor([t[5] for t in transitions], dtype=torch.bool)

        return {
            "obs": obs,
            "action": actions,
            "reward": rewards,
            "done": dones,
            "next_obs": next_obs,
            "is_weights": torch.tensor(weights, dtype=torch.float32),
            "indices": np.array(indices),
            "is_demo": is_demo,
        }

    def sample_sequence(self, batch_size: int, seq_len: int) -> Optional[dict]:
        """Amostra sequências de episódios (para treinamento recorrente)."""
        with self._lock:
            available = [ep for ep in self._episodes if len(ep) >= 1]
        if len(available) < batch_size:
            return None

        episodes = random.sample(available, batch_size)
        sequences, masks = [], []
        for ep in episodes:
            actual = min(seq_len, len(ep))
            start = random.randint(0, len(ep) - actual)
            seq = list(ep[start: start + actual])
            mask = [1] * actual + [0] * (seq_len - actual)
            if actual < seq_len:
                pad = (
                    torch.zeros_like(seq[0][0]),
                    torch.zeros_like(seq[0][1]),
                    0.0, 1.0,
                    torch.zeros_like(seq[0][4]),
                    False,
                )
                seq += [pad] * (seq_len - actual)
            sequences.append(seq)
            masks.append(mask)

        return {
            "obs": torch.stack([torch.stack([t[0] for t in s]) for s in sequences]),
            "action": torch.stack([torch.stack([t[1] for t in s]) for s in sequences]),
            "reward": torch.stack([
                torch.tensor([t[2] for t in s], dtype=torch.float32) for s in sequences
            ]),
            "done": torch.stack([
                torch.tensor([t[3] for t in s], dtype=torch.float32) for s in sequences
            ]),
            "next_obs": torch.stack([torch.stack([t[4] for t in s]) for s in sequences]),
            "mask": torch.tensor(masks, dtype=torch.float32),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Utilitários
    # ──────────────────────────────────────────────────────────────────────────

    def ready(self, batch_size: int) -> bool:
        with self._lock:
            return len(self._tree) >= batch_size

    def ready_sequence(self, batch_size: int) -> bool:
        with self._lock:
            return len(self._episodes) >= batch_size

    def __len__(self) -> int:
        with self._lock:
            return len(self._tree)

    @property
    def demo_count(self) -> int:
        return self._demo_count

    @property
    def episodes(self) -> List[list]:
        with self._lock:
            return list(self._episodes)
