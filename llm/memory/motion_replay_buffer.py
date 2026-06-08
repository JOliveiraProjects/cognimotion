from __future__ import annotations
import threading
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from config import MemoryConfig


@dataclass
class ReplayEntry:
    embedding: np.ndarray
    style: int
    movement_mode: int
    speed: float
    confidence: float
    timestamp: float
    priority: float
    entry_id: int


class MotionReplayBuffer:
    def __init__(self, config: MemoryConfig, embedding_dim: int = 256):
        self.config       = config
        self.embedding_dim = embedding_dim
        self._entries: List[ReplayEntry] = []
        self._lock      = threading.RLock()
        self._next_id   = 0
        self._total_added = 0

    def _compute_priority(
        self,
        confidence: float,
        timestamp: float,
        embedding: np.ndarray,
    ) -> float:
        recency = 1.0 / (1.0 + (time.time() - timestamp))
        conf_score = float(np.clip(confidence, 0.0, 1.0))
        diversity = 1.0
        if len(self._entries) > 0:
            sample_size = min(32, len(self._entries))
            idxs = np.random.choice(len(self._entries), size=sample_size, replace=False)
            similarities = [
                float(np.dot(embedding, self._entries[i].embedding) /
                      (np.linalg.norm(embedding) * np.linalg.norm(self._entries[i].embedding) + 1e-8))
                for i in idxs
            ]
            diversity = 1.0 - float(np.mean(similarities))

        return (self.config.confidence_weight * conf_score +
                self.config.recency_weight * recency +
                self.config.diversity_weight * diversity)

    def add(
        self,
        embedding: np.ndarray,
        style: int = 0,
        movement_mode: int = 0,
        speed: float = 0.0,
        confidence: float = 1.0,
    ) -> int:
        with self._lock:
            ts = time.time()
            emb = embedding.astype(np.float32).copy()
            priority = self._compute_priority(confidence, ts, emb)

            entry = ReplayEntry(
                embedding=emb,
                style=style,
                movement_mode=movement_mode,
                speed=speed,
                confidence=confidence,
                timestamp=ts,
                priority=priority,
                entry_id=self._next_id,
            )
            self._next_id += 1
            self._total_added += 1

            if len(self._entries) >= self.config.replay_capacity:
                min_idx = min(range(len(self._entries)),
                              key=lambda i: self._entries[i].priority)
                self._entries[min_idx] = entry
            else:
                self._entries.append(entry)

            return entry.entry_id

    def sample(self, batch_size: int) -> Optional[List[ReplayEntry]]:
        with self._lock:
            if len(self._entries) < batch_size:
                return None

            priorities = np.array([e.priority for e in self._entries], dtype=np.float64)
            priorities = np.power(priorities, self.config.priority_alpha)
            probs = priorities / priorities.sum()

            probs = probs / probs.sum()
            idxs = np.random.choice(len(self._entries), size=batch_size, replace=False, p=probs)
            return [self._entries[i] for i in idxs]

    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    def get_stats(self) -> dict:
        with self._lock:
            if not self._entries:
                return {"size": 0, "total_added": self._total_added}
            priorities = [e.priority for e in self._entries]
            confidences = [e.confidence for e in self._entries]
            return {
                "size": len(self._entries),
                "total_added": self._total_added,
                "mean_priority": float(np.mean(priorities)),
                "mean_confidence": float(np.mean(confidences)),
                "capacity_pct": len(self._entries) / self.config.replay_capacity,
            }
