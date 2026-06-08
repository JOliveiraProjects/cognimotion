from __future__ import annotations
import threading
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from config import MemoryConfig


@dataclass
class MotionMemoryEntry:
    embedding: np.ndarray
    style: int
    movement_mode: int
    speed: float
    timestamp: float
    confidence: float
    entry_id: int


class MotionMemoryBank:
    def __init__(self, config: MemoryConfig, embedding_dim: int = 256):
        self.config       = config
        self.embedding_dim = embedding_dim
        self._entries: List[MotionMemoryEntry] = []
        self._embeddings_matrix: Optional[np.ndarray] = None
        self._lock        = threading.RLock()
        self._next_id     = 0
        self._index_dirty = True

        try:
            import faiss
            if config.faiss_index_type == "IVF":
                quantizer = faiss.IndexFlatL2(embedding_dim)
                self._faiss_index = faiss.IndexIVFFlat(
                    quantizer, embedding_dim, 64)
                self._faiss_trained = False
            else:
                self._faiss_index = faiss.IndexFlatIP(embedding_dim)
                self._faiss_trained = True
            self._use_faiss = True
        except ImportError:
            self._use_faiss = False
            self._faiss_index = None
            self._faiss_trained = False

    def add(
        self,
        embedding: np.ndarray,
        style: int = 0,
        movement_mode: int = 0,
        speed: float = 0.0,
        confidence: float = 1.0,
    ) -> int:
        with self._lock:
            entry = MotionMemoryEntry(
                embedding=embedding.astype(np.float32).copy(),
                style=style,
                movement_mode=movement_mode,
                speed=speed,
                timestamp=time.time(),
                confidence=confidence,
                entry_id=self._next_id,
            )
            self._next_id += 1

            if len(self._entries) >= self.config.bank_capacity:
                self._entries.pop(0)

            self._entries.append(entry)
            self._index_dirty = True

            if len(self._entries) == self.config.bank_capacity:
                self._rebuild_index()

            return entry.entry_id

    def _rebuild_index(self) -> None:
        if not self._entries:
            return

        mat = np.stack([e.embedding for e in self._entries], axis=0)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        self._embeddings_matrix = mat / norms

        if self._use_faiss and self._faiss_index is not None:
            import faiss
            self._faiss_index.reset()
            if not self._faiss_trained:
                n_clusters = getattr(self._faiss_index, 'nlist', 64)
                min_train  = max(n_clusters * 4, 256)
                if self._embeddings_matrix.shape[0] >= min_train:
                    self._faiss_index.train(self._embeddings_matrix)
                    self._faiss_trained = True
            if self._faiss_trained:
                self._faiss_index.reset()
                self._faiss_index.add(self._embeddings_matrix)

        self._index_dirty = False

    def search(
        self,
        query: np.ndarray,
        k: int = 8,
        style_filter: Optional[int] = None,
    ) -> List[Tuple[MotionMemoryEntry, float]]:
        with self._lock:
            if not self._entries:
                return []

            if self._index_dirty:
                self._rebuild_index()

            q = query.astype(np.float32)
            q_norm = np.linalg.norm(q)
            if q_norm > 1e-8:
                q = q / q_norm

            if self._use_faiss and self._faiss_index is not None and self._faiss_trained:
                import faiss
                D, I = self._faiss_index.search(q.reshape(1, -1), min(k * 2, len(self._entries)))
                candidates = [(self._entries[i], float(d)) for d, i in zip(D[0], I[0]) if i >= 0]
            else:
                if self._embeddings_matrix is None:
                    self._rebuild_index()
                scores = self._embeddings_matrix @ q
                top_k_idx = np.argsort(scores)[::-1][:k * 2]
                candidates = [(self._entries[i], float(scores[i])) for i in top_k_idx]

            if style_filter is not None:
                candidates = [(e, s) for e, s in candidates if e.style == style_filter]

            return candidates[:k]

    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._embeddings_matrix = None
            self._index_dirty = True
            if self._use_faiss and self._faiss_index is not None:
                self._faiss_index.reset()
