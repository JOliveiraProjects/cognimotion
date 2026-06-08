"""
vector_store.py
===============
Wrapper de busca vetorial — usa FAISS quando disponível, fallback NumPy O(n).
Copiado de itens.zip (sem dependências quebradas — reutilizado diretamente).
"""
from __future__ import annotations

import logging

import numpy as np
import torch

try:
    import faiss
    _HAS_FAISS = True
except (ImportError, OSError, Exception):
    _HAS_FAISS = False

logger = logging.getLogger(__name__)


class _NumpyVectorStore:
    """Implementação NumPy O(n) — fallback quando FAISS não está disponível."""

    def __init__(self, dim: int, use_cosine: bool = True) -> None:
        self.dim = dim
        self.use_cosine = use_cosine
        self._matrix: np.ndarray = np.empty((0, dim), dtype=np.float32)
        self.vectors: list = []
        self.ids: list = []
        self.id_to_idx: dict = {}

    def add(self, vector: torch.Tensor, vector_id: int) -> None:
        idx = len(self.vectors)
        vec_np = vector.detach().cpu().numpy().astype("float32").reshape(1, -1)
        if self.use_cosine:
            norm = np.linalg.norm(vec_np, axis=1, keepdims=True)
            vec_np = vec_np / (norm + 1e-8)
        self._matrix = np.vstack([self._matrix, vec_np]) if self._matrix.size else vec_np
        self.vectors.append(vector.detach().cpu())
        self.ids.append(vector_id)
        self.id_to_idx[vector_id] = idx

    def search(self, query: torch.Tensor, k: int):
        if not self.vectors:
            return np.zeros((1, 1)), np.zeros((1, 1), dtype=int)
        if query.dim() == 1:
            query = query.unsqueeze(0)
        q_np = query.detach().cpu().numpy().astype("float32")
        if self.use_cosine:
            norm = np.linalg.norm(q_np, axis=1, keepdims=True)
            q_np = q_np / (norm + 1e-8)
            distances = 1.0 - (q_np @ self._matrix.T)
        else:
            diff = q_np[:, None, :] - self._matrix[None, :, :]
            distances = (diff ** 2).sum(-1)
        k = min(k, len(self.vectors))
        indices = np.argsort(distances, axis=1)[:, :k]
        distances = np.take_along_axis(distances, indices, axis=1)
        return distances, indices

    def get_vector(self, idx: int) -> torch.Tensor:
        return self.vectors[idx]

    def get_id(self, idx: int) -> int:
        return self.ids[idx]

    def __len__(self) -> int:
        return len(self.vectors)


class _FaissVectorStore:
    """Implementação FAISS — O(log n) para busca vetorial."""

    def __init__(self, dim: int, use_cosine: bool = True) -> None:
        self.dim = dim
        self.use_cosine = use_cosine
        self.index = faiss.IndexFlatIP(dim) if use_cosine else faiss.IndexFlatL2(dim)
        self.vectors: list = []
        self.ids: list = []
        self.id_to_idx: dict = {}

    def add(self, vector: torch.Tensor, vector_id: int) -> None:
        idx = len(self.vectors)
        vec_np = vector.detach().cpu().numpy().astype("float32").reshape(1, -1)
        self.index.add(vec_np)
        self.vectors.append(vector.detach().cpu())
        self.ids.append(vector_id)
        self.id_to_idx[vector_id] = idx

    def search(self, query: torch.Tensor, k: int):
        if query.dim() == 1:
            query = query.unsqueeze(0)
        q_np = query.detach().cpu().numpy().astype("float32")
        k = min(k, len(self.vectors))
        if k == 0:
            return np.zeros((q_np.shape[0], 1)), np.zeros((q_np.shape[0], 1), dtype=int)
        distances, indices = self.index.search(q_np, k)
        if self.use_cosine:
            distances = 1.0 - distances
        return distances, indices

    def get_vector(self, idx: int) -> torch.Tensor:
        return self.vectors[idx]

    def get_id(self, idx: int) -> int:
        return self.ids[idx]

    def __len__(self) -> int:
        return len(self.vectors)


def VectorStore(dim: int, use_cosine: bool = True):
    """Factory — retorna FAISS se disponível, caso contrário NumPy."""
    if _HAS_FAISS:
        return _FaissVectorStore(dim, use_cosine)
    logger.warning(
        "faiss não instalado — usando NumpyVectorStore O(n). "
        "Para produção: pip install faiss-cpu"
    )
    return _NumpyVectorStore(dim, use_cosine)
