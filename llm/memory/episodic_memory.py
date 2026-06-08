"""
episodic_memory.py
==================
Memória episódica por NPC — armazena sequências de embeddings comprimidos.

Adaptado de itens.zip:
  - Remove dependências de brain/rssm (inexistentes no projeto)
  - Usa embedding_dim do projeto (256) ao invés de stochastic_dim
  - Integra com VectorStore do projeto
  - Remove core.logger, usa logging padrão
"""
from __future__ import annotations

import logging
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from memory.vector_store import VectorStore

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers internos
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _compress_episode(
    episode: list,
    embedding_dim: int,
) -> Optional[torch.Tensor]:
    """
    Comprime uma sequência de transições num embedding médio ponderado por reward.
    Cada transição é: (embedding: np.ndarray, reward: float, done: bool)
    """
    z_list: List[torch.Tensor] = []
    w_list: List[float] = []

    for trans in episode:
        emb = trans[0]   # np.ndarray shape (embedding_dim,)
        r   = float(trans[1]) if len(trans) > 1 else 0.0

        if emb is None:
            continue
        if isinstance(emb, np.ndarray):
            emb = torch.from_numpy(emb.astype(np.float32))
        z_t = emb.squeeze()
        if z_t.shape[0] != embedding_dim:
            continue

        w = max(abs(r), 0.1)
        z_list.append(z_t.cpu())
        w_list.append(w)

    if not z_list:
        return None

    weights = torch.tensor(w_list, dtype=torch.float32)
    weights = weights / weights.sum()
    stacked = torch.stack(z_list)
    emb_mean = (stacked * weights.unsqueeze(1)).sum(0)
    emb_mean = emb_mean / (emb_mean.norm() + 1e-8)
    return emb_mean


def _extract_keyframes(episode: list) -> list:
    """Retorna as transições mais informativas do episódio."""
    keyframes = []
    for i, trans in enumerate(episode):
        r    = float(trans[1]) if len(trans) > 1 else 0.0
        done = bool(trans[2]) if len(trans) > 2 else False
        is_key = abs(r) > 1e-4 or done or i == 0 or i == len(episode) - 1
        if is_key:
            keyframes.append(trans)
    return keyframes or (episode[:1] if episode else [])


# ──────────────────────────────────────────────────────────────────────────────
# EpisodicMemory
# ──────────────────────────────────────────────────────────────────────────────

class EpisodicMemory:
    """
    Memória episódica por NPC.

    Armazena episódios comprimidos como embeddings vetoriais e permite
    recuperação por similaridade coseno via VectorStore (FAISS/NumPy).

    Cada episódio é indexado por um embedding comprimido.
    Consultas retornam os episódios mais similares ao embedding de query.
    """

    def __init__(
        self,
        max_episodes: int = 200,
        embedding_dim: int = 256,
        max_keyframes_per_episode: int = 64,
        similarity_threshold: float = 0.75,
    ) -> None:
        self.max_episodes = max_episodes
        self.embedding_dim = embedding_dim
        self.max_keyframes = max_keyframes_per_episode
        self.similarity_threshold = similarity_threshold

        self._store = VectorStore(embedding_dim, use_cosine=True)
        self._episodes: deque = deque(maxlen=max_episodes)
        self._episode_meta: List[Dict] = []
        self._episode_id = 0

        logger.info(
            f"EpisodicMemory | max_episodes={max_episodes} "
            f"| embedding_dim={embedding_dim}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Armazenamento
    # ──────────────────────────────────────────────────────────────────────────

    def store_episode(
        self,
        episode: list,
        session_id: str = "",
        motion_style: int = 0,
        mean_reward: Optional[float] = None,
    ) -> bool:
        """
        Comprime e armazena um episódio.
        episode: lista de transições (embedding, reward, done, ...)
        Retorna True se foi armazenado.
        """
        if not episode:
            return False

        compressed = _compress_episode(episode, self.embedding_dim)
        if compressed is None:
            return False

        keyframes = _extract_keyframes(episode)[:self.max_keyframes]
        r = mean_reward if mean_reward is not None else np.mean([t[1] for t in episode if len(t) > 1])

        eid = self._episode_id
        self._episode_id += 1

        self._store.add(compressed, eid)
        self._episodes.append(keyframes)

        meta = {
            "episode_id": eid,
            "session_id": session_id,
            "motion_style": motion_style,
            "mean_reward": float(r),
            "n_steps": len(episode),
            "stored_at": time.time(),
        }
        # Mantém meta sincronizado com deque (remove o mais antigo se cheio)
        if len(self._episode_meta) >= self.max_episodes:
            self._episode_meta.pop(0)
        self._episode_meta.append(meta)

        return True

    # ──────────────────────────────────────────────────────────────────────────
    # Recuperação
    # ──────────────────────────────────────────────────────────────────────────

    def retrieve_similar(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> List[Tuple[list, Dict, float]]:
        """
        Recupera os k episódios mais similares ao embedding de query.
        Retorna lista de (keyframes, meta, similarity_score).
        """
        if len(self._store) == 0:
            return []

        q_tensor = torch.from_numpy(query_embedding.astype(np.float32))
        distances, indices = self._store.search(q_tensor, k)

        results = []
        for dist_row, idx_row in zip(distances, indices):
            for dist, idx in zip(dist_row, idx_row):
                if idx < 0 or idx >= len(self._episodes):
                    continue
                similarity = 1.0 - float(dist)
                if similarity < self.similarity_threshold:
                    continue
                keyframes = list(self._episodes)[idx]
                meta = self._episode_meta[idx] if idx < len(self._episode_meta) else {}
                results.append((keyframes, meta, similarity))

        return sorted(results, key=lambda x: x[2], reverse=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Utilitários
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def episode_count(self) -> int:
        return len(self._episodes)

    def summary(self) -> Dict:
        if not self._episode_meta:
            return {"episode_count": 0}
        rewards = [m["mean_reward"] for m in self._episode_meta]
        return {
            "episode_count": len(self._episodes),
            "mean_reward": float(np.mean(rewards)),
            "best_reward": float(np.max(rewards)),
            "worst_reward": float(np.min(rewards)),
        }

    def clear(self) -> None:
        self._store = VectorStore(self.embedding_dim, use_cosine=True)
        self._episodes.clear()
        self._episode_meta.clear()
        self._episode_id = 0
