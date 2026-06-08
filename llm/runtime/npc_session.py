"""
runtime/npc_session.py
=======================
NPCSessionManager — mantém estado recorrente (h, z) por NPC individual.

Cada NPC identificado por int64 NPCId tem seu próprio:
  - h: estado determinístico (GRU hidden)
  - z: estado estocástico (categorical one-hot)
  - last_seen: timestamp para LRU eviction

Thread-safe (chamado de asyncio via run_in_executor).
"""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class NPCSession:
    """Estado recorrente + metadados de um único NPC."""

    __slots__ = (
        "npc_id", "h", "z",
        "last_seen", "step",
        "last_action", "last_reward",
        "style_vector", "leader_npc_id",
        "last_combined",
        "reward_sum", "reward_n",
    )

    def __init__(
        self,
        npc_id:        int,
        hidden_dim:    int,
        stochastic_dim: int,
        device:        str = "cpu",
    ) -> None:
        self.npc_id        = npc_id
        self.h             = torch.zeros(1, hidden_dim,     device=device)
        self.z             = torch.zeros(1, stochastic_dim, device=device)
        self.last_seen:     float = time.time()
        self.step:          int   = 0
        self.last_action:   int   = 0
        self.last_reward:   float = 0.0
        self.style_vector:  Optional[torch.Tensor] = None
        self.leader_npc_id: int   = 0
        self.last_combined = None  # latente [z,h] p/ PoseDecoder gerar animação
        self.reward_sum:   float = 0.0
        self.reward_n:     int   = 0

    def record_reward(self, reward: float) -> None:
        """Registra a recompensa recebida neste passo e acumula soma/contagem
        para o cálculo de mean_reward desta sessão. Observação: estes valores
        são locais à NPCSession; o PolicyRegistry recebe seu reward de outra
        fonte (o retorno imaginado do trainer em dreamer_trainer.py)."""
        self.last_reward = float(reward)
        self.reward_sum += float(reward)
        self.reward_n   += 1

    @property
    def mean_reward(self) -> float:
        return self.reward_sum / self.reward_n if self.reward_n > 0 else 0.0

    def touch(self) -> None:
        self.last_seen = time.time()

    def update_state(self, h: torch.Tensor, z: torch.Tensor) -> None:
        self.h = h.detach()
        self.z = z.detach()
        self.step += 1
        self.touch()

    def reset_state(self, hidden_dim: int, stochastic_dim: int) -> None:
        device = self.h.device
        self.h = torch.zeros(1, hidden_dim,     device=device)
        self.z = torch.zeros(1, stochastic_dim, device=device)
        self.step = 0

    def age_seconds(self) -> float:
        return time.time() - self.last_seen

    def to_dict(self) -> dict:
        return {
            "npc_id":        self.npc_id,
            "step":          self.step,
            "last_action":   self.last_action,
            "age_s":         round(self.age_seconds(), 1),
            "leader_npc_id": self.leader_npc_id,
        }


class NPCSessionManager:
    """
    Gerencia NPCSessions com LRU eviction.

    Thread-safe. Usado pelo MotionInferenceService para manter estado
    per-NPC entre requests de inferência autônoma.
    """

    def __init__(
        self,
        max_sessions:   int = 256,
        timeout_s:      float = 120.0,
        hidden_dim:     int = 512,
        stochastic_dim: int = 1024,
        device:         str = "cpu",
    ) -> None:
        self.max_sessions   = max_sessions
        self.timeout_s      = timeout_s
        self.hidden_dim     = hidden_dim
        self.stochastic_dim = stochastic_dim
        self.device         = device

        # OrderedDict mantém ordem de inserção para LRU simples
        self._sessions: OrderedDict[int, NPCSession] = OrderedDict()
        self._lock = threading.RLock()

        logger.info(
            f"NPCSessionManager | max={max_sessions} | timeout={timeout_s}s "
            f"| h={hidden_dim} | z={stochastic_dim} | device={device}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Get / Create
    # ──────────────────────────────────────────────────────────────────────────

    def get_or_create(self, npc_id: int) -> NPCSession:
        """
        Retorna NPCSession existente ou cria uma nova.
        Move para o final do OrderedDict (LRU).
        """
        with self._lock:
            if npc_id in self._sessions:
                session = self._sessions.pop(npc_id)
                session.touch()
                self._sessions[npc_id] = session   # move to end (recently used)
                return session

            # Cria nova sessão
            session = NPCSession(
                npc_id=npc_id,
                hidden_dim=self.hidden_dim,
                stochastic_dim=self.stochastic_dim,
                device=self.device,
            )
            self._sessions[npc_id] = session

            # Evicção LRU se necessário
            while len(self._sessions) > self.max_sessions:
                evicted_id, _ = self._sessions.popitem(last=False)
                logger.debug(f"NPCSessionManager | evicted NPC {evicted_id} (LRU)")

            return session

    def get(self, npc_id: int) -> Optional[NPCSession]:
        with self._lock:
            return self._sessions.get(npc_id)

    def remove(self, npc_id: int) -> None:
        with self._lock:
            self._sessions.pop(npc_id, None)

    # ──────────────────────────────────────────────────────────────────────────
    # Batch update (chamado pelo RSSM após inferência)
    # ──────────────────────────────────────────────────────────────────────────

    def update_state(
        self,
        npc_id: int,
        h:      torch.Tensor,
        z:      torch.Tensor,
        action: int,
        reward: float = 0.0,
    ) -> None:
        session = self.get_or_create(npc_id)
        session.update_state(h, z)
        session.last_action = action
        session.last_reward = reward

    def get_state(
        self, npc_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retorna (h, z) do NPC. Cria zeros se não existir.
        Não cria sessão persistente — use get_or_create para isso.
        """
        with self._lock:
            session = self._sessions.get(npc_id)
        if session is None:
            h = torch.zeros(1, self.hidden_dim,     device=self.device)
            z = torch.zeros(1, self.stochastic_dim, device=self.device)
            return h, z
        return session.h, session.z

    # ──────────────────────────────────────────────────────────────────────────
    # Manutenção
    # ──────────────────────────────────────────────────────────────────────────

    def remove_stale(self) -> int:
        """Remove sessões inativas por mais de timeout_s. Retorna N removidos."""
        now   = time.time()
        stale = []
        with self._lock:
            for npc_id, session in list(self._sessions.items()):
                if (now - session.last_seen) > self.timeout_s:
                    stale.append(npc_id)
            for npc_id in stale:
                del self._sessions[npc_id]

        if stale:
            logger.info(f"NPCSessionManager | removidas {len(stale)} sessões inativas")
        return len(stale)

    def session_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def all_sessions(self) -> Dict[int, dict]:
        with self._lock:
            return {nid: s.to_dict() for nid, s in self._sessions.items()}

    def summary(self) -> dict:
        with self._lock:
            sessions = list(self._sessions.values())

        return {
            "active_npcs":  len(sessions),
            "max_sessions": self.max_sessions,
            "timeout_s":    self.timeout_s,
            "sessions":     [s.to_dict() for s in sessions],
        }
