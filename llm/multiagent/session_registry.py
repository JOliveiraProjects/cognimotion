"""
session_registry.py
===================
Registro de sessões multiplayer.

Cada instância do jogo UE5 que conecta ao servidor recebe uma sessão.
Cada sessão pode ter múltiplos NPCs (identificados por session_id + seq_id).
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class NPCSession:
    """Representa a sessão de um único cliente UE5 conectado."""
    session_id: str
    peer_addr: str
    connected_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)

    # Estatísticas de throughput
    requests_received: int = 0
    responses_sent: int = 0
    total_latency_ms: float = 0.0

    # Worker process index que atende esta sessão
    worker_index: int = 0

    # Estado LLM: último motion_style sugerido pelo LLM
    llm_motion_style: int = 0
    llm_last_update: float = 0.0

    # Métricas de qualidade
    mean_confidence: float = 0.0
    _confidence_accum: float = 0.0
    _confidence_count: int = 0

    @property
    def avg_latency_ms(self) -> float:
        if self.responses_sent == 0:
            return 0.0
        return self.total_latency_ms / self.responses_sent

    @property
    def uptime_s(self) -> float:
        return time.time() - self.connected_at

    def record_response(self, latency_ms: float, confidence: float) -> None:
        self.responses_sent += 1
        self.total_latency_ms += latency_ms
        self._confidence_accum += confidence
        self._confidence_count += 1
        self.mean_confidence = self._confidence_accum / self._confidence_count
        self.last_heartbeat = time.time()

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "peer_addr": self.peer_addr,
            "uptime_s": round(self.uptime_s, 1),
            "requests_received": self.requests_received,
            "responses_sent": self.responses_sent,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "mean_confidence": round(self.mean_confidence, 3),
            "worker_index": self.worker_index,
            "llm_motion_style": self.llm_motion_style,
        }


class SessionRegistry:
    """
    Registro global de sessões multiplayer (thread-safe).

    O servidor cria uma sessão ao aceitar conexão e a remove ao desconectar.
    """

    def __init__(self, max_sessions: int = 32) -> None:
        self.max_sessions = max_sessions
        self._sessions: Dict[str, NPCSession] = {}
        self._lock = threading.RLock()
        self._total_created = 0
        self._worker_counter = 0

        logger.info(f"SessionRegistry | max_sessions={max_sessions}")

    # ──────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────────

    def create_session(self, session_id: str, peer_addr: str) -> Optional[NPCSession]:
        with self._lock:
            if len(self._sessions) >= self.max_sessions:
                logger.warning(
                    f"SessionRegistry: limite atingido ({self.max_sessions}), "
                    f"rejeitando {peer_addr}"
                )
                return None

            # Round-robin de workers
            worker_idx = self._worker_counter % max(self.max_sessions, 1)
            self._worker_counter += 1

            session = NPCSession(
                session_id=session_id,
                peer_addr=peer_addr,
                worker_index=worker_idx,
            )
            self._sessions[session_id] = session
            self._total_created += 1

            logger.info(
                f"SessionRegistry: nova sessão {session_id} de {peer_addr} "
                f"→ worker={worker_idx} | total={len(self._sessions)}"
            )
            return session

    def remove_session(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
            if session:
                logger.info(
                    f"SessionRegistry: sessão {session_id} encerrada "
                    f"| uptime={session.uptime_s:.1f}s "
                    f"| req={session.requests_received}"
                )

    def get_session(self, session_id: str) -> Optional[NPCSession]:
        with self._lock:
            return self._sessions.get(session_id)

    # ──────────────────────────────────────────────────────────────────────────
    # Consultas
    # ──────────────────────────────────────────────────────────────────────────

    def all_sessions(self) -> List[NPCSession]:
        with self._lock:
            return list(self._sessions.values())

    def session_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def update_llm_style(self, session_id: str, motion_style: int) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.llm_motion_style = motion_style
                session.llm_last_update = time.time()

    def heartbeat(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.last_heartbeat = time.time()

    def remove_stale_sessions(self, timeout_s: float = 30.0) -> List[str]:
        """Remove sessões sem heartbeat por mais de timeout_s segundos."""
        now = time.time()
        stale = []
        with self._lock:
            for sid, session in list(self._sessions.items()):
                if now - session.last_heartbeat > timeout_s:
                    stale.append(sid)
            for sid in stale:
                self._sessions.pop(sid, None)
                logger.warning(f"SessionRegistry: sessão {sid} removida por timeout")
        return stale

    def summary(self) -> dict:
        with self._lock:
            sessions = list(self._sessions.values())
        if not sessions:
            return {
                "active_sessions": 0,
                "total_created": self._total_created,
            }
        return {
            "active_sessions": len(sessions),
            "total_created": self._total_created,
            "total_requests": sum(s.requests_received for s in sessions),
            "avg_latency_ms": round(
                sum(s.avg_latency_ms for s in sessions) / len(sessions), 2
            ),
            "sessions": [s.to_dict() for s in sessions],
        }
