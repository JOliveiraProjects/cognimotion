"""
Log de pipeline transparente — mostra, em tempo real e de forma legível, o que
o servidor está fazendo a cada passo. Objetivo: nunca mais ficar na dúvida sobre
"o que está acontecendo" ou "por que o treino não anda".

Emite um painel periódico com o estado COMPLETO do servidor:
  - conexões ativas (quantos NPCs/líderes)
  - frames recebidos do Unreal
  - respostas enviadas de volta
  - estado do buffer de treino (transições, episódios)
  - se o treino está rodando e por quê (ou por que não)
  - latência de inferência
"""
from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from threading import Lock

logger = logging.getLogger("pipeline")


@dataclass
class PipelineStats:
    """Contadores do pipeline, atualizados ao vivo pelas threads do servidor."""
    frames_received: int = 0          # frames de pose recebidos do Unreal
    responses_sent: int = 0           # respostas de inferência enviadas
    sequences_buffered: int = 0       # sequências completas enviadas ao treino
    active_sessions: int = 0          # conexões ativas agora
    wm_steps: int = 0                 # passos de treino do world model
    ac_steps: int = 0                 # passos de treino do actor-critic
    last_wm_loss: float = 0.0
    last_pose_loss: float = 0.0
    last_confidence: float = 0.0
    last_latency_ms: float = 0.0
    buffer_transitions: int = 0
    buffer_episodes: int = 0
    training_active: bool = False
    pose_ready: bool = False          # modelo treinou o bastante p/ gerar poses
    waiting_reason: str = ""          # por que o treino NÃO está rodando, se aplicável

    _lock: Lock = field(default_factory=Lock, repr=False)

    def inc(self, field_name: str, amount: int = 1) -> None:
        with self._lock:
            setattr(self, field_name, getattr(self, field_name) + amount)

    def set(self, **kwargs) -> None:
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)


class PipelineLogger:
    """Emite o painel de status a cada `interval_s` segundos."""

    def __init__(self, stats: PipelineStats, interval_s: float = 5.0) -> None:
        self.stats = stats
        self.interval_s = interval_s
        self._last_emit = 0.0
        self._last_frames = 0
        self._last_responses = 0

    def maybe_emit(self) -> None:
        """Chame isto periodicamente (ex.: a cada resposta). Emite só no intervalo."""
        now = time.time()
        if now - self._last_emit < self.interval_s:
            return
        elapsed = now - self._last_emit if self._last_emit > 0 else self.interval_s
        self._last_emit = now

        s = self.stats
        # Taxas por segundo (desde o último painel)
        fps_in = (s.frames_received - self._last_frames) / max(elapsed, 0.001)
        rps_out = (s.responses_sent - self._last_responses) / max(elapsed, 0.001)
        self._last_frames = s.frames_received
        self._last_responses = s.responses_sent

        # Veredito de treino
        if s.training_active:
            train_line = (
                f"  TREINO: ATIVO ✓  | WM steps={s.wm_steps}  AC steps={s.ac_steps}\n"
                f"          WM loss={s.last_wm_loss:.4f}  pose loss={s.last_pose_loss:.4f}"
            )
        else:
            motivo = s.waiting_reason or "coletando dados"
            train_line = (
                f"  TREINO: AGUARDANDO ✗  | motivo: {motivo}\n"
                f"          buffer: {s.buffer_transitions} transições, "
                f"{s.buffer_episodes} episódios"
            )

        panel = (
            "\n┌─────────────── PIPELINE (estado ao vivo) ───────────────┐\n"
            f"  Conexões ativas: {s.active_sessions}\n"
            f"  ← Recebido do Unreal: {s.frames_received} frames "
            f"({fps_in:.0f}/s)\n"
            f"  → Enviado ao Unreal:  {s.responses_sent} respostas "
            f"({rps_out:.0f}/s)  | latência={s.last_latency_ms:.1f}ms\n"
            f"  Sequências p/ treino: {s.sequences_buffered}\n"
            f"  Confiança atual: {s.last_confidence:.3f}\n"
            f"{train_line}\n"
            "└──────────────────────────────────────────────────────────┘"
        )
        logger.info(panel)
