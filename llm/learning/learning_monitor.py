"""
learning/learning_monitor.py
============================
Telemetria de APRENDIZADO legível. Responde objetivamente: "está aprendendo?".

Um número solto (loss=0.5) não diz nada. O que importa é a TENDÊNCIA: a loss
está caindo? a entropia está diminuindo? o reward está subindo? Este módulo
mantém janelas móveis dessas métricas, calcula a direção de cada uma, e emite
um veredito claro a cada N passos:

    ╔══════════════════════════════════════════════════════════════╗
    ║ APRENDIZADO: SIM ✓   | treino=Luta|MMA  | passo 1500          ║
    ║   WM loss      2.41 ↓ caindo      (bom — modela melhor)       ║
    ║   Pose loss    0.83 ↓ caindo      (bom — animação melhora)    ║
    ║   Entropia     0.42 ↓ caindo      (bom — política decidindo)  ║
    ║   Retorno     +1.15 ↑ subindo     (bom — recompensa cresce)   ║
    ║   Confiança    0.61 ↑ subindo                                 ║
    ╚══════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

logger = logging.getLogger("learning_monitor")


def _trend(window: Deque[float]) -> float:
    """Inclinação simples (fim - início) sobre a janela. >0 sobe, <0 desce."""
    if len(window) < 2:
        return 0.0
    n = len(window)
    first_half = list(window)[: n // 2]
    second_half = list(window)[n // 2:]
    if not first_half or not second_half:
        return 0.0
    return (sum(second_half) / len(second_half)) - (sum(first_half) / len(first_half))


def _arrow(delta: float, eps: float = 1e-4) -> str:
    if delta < -eps:
        return "↓ caindo"
    if delta > eps:
        return "↑ subindo"
    return "→ estável"


@dataclass
class LearningMonitor:
    """Acompanha métricas e emite veredito de aprendizado."""

    window: int = 50
    wm_loss:   Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    pose_loss: Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    entropy:   Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    ret_mean:  Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    conf:      Deque[float] = field(default_factory=lambda: deque(maxlen=50))

    current_training: str = "—"     # ex.: "Luta|MMA"
    current_mode: str = "—"          # ex.: "Observing" / "Inferring"
    frames_received: int = 0
    sequences_received: int = 0

    def record_metrics(self, loss_dict: Dict[str, float]) -> None:
        self.wm_loss.append(float(loss_dict.get("wm/loss", 0.0)))
        self.pose_loss.append(float(loss_dict.get("wm/pose", 0.0)))
        self.entropy.append(float(loss_dict.get("ac/entropy", 0.0)))
        self.ret_mean.append(float(loss_dict.get("ac/returns_mean", 0.0)))

    def record_confidence(self, c: float) -> None:
        self.conf.append(float(c))

    def set_context(self, training: str, mode: str) -> None:
        self.current_training = training or "—"
        self.current_mode = mode or "—"

    # ──────────────────────────────────────────────────────────────────────────
    def verdict(self) -> Dict[str, object]:
        """
        Decide se está aprendendo. Critérios:
          - WM loss caindo OU pose loss caindo (modela melhor)
          - entropia caindo (política decidindo, menos aleatória)
          - retorno subindo (recompensa cresce)
        2+ de 3 positivos = aprendendo.
        """
        d_wm   = _trend(self.wm_loss)
        d_pose = _trend(self.pose_loss)
        d_ent  = _trend(self.entropy)
        d_ret  = _trend(self.ret_mean)

        signals = 0
        if d_wm < -1e-3 or d_pose < -1e-3:
            signals += 1
        if d_ent < -1e-3:
            signals += 1
        if d_ret > 1e-3:
            signals += 1

        learning = signals >= 2
        # Sem dados suficientes ainda
        enough = len(self.wm_loss) >= max(8, self.window // 4)

        return {
            "learning": learning if enough else None,
            "signals": signals,
            "enough_data": enough,
            "d_wm": d_wm, "d_pose": d_pose, "d_ent": d_ent, "d_ret": d_ret,
        }

    # ──────────────────────────────────────────────────────────────────────────
    def _last(self, w: Deque[float]) -> float:
        return w[-1] if w else 0.0

    def render_panel(self, step: int) -> str:
        v = self.verdict()
        if v["learning"] is None:
            status = "COLETANDO DADOS… (aguarde mais frames)"
        elif v["learning"]:
            status = "APRENDIZADO: SIM \u2713"
        else:
            status = "APRENDIZADO: AINDA NÃO \u2717 (sinais insuficientes)"

        def line(label, val, delta, good_when_down=True, extra=""):
            arrow = _arrow(delta)
            good = (delta < 0) if good_when_down else (delta > 0)
            tag = "(bom)" if good and abs(delta) > 1e-3 else ""
            return f"  {label:<12} {val:+8.4f}  {arrow:<12} {tag}{extra}"

        lines = [
            "╔" + "═" * 64 + "╗",
            f"  {status}",
            f"  treino = {self.current_training}   |   modo = {self.current_mode}   |   passo {step}",
            f"  frames recebidos = {self.frames_received}   sequências = {self.sequences_received}",
            "  " + "-" * 60,
            line("WM loss",   self._last(self.wm_loss),   v["d_wm"],  True,  "  modela o mundo"),
            line("Pose loss", self._last(self.pose_loss), v["d_pose"],True,  "  gera animação"),
            line("Entropia",  self._last(self.entropy),   v["d_ent"], True,  "  política decide"),
            line("Retorno",   self._last(self.ret_mean),  v["d_ret"], False, "  recompensa"),
            line("Confiança", self._last(self.conf),      _trend(self.conf), False),
            "╚" + "═" * 64 + "╝",
        ]
        return "\n".join(lines)

    def log_panel(self, step: int) -> None:
        logger.info("\n" + self.render_panel(step))
