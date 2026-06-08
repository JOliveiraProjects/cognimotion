from __future__ import annotations

import math
import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ─── Ações disponíveis ────────────────────────────────────────────────────────
# Índice → (dx, dy, speed_multiplier)
_ACTION_MAP: List[Tuple[float, float, float]] = [
    ( 0.0,  0.0, 0.0),   # 0: idle
    ( 1.0,  0.0, 0.5),   # 1: walk fwd
    (-1.0,  0.0, 0.5),   # 2: walk bwd
    ( 0.0,  1.0, 0.5),   # 3: strafe right
    ( 0.0, -1.0, 0.5),   # 4: strafe left
    ( 1.0,  0.0, 1.0),   # 5: run fwd
    (-1.0,  0.0, 1.0),   # 6: run bwd
    ( 0.707, 0.707, 0.7), # 7: diagonal fwd-right
    ( 0.707,-0.707, 0.7), # 8: diagonal fwd-left
]
ACTION_DIM = len(_ACTION_MAP)


@dataclass
class EnvState:
    x:           float = 0.0
    y:           float = 0.0
    vx:          float = 0.0
    vy:          float = 0.0
    target_x:    float = 5.0
    target_y:    float = 5.0
    step:        int   = 0
    done:        bool  = False
    total_reward: float = 0.0


class MotionEnv:
    """
    Ambiente 2D de simulação de movimento para depuração local do DreamerV3.

    Observação: vetor de 256 floats (preenchido com estado + padding zeros)
    para ser compatível com o embedding_dim do PoseEncoder real.

    Recompensa:
      - Negativa proporcional à distância ao alvo
      - Positiva ao chegar no alvo (distância < 0.5)
      - Penalidade suave por velocidade excessiva
      - Penalidade por inatividade (idle repetido)
    """

    OBS_DIM:     int   = 256
    MAX_STEPS:   int   = 200
    ARENA_SIZE:  float = 20.0
    ARRIVE_DIST: float = 0.5
    MAX_SPEED:   float = 2.0
    DT:          float = 0.1
    FRICTION:    float = 0.85

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self._state = EnvState()
        self._prev_dist: float = 0.0
        self._idle_count: int = 0

    # ──────────────────────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        margin = 2.0
        lim    = self.ARENA_SIZE / 2.0 - margin
        self._state = EnvState(
            x=self._rng.uniform(-lim, lim),
            y=self._rng.uniform(-lim, lim),
            target_x=self._rng.uniform(-lim, lim),
            target_y=self._rng.uniform(-lim, lim),
        )
        while self._dist_to_target() < 2.0:
            self._state.target_x = self._rng.uniform(-lim, lim)
            self._state.target_y = self._rng.uniform(-lim, lim)

        self._prev_dist  = self._dist_to_target()
        self._idle_count = 0
        return self._make_obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        action = int(np.clip(action, 0, ACTION_DIM - 1))
        dx, dy, spd = _ACTION_MAP[action]

        # Física
        ax = dx * spd * self.MAX_SPEED
        ay = dy * spd * self.MAX_SPEED
        s = self._state
        s.vx = s.vx * self.FRICTION + ax * self.DT
        s.vy = s.vy * self.FRICTION + ay * self.DT

        # Clip de velocidade
        speed = math.sqrt(s.vx ** 2 + s.vy ** 2)
        if speed > self.MAX_SPEED:
            s.vx *= self.MAX_SPEED / speed
            s.vy *= self.MAX_SPEED / speed

        s.x = float(np.clip(s.x + s.vx * self.DT, -self.ARENA_SIZE / 2, self.ARENA_SIZE / 2))
        s.y = float(np.clip(s.y + s.vy * self.DT, -self.ARENA_SIZE / 2, self.ARENA_SIZE / 2))
        s.step += 1

        # Recompensa
        dist      = self._dist_to_target()
        progress  = self._prev_dist - dist             # positivo se aproximou
        r_progress = progress * 2.0                    # escala
        r_idle    = -0.05 if action == 0 else 0.0
        r_speed   = -0.01 * speed ** 2
        r_arrive  = 0.0

        arrived = dist < self.ARRIVE_DIST
        if arrived:
            r_arrive = 5.0 - s.step * 0.01
            s.done   = True

        timeout = s.step >= self.MAX_STEPS
        if timeout:
            s.done = True

        # idle counter
        self._idle_count = self._idle_count + 1 if action == 0 else 0

        reward        = r_progress + r_idle + r_speed + r_arrive
        s.total_reward += reward
        self._prev_dist = dist

        info = {
            "dist":     dist,
            "arrived":  arrived,
            "timeout":  timeout,
            "step":     s.step,
            "total_r":  s.total_reward,
        }
        obs = self._make_obs()
        return obs, float(reward), bool(s.done), info

    # ──────────────────────────────────────────────────────────────────────────

    def _dist_to_target(self) -> float:
        s = self._state
        return math.sqrt((s.x - s.target_x) ** 2 + (s.y - s.target_y) ** 2)

    def _make_obs(self) -> np.ndarray:
        s   = self._state
        dist  = self._dist_to_target()
        angle = math.atan2(s.target_y - s.y, s.target_x - s.x)

        raw = np.array([
            s.x / (self.ARENA_SIZE / 2),
            s.y / (self.ARENA_SIZE / 2),
            s.vx / self.MAX_SPEED,
            s.vy / self.MAX_SPEED,
            (s.target_x - s.x) / self.ARENA_SIZE,
            (s.target_y - s.y) / self.ARENA_SIZE,
            dist / self.ARENA_SIZE,
            math.sin(angle),
            math.cos(angle),
            s.step / self.MAX_STEPS,
        ], dtype=np.float32)

        # Pad para obs_dim=256 para compatibilidade com PoseEncoder embedding
        obs = np.zeros(self.OBS_DIM, dtype=np.float32)
        obs[:len(raw)] = raw
        # Adiciona ruído leve para enriquecer a distribuição de treinamento
        obs += self._np_rng.normal(0.0, 0.01, self.OBS_DIM).astype(np.float32)
        return obs

    # ──────────────────────────────────────────────────────────────────────────

    @property
    def obs_dim(self) -> int:
        return self.OBS_DIM

    @property
    def action_dim(self) -> int:
        return ACTION_DIM

    @property
    def state(self) -> EnvState:
        return self._state

    def render(self) -> str:
        s    = self._state
        dist = self._dist_to_target()
        return (
            f"Step={s.step:3d} | pos=({s.x:.2f},{s.y:.2f}) "
            f"| target=({s.target_x:.2f},{s.target_y:.2f}) "
            f"| dist={dist:.2f} | R={s.total_reward:.2f}"
        )
