"""
learning/inverse_dynamics.py
=============================
Rotulagem de ações por dinâmica inversa + recompensa de imitação.

PROBLEMA QUE RESOLVE:
As sequências do líder chegam como observações puras (sem ação, sem recompensa).
Armazená-las com action=0 e reward=0 (como era feito) torna o aprendizado de
política IMPOSSÍVEL:
  - O world model nunca aprende a relação ação → resultado (ação sempre era 0)
  - A reward head aprende a prever 0 em todo lugar
  - O actor-critic não recebe gradiente útil → política fica aleatória para sempre
    (entropia travada em ~0.67, exatamente o que víamos nos logs)

SOLUÇÃO:
1. INVERSE DYNAMICS: inferir qual ação discreta o líder "tomou" em cada frame,
   a partir da velocidade linear no espaço local (forward/right/up) e da velocidade
   angular. Isso converte demonstração-só-observação em tuplas (obs, ação, próx_obs)
   reais → o world model passa a aprender dinâmica condicionada à ação, e o
   actor-critic pode fazer behavioral cloning da demonstração.

2. IMITATION REWARD: recompensa por estar em movimento coerente com o líder.
   Dá à reward head um sinal real para aprender, criando gradiente para o
   actor-critic otimizar em imaginação.

As ações seguem a semântica canônica usada tanto no Python quanto no UE5:
  0=idle  1=forward  2=backward  3=left  4=right  5=run  6=jump  7=crouch  8=stop
"""
from __future__ import annotations

import numpy as np
from typing import List, Tuple


# Limiares (cm/s — unidades UE5). Ajustáveis conforme escala do projeto.
SPEED_IDLE_MAX   = 10.0    # abaixo disso: parado
SPEED_WALK_MAX   = 350.0   # acima disso: corrida
JUMP_VVEL_MIN    = 120.0   # velocidade vertical mínima para considerar pulo


def _quat_to_forward_right(quat_wxyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extrai vetores forward (X) e right (Y) de um quaternion [w,x,y,z]."""
    w, x, y, z = float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])
    # Forward (eixo X local) e Right (eixo Y local) — convenção UE5
    fwd = np.array([
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y + w * z),
        2.0 * (x * z - w * y),
    ], dtype=np.float32)
    right = np.array([
        2.0 * (x * y - w * z),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z + w * x),
    ], dtype=np.float32)
    return fwd, right


def infer_action(
    linear_velocity:  np.ndarray,
    angular_velocity: np.ndarray,
    root_rotation:    np.ndarray,
) -> int:
    """
    Infere a ação discreta (0-8) a partir do movimento do líder.

    Decompõe a velocidade no espaço local (forward/right) para distinguir
    andar para frente/trás/lados, corrida (alta velocidade) e pulo (vel. vertical).
    """
    v = np.asarray(linear_velocity, dtype=np.float32)
    speed3d = float(np.linalg.norm(v))
    v_up    = float(v[2])

    # Pulo: velocidade vertical significativa
    if v_up > JUMP_VVEL_MIN:
        return 6  # jump

    speed2d = float(np.linalg.norm(v[:2]))
    if speed2d < SPEED_IDLE_MAX:
        return 0  # idle

    # Projeta velocidade nos eixos locais do personagem
    fwd, right = _quat_to_forward_right(root_rotation)
    fwd2d   = np.array([fwd[0],   fwd[1]],   dtype=np.float32)
    right2d = np.array([right[0], right[1]], dtype=np.float32)
    n_f = np.linalg.norm(fwd2d)   + 1e-6
    n_r = np.linalg.norm(right2d) + 1e-6
    v_fwd   = float(np.dot(v[:2], fwd2d   / n_f))
    v_right = float(np.dot(v[:2], right2d / n_r))

    # Direção dominante
    if abs(v_fwd) >= abs(v_right):
        if v_fwd >= 0.0:
            return 5 if speed2d > SPEED_WALK_MAX else 1  # run / forward
        return 2  # backward
    else:
        return 4 if v_right >= 0.0 else 3  # right / left


def imitation_reward(action: int, linear_velocity: np.ndarray) -> float:
    """
    Recompensa de imitação para um frame demonstrado pelo líder.

    Princípio: todo frame demonstrado é "bom" (o líder é o oráculo). Recompensa
    base positiva por estar em demonstração, com bônus por movimento ativo —
    incentiva a política a reproduzir movimento em vez de ficar parada.
    """
    speed2d = float(np.linalg.norm(np.asarray(linear_velocity, dtype=np.float32)[:2]))
    base = 1.0                                   # estado demonstrado vale a pena
    motion_bonus = min(speed2d / 300.0, 1.0)     # bônus normalizado por movimento
    idle_penalty = 0.3 if action == 0 else 0.0   # leve desincentivo a ficar parado
    return base + 0.5 * motion_bonus - idle_penalty


def label_sequence(
    pose_frames: List,
    action_dim: int = 9,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotula uma sequência de PoseFrames com ações (one-hot) e recompensas.

    Returns:
        action_seq: (T, action_dim) one-hot
        reward_seq: (T,) float32
    """
    T = len(pose_frames)
    action_seq = np.zeros((T, action_dim), dtype=np.float32)
    reward_seq = np.zeros(T, dtype=np.float32)

    for i, pf in enumerate(pose_frames):
        try:
            a = infer_action(pf.linear_velocity, pf.angular_velocity, pf.root_rotation)
        except Exception:
            a = 0
        a = min(a, action_dim - 1)
        action_seq[i, a] = 1.0
        try:
            reward_seq[i] = imitation_reward(a, pf.linear_velocity)
        except Exception:
            reward_seq[i] = 1.0

    return action_seq, reward_seq
