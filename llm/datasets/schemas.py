"""
datasets/schemas.py
====================
Schemas de dados para todos os tipos de interação NPC:
  - Ações interativas (expandidas além de ECognitiveMotionStyle)
  - Tipos de objeto (arma, bola, veículo, animal, sinal)
  - Sockets de IK por tipo de objeto (pontos de fixação no skeleton)
  - Estados de ameaça
  - Sinal de contexto (farol, zona)
  - Sequência de treinamento para o RSSM
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Espaço de ação estendido (ECognitiveInteractionAction)
# 0-8  : movement (ECognitiveMotionStyle — compatível)
# 9-19 : interaction actions (novas)
# ──────────────────────────────────────────────────────────────────────────────

class InteractionAction(IntEnum):
    # ── Movimento (compatível com ECognitiveMotionStyle 0-8) ──────────────────
    IDLE             = 0
    MOVE_AGGRESSIVE  = 1
    MOVE_RELAXED     = 2
    MOVE_INJURED     = 3
    MOVE_FATIGUED    = 4
    MOVE_STEALTH     = 5
    MOVE_MILITARY    = 6
    MOVE_CIVILIAN    = 7
    MOVE_CRIMINAL    = 8
    # ── Interação com objetos ─────────────────────────────────────────────────
    GRAB_OBJECT      = 9
    RELEASE_OBJECT   = 10
    KICK_OBJECT      = 11
    PUSH_OBJECT      = 12
    AIM_WEAPON       = 13
    HOLSTER_WEAPON   = 14
    # ── Veículos / montaria ───────────────────────────────────────────────────
    ENTER_VEHICLE    = 15
    EXIT_VEHICLE     = 16
    MOUNT_ANIMAL     = 17
    DISMOUNT_ANIMAL  = 18
    MOUNT_BICYCLE    = 19
    # ── Ameaça / resposta ─────────────────────────────────────────────────────
    ATTACK_THREAT    = 20
    RETREAT_THREAT   = 21
    SURRENDER        = 22
    TAKE_COVER       = 23

INTERACTION_ACTION_DIM = len(InteractionAction)  # 24


# ──────────────────────────────────────────────────────────────────────────────
# Tipos de objeto que o NPC pode interagir
# ──────────────────────────────────────────────────────────────────────────────

class ObjectType(IntEnum):
    NONE         = 0
    WEAPON_GUN   = 1
    WEAPON_MELEE = 2
    BALL         = 3
    BOX          = 4
    DOOR         = 5
    CAR          = 6
    MOTORCYCLE   = 7
    BICYCLE      = 8
    HORSE        = 9
    TRAFFIC_LIGHT = 10
    ZONE_TRIGGER = 11
    THREAT_NPC   = 12
    ALLY_NPC     = 13
    ITEM_PICKUP  = 14
    VEHICLE_SEAT = 15


# ──────────────────────────────────────────────────────────────────────────────
# Estado de ameaça
# ──────────────────────────────────────────────────────────────────────────────

class ThreatLevel(IntEnum):
    NONE     = 0   # nenhuma ameaça
    DETECTED = 1   # detectado mas não imediato
    NEAR     = 2   # ameaça próxima
    DIRECT   = 3   # ameaça direta (sendo atacado)
    ARMED    = 4   # ameaça armada


class ThreatRole(IntEnum):
    NEUTRAL  = 0
    TARGET   = 1   # este NPC é a ameaça para outro
    VICTIM   = 2   # este NPC está sendo ameaçado


# ──────────────────────────────────────────────────────────────────────────────
# Estado de sinal de contexto (farol, zona)
# ──────────────────────────────────────────────────────────────────────────────

class TrafficLightState(IntEnum):
    UNKNOWN = 0
    RED     = 1
    YELLOW  = 2
    GREEN   = 3

class ZoneType(IntEnum):
    NONE        = 0
    VEHICLE     = 1   # zona de embarque/desembarque
    COMBAT      = 2   # zona de combate
    SAFE        = 3   # zona segura
    INTERACTION = 4   # zona de interação com objeto


# ──────────────────────────────────────────────────────────────────────────────
# Socket IK — ponto de fixação de objeto no skeleton
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class IKSocket:
    """Definição de um socket IK no skeleton UE5."""
    bone_name:    str          # ex: "hand_r", "hand_l", "spine_03"
    socket_name:  str          # ex: "weapon_grip_r"
    offset_pos:   Tuple[float, float, float] = (0.0, 0.0, 0.0)
    offset_rot:   Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Euler degrees
    is_primary:   bool = True   # mão/ponto principal de fixação
    is_secondary: bool = False  # mão/ponto de suporte

    def to_numpy(self) -> "np.ndarray":
        import numpy as np
        return np.array([*self.offset_pos, *self.offset_rot], dtype=np.float32)


@dataclass
class ObjectIKConfig:
    """Configuração de IK para um tipo de objeto."""
    object_type:  ObjectType
    sockets:      List[IKSocket] = field(default_factory=list)
    approach_distance: float = 1.5   # metros — distância para iniciar interação
    interaction_duration_s: float = 0.5  # duração da animação de interação
    requires_facing: bool = True     # NPC deve estar de frente para o objeto

    @property
    def primary_socket(self) -> Optional[IKSocket]:
        return next((s for s in self.sockets if s.is_primary), None)

    @property
    def secondary_socket(self) -> Optional[IKSocket]:
        return next((s for s in self.sockets if s.is_secondary), None)


# ──────────────────────────────────────────────────────────────────────────────
# Cenário de interação — unidade de dado de treinamento
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class InteractionContext:
    """Contexto que o NPC observa em um frame."""
    # Estado do NPC
    npc_position:    Tuple[float, float, float] = (0.0, 0.0, 0.0)
    npc_velocity:    Tuple[float, float, float] = (0.0, 0.0, 0.0)
    npc_health:      float = 1.0
    npc_stamina:     float = 1.0
    is_armed:        bool  = False

    # Objeto mais próximo
    object_type:     ObjectType = ObjectType.NONE
    object_distance: float = 10.0
    object_angle_h:  float = 0.0   # horizontal (graus)
    object_angle_v:  float = 0.0   # vertical (graus)
    object_is_held:  bool  = False  # NPC já segura este objeto
    object_is_moving: bool = False  # bola rolando, carro em movimento

    # Estado de ameaça
    threat_level:    ThreatLevel  = ThreatLevel.NONE
    threat_role:     ThreatRole   = ThreatRole.NEUTRAL
    threat_distance: float        = 100.0
    threat_angle:    float        = 0.0
    threat_armed:    bool         = False

    # Sinais de contexto
    traffic_light:   TrafficLightState = TrafficLightState.UNKNOWN
    zone_type:       ZoneType          = ZoneType.NONE
    vehicle_available: bool            = False
    seat_is_empty:   bool              = False

    # Ação anterior (histórico)
    last_action:     InteractionAction = InteractionAction.IDLE
    action_hold_frames: int            = 0


@dataclass
class InteractionStep:
    """Um passo de uma sequência de treinamento."""
    context:     InteractionContext
    action:      InteractionAction
    reward:      float
    done:        bool
    ik_sockets:  List[IKSocket] = field(default_factory=list)


@dataclass
class InteractionSequence:
    """Sequência completa para treinamento do RSSM."""
    scenario_id:  str
    object_type:  ObjectType
    steps:        List[InteractionStep] = field(default_factory=list)
    total_reward: float = 0.0
    success:      bool  = False

    def to_numpy_arrays(self, obs_dim: int = 256, action_dim: int = INTERACTION_ACTION_DIM) -> tuple:
        """Converte para arrays numpy prontos para SequenceBuffer.

        action_dim: deve coincidir com o SequenceBuffer e o RSSM.
        Ações com índice >= action_dim são mapeadas para a ação 0 (idle).
        """
        import numpy as np
        T = len(self.steps)
        obs_arr  = np.zeros((T, obs_dim),   dtype=np.float32)
        act_arr  = np.zeros((T, action_dim), dtype=np.float32)
        rew_arr  = np.zeros(T,              dtype=np.float32)
        done_arr = np.zeros(T,              dtype=bool)

        for i, step in enumerate(self.steps):
            obs_arr[i]  = context_to_obs(step.context, obs_dim)
            idx = int(step.action)
            if idx < action_dim:
                act_arr[i, idx] = 1.0
            else:
                act_arr[i, 0] = 1.0
            rew_arr[i]  = step.reward
            done_arr[i] = step.done

        return obs_arr, act_arr, rew_arr, done_arr


# ──────────────────────────────────────────────────────────────────────────────
# Encoder: InteractionContext → obs vector (256-d)
# ──────────────────────────────────────────────────────────────────────────────

def context_to_obs(ctx: InteractionContext, obs_dim: int = 256) -> "np.ndarray":
    """
    Serializa InteractionContext em vetor de observação (obs_dim floats).

    Layout (total fixo: 96 floats + padding até obs_dim):
      [0:3]   npc_position (norm / 100)
      [3:6]   npc_velocity (norm / 10)
      [6]     npc_health
      [7]     npc_stamina
      [8]     is_armed
      [9:25]  object_type one-hot (16 classes)
      [25]    object_distance (norm / 50)
      [26]    object_angle_h (sin)
      [27]    object_angle_h (cos)
      [28]    object_angle_v (sin)
      [29]    object_is_held
      [30]    object_is_moving
      [31:36] threat: level/role/distance/angle/armed
      [36:40] traffic_light one-hot (4 states)
      [40:44] zone_type one-hot (5 states, crop at 4)
      [44]    vehicle_available
      [45]    seat_is_empty
      [46:70] last_action one-hot (24 actions)
      [70]    action_hold_frames (norm / 60)
      [71:96] padding zeros (reserved for future signals)
      [96:]   noise-padded zeros to obs_dim
    """
    import math
    import numpy as np

    v = np.zeros(obs_dim, dtype=np.float32)
    i = 0

    def _put(vals):
        nonlocal i
        arr = np.asarray(vals, dtype=np.float32).flatten()
        n   = min(len(arr), obs_dim - i)
        v[i:i + n] = arr[:n]
        i += n

    _put(np.array(ctx.npc_position) / 100.0)
    _put(np.array(ctx.npc_velocity) / 10.0)
    _put([ctx.npc_health, ctx.npc_stamina, float(ctx.is_armed)])

    obj_oh = np.zeros(len(ObjectType), dtype=np.float32)
    obj_oh[int(ctx.object_type)] = 1.0
    _put(obj_oh[:16])

    _put([
        ctx.object_distance / 50.0,
        math.sin(math.radians(ctx.object_angle_h)),
        math.cos(math.radians(ctx.object_angle_h)),
        math.sin(math.radians(ctx.object_angle_v)),
        float(ctx.object_is_held),
        float(ctx.object_is_moving),
    ])

    _put([
        int(ctx.threat_level) / 4.0,
        int(ctx.threat_role)  / 2.0,
        ctx.threat_distance / 50.0,
        math.sin(math.radians(ctx.threat_angle)),
        float(ctx.threat_armed),
    ])

    tl_oh = np.zeros(4, dtype=np.float32)
    tl_oh[int(ctx.traffic_light)] = 1.0
    _put(tl_oh)

    zone_oh = np.zeros(5, dtype=np.float32)
    zone_oh[int(ctx.zone_type)] = 1.0
    _put(zone_oh[:4])

    _put([float(ctx.vehicle_available), float(ctx.seat_is_empty)])

    act_oh = np.zeros(INTERACTION_ACTION_DIM, dtype=np.float32)
    act_oh[int(ctx.last_action)] = 1.0
    _put(act_oh)

    _put([ctx.action_hold_frames / 60.0])

    return v
