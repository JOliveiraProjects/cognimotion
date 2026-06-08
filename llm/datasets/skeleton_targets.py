"""
datasets/skeleton_targets.py
==============================
Registro de sockets IK por tipo de objeto.

Define EXATAMENTE quais bones do skeleton UE5 são usados para cada interação.
Compatível com SK_Mannequin e qualquer skeleton com mapeamento de re-direcionamento.

Convenção de bone names: UE5 Mannequin padrão.
Projetos com skeleton diferente devem sobrescrever via DatasetConfig.bone_remap.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from datasets.schemas import IKSocket, ObjectIKConfig, ObjectType


# ──────────────────────────────────────────────────────────────────────────────
# Registry base (UE5 SK_Mannequin)
# ──────────────────────────────────────────────────────────────────────────────

_REGISTRY: Dict[ObjectType, ObjectIKConfig] = {

    # ── Arma de fogo ──────────────────────────────────────────────────────────
    ObjectType.WEAPON_GUN: ObjectIKConfig(
        object_type=ObjectType.WEAPON_GUN,
        approach_distance=0.0,  # já está na mão, sem approach
        interaction_duration_s=0.3,
        requires_facing=False,
        sockets=[
            IKSocket(
                bone_name="hand_r",
                socket_name="weapon_grip_r",
                offset_pos=(0.0, 0.0, 0.0),
                offset_rot=(0.0, 0.0, 0.0),
                is_primary=True,
                is_secondary=False,
            ),
            IKSocket(
                bone_name="hand_l",
                socket_name="weapon_support_l",
                offset_pos=(0.0, 5.0, 0.0),
                offset_rot=(0.0, 0.0, 0.0),
                is_primary=False,
                is_secondary=True,
            ),
        ],
    ),

    # ── Arma branca (facas, bastões) ──────────────────────────────────────────
    ObjectType.WEAPON_MELEE: ObjectIKConfig(
        object_type=ObjectType.WEAPON_MELEE,
        approach_distance=0.0,
        interaction_duration_s=0.2,
        requires_facing=False,
        sockets=[
            IKSocket(
                bone_name="hand_r",
                socket_name="melee_grip_r",
                offset_pos=(0.0, 0.0, 0.0),
                offset_rot=(0.0, 0.0, 90.0),
                is_primary=True,
                is_secondary=False,
            ),
        ],
    ),

    # ── Bola (futebol, basquete, outros) ─────────────────────────────────────
    ObjectType.BALL: ObjectIKConfig(
        object_type=ObjectType.BALL,
        approach_distance=0.8,
        interaction_duration_s=0.15,
        requires_facing=True,
        sockets=[
            IKSocket(
                bone_name="foot_r",
                socket_name="foot_kick_r",
                offset_pos=(0.0, 0.0, -10.0),
                offset_rot=(30.0, 0.0, 0.0),
                is_primary=True,
                is_secondary=False,
            ),
            IKSocket(
                bone_name="hand_r",
                socket_name="ball_grab_r",
                offset_pos=(0.0, 0.0, 0.0),
                offset_rot=(0.0, 0.0, 0.0),
                is_primary=False,
                is_secondary=False,
            ),
            IKSocket(
                bone_name="hand_l",
                socket_name="ball_grab_l",
                offset_pos=(0.0, 0.0, 0.0),
                offset_rot=(0.0, 0.0, 0.0),
                is_primary=False,
                is_secondary=True,
            ),
        ],
    ),

    # ── Caixa / objeto pesado ─────────────────────────────────────────────────
    ObjectType.BOX: ObjectIKConfig(
        object_type=ObjectType.BOX,
        approach_distance=1.0,
        interaction_duration_s=0.6,
        requires_facing=True,
        sockets=[
            IKSocket(
                bone_name="hand_r",
                socket_name="box_grip_r",
                offset_pos=(15.0, 0.0, 0.0),
                offset_rot=(0.0, 0.0, -90.0),
                is_primary=True,
                is_secondary=False,
            ),
            IKSocket(
                bone_name="hand_l",
                socket_name="box_grip_l",
                offset_pos=(-15.0, 0.0, 0.0),
                offset_rot=(0.0, 0.0, 90.0),
                is_primary=False,
                is_secondary=True,
            ),
            IKSocket(
                bone_name="spine_03",
                socket_name="box_body_brace",
                offset_pos=(0.0, 20.0, 0.0),
                offset_rot=(0.0, 0.0, 0.0),
                is_primary=False,
                is_secondary=False,
            ),
        ],
    ),

    # ── Porta ─────────────────────────────────────────────────────────────────
    ObjectType.DOOR: ObjectIKConfig(
        object_type=ObjectType.DOOR,
        approach_distance=0.8,
        interaction_duration_s=0.8,
        requires_facing=True,
        sockets=[
            IKSocket(
                bone_name="hand_r",
                socket_name="door_handle_r",
                offset_pos=(0.0, 0.0, 0.0),
                offset_rot=(0.0, 0.0, 0.0),
                is_primary=True,
                is_secondary=False,
            ),
        ],
    ),

    # ── Carro ─────────────────────────────────────────────────────────────────
    ObjectType.CAR: ObjectIKConfig(
        object_type=ObjectType.CAR,
        approach_distance=1.5,
        interaction_duration_s=1.2,
        requires_facing=False,
        sockets=[
            IKSocket(
                bone_name="hand_r",
                socket_name="car_door_handle",
                offset_pos=(0.0, 0.0, 0.0),
                offset_rot=(0.0, 0.0, 0.0),
                is_primary=True,
                is_secondary=False,
            ),
            IKSocket(
                bone_name="hand_l",
                socket_name="car_roof_grab",
                offset_pos=(0.0, 0.0, 0.0),
                offset_rot=(0.0, 0.0, 0.0),
                is_primary=False,
                is_secondary=True,
            ),
            IKSocket(
                bone_name="foot_r",
                socket_name="car_step_foot_r",
                offset_pos=(0.0, 0.0, 0.0),
                offset_rot=(0.0, 0.0, 0.0),
                is_primary=False,
                is_secondary=False,
            ),
        ],
    ),

    # ── Moto ──────────────────────────────────────────────────────────────────
    ObjectType.MOTORCYCLE: ObjectIKConfig(
        object_type=ObjectType.MOTORCYCLE,
        approach_distance=1.0,
        interaction_duration_s=0.8,
        requires_facing=False,
        sockets=[
            IKSocket(
                bone_name="hand_r",
                socket_name="moto_handlebar_r",
                offset_pos=(0.0, 0.0, 0.0),
                offset_rot=(0.0, 0.0, 0.0),
                is_primary=True,
                is_secondary=False,
            ),
            IKSocket(
                bone_name="hand_l",
                socket_name="moto_handlebar_l",
                offset_pos=(0.0, 0.0, 0.0),
                offset_rot=(0.0, 0.0, 0.0),
                is_primary=False,
                is_secondary=True,
            ),
            IKSocket(
                bone_name="foot_r",
                socket_name="moto_footpeg_r",
                offset_pos=(0.0, 0.0, 0.0),
                offset_rot=(0.0, 0.0, 0.0),
                is_primary=False,
                is_secondary=False,
            ),
            IKSocket(
                bone_name="foot_l",
                socket_name="moto_footpeg_l",
                offset_pos=(0.0, 0.0, 0.0),
                offset_rot=(0.0, 0.0, 0.0),
                is_primary=False,
                is_secondary=False,
            ),
        ],
    ),

    # ── Bicicleta ─────────────────────────────────────────────────────────────
    ObjectType.BICYCLE: ObjectIKConfig(
        object_type=ObjectType.BICYCLE,
        approach_distance=1.0,
        interaction_duration_s=0.6,
        requires_facing=False,
        sockets=[
            IKSocket(
                bone_name="hand_r",
                socket_name="bike_handle_r",
                offset_pos=(0.0, 0.0, 0.0),
                offset_rot=(0.0, 0.0, 0.0),
                is_primary=True,
                is_secondary=False,
            ),
            IKSocket(
                bone_name="hand_l",
                socket_name="bike_handle_l",
                offset_pos=(0.0, 0.0, 0.0),
                offset_rot=(0.0, 0.0, 0.0),
                is_primary=False,
                is_secondary=True,
            ),
            IKSocket(
                bone_name="foot_r",
                socket_name="bike_pedal_r",
                offset_pos=(0.0, 0.0, 0.0),
                offset_rot=(0.0, 0.0, 0.0),
                is_primary=False,
                is_secondary=False,
            ),
            IKSocket(
                bone_name="foot_l",
                socket_name="bike_pedal_l",
                offset_pos=(0.0, 0.0, 0.0),
                offset_rot=(0.0, 0.0, 0.0),
                is_primary=False,
                is_secondary=False,
            ),
        ],
    ),

    # ── Cavalo / animal ────────────────────────────────────────────────────────
    ObjectType.HORSE: ObjectIKConfig(
        object_type=ObjectType.HORSE,
        approach_distance=1.5,
        interaction_duration_s=1.5,
        requires_facing=False,
        sockets=[
            IKSocket(
                bone_name="hand_r",
                socket_name="horse_rein_r",
                offset_pos=(0.0, 0.0, 0.0),
                offset_rot=(0.0, 0.0, 0.0),
                is_primary=True,
                is_secondary=False,
            ),
            IKSocket(
                bone_name="hand_l",
                socket_name="horse_rein_l",
                offset_pos=(0.0, 0.0, 0.0),
                offset_rot=(0.0, 0.0, 0.0),
                is_primary=False,
                is_secondary=True,
            ),
            IKSocket(
                bone_name="foot_r",
                socket_name="horse_stirrup_r",
                offset_pos=(0.0, 0.0, 0.0),
                offset_rot=(0.0, 0.0, 0.0),
                is_primary=False,
                is_secondary=False,
            ),
            IKSocket(
                bone_name="foot_l",
                socket_name="horse_stirrup_l",
                offset_pos=(0.0, 0.0, 0.0),
                offset_rot=(0.0, 0.0, 0.0),
                is_primary=False,
                is_secondary=False,
            ),
        ],
    ),

    # ── Item de pickup (chave, garrafa, etc.) ─────────────────────────────────
    ObjectType.ITEM_PICKUP: ObjectIKConfig(
        object_type=ObjectType.ITEM_PICKUP,
        approach_distance=0.6,
        interaction_duration_s=0.4,
        requires_facing=True,
        sockets=[
            IKSocket(
                bone_name="hand_r",
                socket_name="item_grip_r",
                offset_pos=(0.0, 0.0, 0.0),
                offset_rot=(0.0, 0.0, 0.0),
                is_primary=True,
                is_secondary=False,
            ),
        ],
    ),

    # ── Assento de veículo (avião, ônibus, etc.) ───────────────────────────────
    ObjectType.VEHICLE_SEAT: ObjectIKConfig(
        object_type=ObjectType.VEHICLE_SEAT,
        approach_distance=1.0,
        interaction_duration_s=1.0,
        requires_facing=False,
        sockets=[
            IKSocket(
                bone_name="hand_r",
                socket_name="seat_armrest_r",
                offset_pos=(0.0, 0.0, 0.0),
                offset_rot=(0.0, 0.0, 0.0),
                is_primary=True,
                is_secondary=False,
            ),
            IKSocket(
                bone_name="hand_l",
                socket_name="seat_armrest_l",
                offset_pos=(0.0, 0.0, 0.0),
                offset_rot=(0.0, 0.0, 0.0),
                is_primary=False,
                is_secondary=True,
            ),
        ],
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# API pública
# ──────────────────────────────────────────────────────────────────────────────

def get_ik_config(object_type: ObjectType) -> Optional[ObjectIKConfig]:
    return _REGISTRY.get(object_type)


def get_primary_socket(object_type: ObjectType) -> Optional[IKSocket]:
    cfg = _REGISTRY.get(object_type)
    return cfg.primary_socket if cfg else None


def get_all_sockets(object_type: ObjectType) -> List[IKSocket]:
    cfg = _REGISTRY.get(object_type)
    return cfg.sockets if cfg else []


def register_custom(config: ObjectIKConfig) -> None:
    _REGISTRY[config.object_type] = config


def apply_bone_remap(remap: Dict[str, str]) -> None:
    """
    Remapeia bone_names para skeleton customizado.
    remap = {"hand_r": "RightHand", "foot_r": "RightFoot", ...}
    """
    for cfg in _REGISTRY.values():
        for socket in cfg.sockets:
            if socket.bone_name in remap:
                socket.bone_name = remap[socket.bone_name]


def all_object_types() -> List[ObjectType]:
    return list(_REGISTRY.keys())
