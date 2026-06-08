"""
datasets/scenario_generator.py
================================
Gerador de sequências de treinamento para cada tipo de interação.

Cada método gera N sequências completas (approach → interact → result)
prontas para inserção no SequenceBuffer do DreamerV3.
"""
from __future__ import annotations

import logging
import math
import random
from typing import List, Optional

from datasets.schemas import (
    InteractionAction, InteractionContext, InteractionSequence, InteractionStep,
    ObjectType, ThreatLevel, ThreatRole, TrafficLightState, ZoneType,
    INTERACTION_ACTION_DIM,
)
from datasets.skeleton_targets import get_ik_config, get_all_sockets

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Reward shaping constants
# ──────────────────────────────────────────────────────────────────────────────

R_APPROACH_PROGRESS    =  0.10   # por metro de aproximação
R_INTERACTION_SUCCESS  =  2.00
R_INTERACTION_FAIL     = -0.50
R_IDLE_PENALTY         = -0.02
R_THREAT_FLEE          =  0.30   # por metro de distância de ameaça direta
R_THREAT_ATTACK_HIT    =  1.50
R_THREAT_WRONG_ACTION  = -0.80
R_VEHICLE_ENTER        =  1.80
R_VEHICLE_EXIT         =  1.00
R_SIGNAL_OBEY          =  0.50   # obedece farol vermelho
R_SIGNAL_DISOBEY       = -1.00   # passa no vermelho


class ScenarioGenerator:
    """
    Gera sequências de treinamento para cada categoria de interação.
    Todas as sequências são determinísticas dado um seed.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)

    # ──────────────────────────────────────────────────────────────────────────
    # Arma
    # ──────────────────────────────────────────────────────────────────────────

    def gen_weapon_pickup(self, n: int = 50) -> List[InteractionSequence]:
        seqs = []
        for i in range(n):
            dist     = self._rng.uniform(1.5, 8.0)
            angle    = self._rng.uniform(-45.0, 45.0)
            obj_type = self._rng.choice([ObjectType.WEAPON_GUN, ObjectType.WEAPON_MELEE])
            ik_cfg   = get_ik_config(obj_type)
            steps: List[InteractionStep] = []
            prev_dist = dist

            # Approach phase (até 10 steps)
            for t in range(10):
                dist -= self._rng.uniform(0.3, 0.8)
                dist  = max(dist, 0.0)
                ctx   = InteractionContext(
                    object_type=obj_type,
                    object_distance=dist,
                    object_angle_h=angle,
                    last_action=InteractionAction.MOVE_CIVILIAN,
                )
                progress = prev_dist - dist
                reward   = progress * R_APPROACH_PROGRESS
                done     = dist < 0.5
                steps.append(InteractionStep(
                    context=ctx,
                    action=InteractionAction.MOVE_CIVILIAN,
                    reward=reward,
                    done=done and t < 9,
                    ik_sockets=[],
                ))
                prev_dist = dist
                if dist < 0.5:
                    break

            # Grab phase (1 step)
            ctx_grab = InteractionContext(
                object_type=obj_type, object_distance=0.1,
                object_angle_h=0.0, is_armed=(obj_type == ObjectType.WEAPON_GUN),
                object_is_held=True,
                last_action=InteractionAction.MOVE_CIVILIAN,
            )
            steps.append(InteractionStep(
                context=ctx_grab,
                action=InteractionAction.GRAB_OBJECT,
                reward=R_INTERACTION_SUCCESS,
                done=True,
                ik_sockets=get_all_sockets(obj_type),
            ))

            seqs.append(InteractionSequence(
                scenario_id=f"weapon_pickup_{i}",
                object_type=obj_type,
                steps=steps,
                total_reward=sum(s.reward for s in steps),
                success=True,
            ))
        return seqs

    def gen_weapon_aim(self, n: int = 30) -> List[InteractionSequence]:
        seqs = []
        for i in range(n):
            obj_type = ObjectType.WEAPON_GUN
            ik_cfg   = get_ik_config(obj_type)
            steps    = []

            for t in range(5):
                ctx = InteractionContext(
                    object_type=ObjectType.THREAT_NPC,
                    object_distance=self._rng.uniform(2.0, 15.0),
                    object_angle_h=self._rng.uniform(-30.0, 30.0),
                    is_armed=True,
                    last_action=InteractionAction.AIM_WEAPON,
                    threat_level=ThreatLevel.DIRECT,
                    threat_role=ThreatRole.TARGET,
                )
                steps.append(InteractionStep(
                    context=ctx,
                    action=InteractionAction.AIM_WEAPON,
                    reward=0.05,
                    done=(t == 4),
                    ik_sockets=get_all_sockets(obj_type),
                ))

            seqs.append(InteractionSequence(
                scenario_id=f"weapon_aim_{i}",
                object_type=obj_type,
                steps=steps,
                total_reward=sum(s.reward for s in steps),
                success=True,
            ))
        return seqs

    # ──────────────────────────────────────────────────────────────────────────
    # Bola
    # ──────────────────────────────────────────────────────────────────────────

    def gen_ball_kick(self, n: int = 60) -> List[InteractionSequence]:
        seqs = []
        for i in range(n):
            dist = self._rng.uniform(0.5, 5.0)
            steps = []
            prev  = dist

            for t in range(8):
                dist -= self._rng.uniform(0.3, 0.7)
                dist  = max(dist, 0.0)
                ctx   = InteractionContext(
                    object_type=ObjectType.BALL,
                    object_distance=dist,
                    object_angle_h=self._rng.uniform(-20.0, 20.0),
                    object_is_moving=(dist > 1.0),
                    last_action=InteractionAction.MOVE_CIVILIAN,
                )
                progress = prev - dist
                reward   = progress * R_APPROACH_PROGRESS
                done     = dist < 0.5
                action   = InteractionAction.KICK_OBJECT if dist < 0.5 else InteractionAction.MOVE_CIVILIAN
                if dist < 0.5:
                    reward += R_INTERACTION_SUCCESS
                steps.append(InteractionStep(
                    context=ctx, action=action,
                    reward=reward, done=done,
                    ik_sockets=get_all_sockets(ObjectType.BALL) if dist < 0.5 else [],
                ))
                prev = dist
                if dist < 0.5:
                    break

            seqs.append(InteractionSequence(
                scenario_id=f"ball_kick_{i}",
                object_type=ObjectType.BALL,
                steps=steps,
                total_reward=sum(s.reward for s in steps),
                success=True,
            ))
        return seqs

    def gen_ball_grab(self, n: int = 40) -> List[InteractionSequence]:
        seqs = []
        for i in range(n):
            dist  = self._rng.uniform(0.3, 3.0)
            steps = []
            prev  = dist

            for t in range(6):
                dist -= self._rng.uniform(0.2, 0.6)
                dist  = max(dist, 0.0)
                action = InteractionAction.GRAB_OBJECT if dist < 0.6 else InteractionAction.MOVE_CIVILIAN
                ctx    = InteractionContext(
                    object_type=ObjectType.BALL,
                    object_distance=dist,
                    object_angle_h=self._rng.uniform(-15.0, 15.0),
                    object_is_held=(dist < 0.3),
                    last_action=action,
                )
                reward = (prev - dist) * R_APPROACH_PROGRESS + (R_INTERACTION_SUCCESS if dist < 0.3 else 0.0)
                steps.append(InteractionStep(
                    context=ctx, action=action,
                    reward=reward, done=(dist < 0.3),
                    ik_sockets=get_all_sockets(ObjectType.BALL) if dist < 0.3 else [],
                ))
                prev = dist
                if dist < 0.3:
                    break

            seqs.append(InteractionSequence(
                scenario_id=f"ball_grab_{i}",
                object_type=ObjectType.BALL,
                steps=steps,
                total_reward=sum(s.reward for s in steps),
                success=True,
            ))
        return seqs

    def gen_ball_push(self, n: int = 30) -> List[InteractionSequence]:
        seqs = []
        for i in range(n):
            dist  = self._rng.uniform(0.5, 2.0)
            steps = []
            prev  = dist
            for t in range(5):
                dist -= self._rng.uniform(0.2, 0.5)
                dist  = max(dist, 0.0)
                action = InteractionAction.PUSH_OBJECT if dist < 0.6 else InteractionAction.MOVE_CIVILIAN
                ctx    = InteractionContext(
                    object_type=ObjectType.BALL, object_distance=dist,
                    object_angle_h=0.0, object_is_moving=(dist < 0.3),
                    last_action=action,
                )
                reward = (prev - dist) * R_APPROACH_PROGRESS + (R_INTERACTION_SUCCESS * 0.7 if dist < 0.3 else 0.0)
                steps.append(InteractionStep(
                    context=ctx, action=action, reward=reward, done=(dist < 0.2),
                    ik_sockets=get_all_sockets(ObjectType.BALL) if dist < 0.3 else [],
                ))
                prev = dist
                if dist < 0.2:
                    break

            seqs.append(InteractionSequence(
                scenario_id=f"ball_push_{i}",
                object_type=ObjectType.BALL, steps=steps,
                total_reward=sum(s.reward for s in steps), success=True,
            ))
        return seqs

    # ──────────────────────────────────────────────────────────────────────────
    # Ameaça
    # ──────────────────────────────────────────────────────────────────────────

    def gen_threat_detection(self, n: int = 80) -> List[InteractionSequence]:
        seqs = []
        for i in range(n):
            armed     = self._rng.random() > 0.4
            role      = self._rng.choice([ThreatRole.VICTIM, ThreatRole.TARGET])
            level     = self._rng.choice([ThreatLevel.DETECTED, ThreatLevel.NEAR, ThreatLevel.DIRECT])
            dist_init = self._rng.uniform(5.0, 20.0)
            steps     = []

            dist = dist_init
            for t in range(12):
                if role == ThreatRole.VICTIM:
                    dist -= self._rng.uniform(0.2, 0.5)  # ameaça se aproxima
                else:
                    pass  # NPC é a ameaça — mantém distância

                dist    = max(0.5, dist)
                t_level = ThreatLevel.DIRECT if dist < 3.0 else (ThreatLevel.NEAR if dist < 8.0 else ThreatLevel.DETECTED)

                ctx = InteractionContext(
                    threat_level=t_level,
                    threat_role=role,
                    threat_distance=dist,
                    threat_angle=self._rng.uniform(-180.0, 180.0),
                    threat_armed=armed,
                    is_armed=(role == ThreatRole.TARGET and armed),
                    last_action=InteractionAction.IDLE,
                )

                if role == ThreatRole.VICTIM:
                    if t_level == ThreatLevel.DIRECT:
                        action = InteractionAction.TAKE_COVER if not armed else InteractionAction.ATTACK_THREAT
                        reward = R_THREAT_FLEE * 0.5
                    elif t_level == ThreatLevel.NEAR:
                        action = InteractionAction.RETREAT_THREAT
                        reward = R_THREAT_FLEE
                    else:
                        action = InteractionAction.MOVE_STEALTH
                        reward = 0.05
                else:
                    if t_level == ThreatLevel.DIRECT and armed:
                        action = InteractionAction.AIM_WEAPON
                        reward = R_THREAT_ATTACK_HIT * 0.3
                    else:
                        action = InteractionAction.MOVE_MILITARY
                        reward = 0.05

                done = (dist < 1.0 and t_level == ThreatLevel.DIRECT)
                steps.append(InteractionStep(
                    context=ctx, action=action, reward=reward, done=done,
                    ik_sockets=get_all_sockets(ObjectType.WEAPON_GUN) if armed and role == ThreatRole.TARGET else [],
                ))
                if done:
                    break

            seqs.append(InteractionSequence(
                scenario_id=f"threat_{i}",
                object_type=ObjectType.THREAT_NPC,
                steps=steps,
                total_reward=sum(s.reward for s in steps),
                success=(role == ThreatRole.VICTIM and dist > 3.0),
            ))
        return seqs

    # ──────────────────────────────────────────────────────────────────────────
    # Veículos (farol como sinal de contexto)
    # ──────────────────────────────────────────────────────────────────────────

    def gen_vehicle_enter_exit(
        self,
        vehicle_type: ObjectType = ObjectType.CAR,
        n: int = 50,
    ) -> List[InteractionSequence]:
        seqs = []
        for i in range(n):
            ik_cfg = get_ik_config(vehicle_type)
            dist   = self._rng.uniform(2.0, 8.0)
            steps  = []
            prev   = dist

            # Approach com farol verde (pode entrar)
            light  = TrafficLightState.GREEN
            for t in range(8):
                dist -= self._rng.uniform(0.4, 1.0)
                dist  = max(0.0, dist)
                ctx   = InteractionContext(
                    object_type=vehicle_type,
                    object_distance=dist,
                    object_angle_h=self._rng.uniform(-30.0, 30.0),
                    vehicle_available=True,
                    seat_is_empty=True,
                    traffic_light=light,
                    zone_type=ZoneType.VEHICLE,
                    last_action=InteractionAction.MOVE_CIVILIAN,
                )
                action = InteractionAction.ENTER_VEHICLE if dist < ik_cfg.approach_distance else InteractionAction.MOVE_CIVILIAN
                reward = (prev - dist) * R_APPROACH_PROGRESS
                if action == InteractionAction.ENTER_VEHICLE:
                    reward += R_VEHICLE_ENTER
                done = (action == InteractionAction.ENTER_VEHICLE)
                steps.append(InteractionStep(
                    context=ctx, action=action, reward=reward, done=done,
                    ik_sockets=get_all_sockets(vehicle_type) if done else [],
                ))
                prev = dist
                if done:
                    break

            # Exit phase (após estar no veículo)
            for t in range(4):
                ctx_exit = InteractionContext(
                    object_type=vehicle_type,
                    object_distance=0.0,
                    vehicle_available=True,
                    seat_is_empty=False,
                    traffic_light=TrafficLightState.RED,
                    zone_type=ZoneType.VEHICLE,
                    last_action=InteractionAction.ENTER_VEHICLE,
                )
                action = InteractionAction.EXIT_VEHICLE
                steps.append(InteractionStep(
                    context=ctx_exit, action=action,
                    reward=R_VEHICLE_EXIT, done=(t == 3),
                    ik_sockets=get_all_sockets(vehicle_type),
                ))
                if t == 3:
                    break

            seqs.append(InteractionSequence(
                scenario_id=f"vehicle_{vehicle_type.name.lower()}_{i}",
                object_type=vehicle_type,
                steps=steps,
                total_reward=sum(s.reward for s in steps),
                success=True,
            ))
        return seqs

    def gen_traffic_light_response(self, n: int = 60) -> List[InteractionSequence]:
        seqs = []
        for i in range(n):
            light_seq = [
                TrafficLightState.GREEN,
                TrafficLightState.YELLOW,
                TrafficLightState.RED,
            ]
            steps = []

            for light in light_seq:
                ctx = InteractionContext(
                    object_type=ObjectType.TRAFFIC_LIGHT,
                    object_distance=5.0,
                    traffic_light=light,
                    zone_type=ZoneType.VEHICLE,
                    vehicle_available=(self._rng.random() > 0.4),
                    last_action=InteractionAction.IDLE,
                )

                if light == TrafficLightState.GREEN:
                    action = self._rng.choice([
                        InteractionAction.MOVE_CIVILIAN,
                        InteractionAction.ENTER_VEHICLE,
                    ])
                    reward = R_SIGNAL_OBEY
                elif light == TrafficLightState.YELLOW:
                    action = InteractionAction.MOVE_CIVILIAN
                    reward = R_SIGNAL_OBEY * 0.5
                else:  # RED
                    action = InteractionAction.IDLE
                    reward = R_SIGNAL_OBEY

                steps.append(InteractionStep(
                    context=ctx, action=action, reward=reward,
                    done=(light == TrafficLightState.RED),
                    ik_sockets=[],
                ))

            seqs.append(InteractionSequence(
                scenario_id=f"traffic_light_{i}",
                object_type=ObjectType.TRAFFIC_LIGHT,
                steps=steps,
                total_reward=sum(s.reward for s in steps),
                success=True,
            ))
        return seqs

    def gen_mount_dismount(
        self,
        mount_type: ObjectType = ObjectType.HORSE,
        n: int = 40,
    ) -> List[InteractionSequence]:
        ik_cfg = get_ik_config(mount_type)
        action_enter  = InteractionAction.MOUNT_ANIMAL
        action_exit   = InteractionAction.DISMOUNT_ANIMAL
        if mount_type == ObjectType.BICYCLE:
            action_enter = InteractionAction.MOUNT_BICYCLE
            action_exit  = InteractionAction.DISMOUNT_ANIMAL

        seqs = []
        for i in range(n):
            dist  = self._rng.uniform(2.0, 6.0)
            steps = []
            prev  = dist

            for t in range(8):
                dist -= self._rng.uniform(0.3, 0.8)
                dist  = max(0.0, dist)
                action = action_enter if dist < ik_cfg.approach_distance else InteractionAction.MOVE_CIVILIAN
                ctx    = InteractionContext(
                    object_type=mount_type, object_distance=dist,
                    vehicle_available=True, seat_is_empty=True,
                    last_action=InteractionAction.MOVE_CIVILIAN,
                )
                reward = (prev - dist) * R_APPROACH_PROGRESS + (R_VEHICLE_ENTER if action == action_enter else 0.0)
                steps.append(InteractionStep(
                    context=ctx, action=action, reward=reward,
                    done=(action == action_enter),
                    ik_sockets=get_all_sockets(mount_type) if action == action_enter else [],
                ))
                prev = dist
                if action == action_enter:
                    break

            # Dismount
            ctx_dis = InteractionContext(
                object_type=mount_type, object_distance=0.0,
                vehicle_available=True, seat_is_empty=False,
                last_action=action_enter,
            )
            steps.append(InteractionStep(
                context=ctx_dis, action=action_exit,
                reward=R_VEHICLE_EXIT, done=True,
                ik_sockets=get_all_sockets(mount_type),
            ))

            seqs.append(InteractionSequence(
                scenario_id=f"mount_{mount_type.name.lower()}_{i}",
                object_type=mount_type, steps=steps,
                total_reward=sum(s.reward for s in steps), success=True,
            ))
        return seqs

    # ──────────────────────────────────────────────────────────────────────────
    # Generate all
    # ──────────────────────────────────────────────────────────────────────────

    def generate_all(self, scale: float = 1.0) -> List[InteractionSequence]:
        n = lambda base: max(1, int(base * scale))
        all_seqs: List[InteractionSequence] = []
        all_seqs += self.gen_weapon_pickup(n(50))
        all_seqs += self.gen_weapon_aim(n(30))
        all_seqs += self.gen_ball_kick(n(60))
        all_seqs += self.gen_ball_grab(n(40))
        all_seqs += self.gen_ball_push(n(30))
        all_seqs += self.gen_threat_detection(n(80))
        all_seqs += self.gen_vehicle_enter_exit(ObjectType.CAR, n(50))
        all_seqs += self.gen_vehicle_enter_exit(ObjectType.MOTORCYCLE, n(40))
        all_seqs += self.gen_vehicle_enter_exit(ObjectType.BICYCLE, n(30))
        all_seqs += self.gen_traffic_light_response(n(60))
        all_seqs += self.gen_mount_dismount(ObjectType.HORSE, n(40))
        all_seqs += self.gen_mount_dismount(ObjectType.BICYCLE, n(30))
        logger.info(
            f"ScenarioGenerator | {len(all_seqs)} sequências geradas "
            f"| scale={scale}"
        )
        return all_seqs
