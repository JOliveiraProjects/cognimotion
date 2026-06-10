"""
motion_inference_service.py
============================
Servidor TCP assíncrono — paradigma DreamerV3.

Fluxo de inferência (por request MSG_MOTION_REQUEST):
  PoseFrame → PoseEncoder → obs_enc (256-d)
  → RSSM.observe(h, z, a, obs_enc) → (h_new, z_new)
  → Policy.get_action(h_new, z_new)  → action_idx
  → ActionExecutor.decode(action_idx) → (direction, speed)
  → build_motion_action(...) → MSG_MOTION_ACTION enviado ao UE5

O MotionMemoryBank é usado APENAS para recompensa intrínseca (curiosidade).
O FAISS nunca é consultado na rota de inferência principal.

Background:
  - WorldModelTrainerThread: treina RSSM + Policy via imaginação
  - DreamerProcess: processo dedicado para treinamento pesado (opcional)
  - ContinuousTrainer: mantém PoseEncoder (VAE) atualizado
  - LLMProcess: hints de estilo via GPT-2 (opcional)
"""
from __future__ import annotations
import threading

import asyncio
import logging
import multiprocessing as mp
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from config import MotionIntelligenceConfig, DEFAULT_CONFIG
from data.pose_frame import Trajectory, TrajectorySample
from learning.online_imitation_learner import OnlineImitationLearner
from learning.continuous_trainer import ContinuousTrainer
from learning.policy_registry import PolicyRegistry
from learning.reward_normalizer import RewardNormalizer
from learning.world_model_trainer import WorldModelTrainerThread
from memory.motion_memory_bank import MotionMemoryBank
from memory.semantic_memory import SemanticMemory, Relations
from memory.episodic_memory import EpisodicMemory
from memory.intrinsic_reward import IntrinsicRewardModule
from multiagent.session_registry import SessionRegistry
from protocol.binary_protocol import (
    HEADER_SIZE, parse_handshake, parse_motion_request,
    parse_pose_frame, build_handshake_ack, build_motion_response,
    MSG_MOTION_REQUEST, MSG_POSE_FRAME, MSG_PING, MSG_PONG,
    MSG_LEADER_SEQUENCE, MSG_AUTONOMOUS_REQUEST,
    MSG_PERCEPTION, MSG_TEACH,
    parse_leader_sequence, parse_autonomous_request, build_motion_action,
    parse_perception, parse_teach,
    _parse_header, _build_frame,
)
from planning.policy import Policy
from planning.action_executor import ActionExecutor
from planning.reactive_decision import ReactiveDecisionLayer, ReactiveConfig
from planning.uncertainty_controller import UncertaintyController
from world_model.world_model import WorldModel
from world_model.dreamer_trainer import DreamerTrainer, dreamer_worker_process
from runtime.npc_session import NPCSessionManager
from runtime.sequence_buffer import SequenceBuffer
from learning.inverse_dynamics import label_sequence
from encoding.perception_features import augment_obs, PERCEPTION_DIM
from world_model.pose_decoder import bones_to_target_tensor
from llm.llm_interface import LLMRequest, llm_worker_process
from curriculum.expected_learning_value import ELVEstimator, TrainingRecord
from behavior.behavior_control import NPCBehaviorController

logger = logging.getLogger("motion_inference")

# ── Debug toggle: CMI_DEBUG=1 no ambiente, ou chamar set_debug(True) em runtime ──
import os as _os
_DEBUG = _os.environ.get("CMI_DEBUG", "0") == "1"

def _dbg(cat: str, msg: str) -> None:
    if _DEBUG: logger.info(f"[DBG][{cat}] {msg}")

def set_debug(enabled: bool) -> None:
    """Ativa/desativa logs de debug: set_debug(True) no console Python."""
    global _DEBUG
    _DEBUG = enabled
    logger.info(f"[CMI] Debug: {'ON' if enabled else 'OFF'}")


# ──────────────────────────────────────────────────────────────────────────────
# ClientSession — uma conexão UE5
# ──────────────────────────────────────────────────────────────────────────────



def _direction_to_quat_xyzw(fwd: np.ndarray) -> np.ndarray:
    """
    Converte vetor de direção normalizado para quaternion UE5 (XYZW).

    UE5 usa coordenadas mão-direita com Z-up. Rotação no plano XY em torno de Z:
      ângulo θ = atan2(fwd.y, fwd.x)
      q = [0, 0, sin(θ/2), cos(θ/2)]   (XYZW)

    Quaternion identidade [0, 0, 0, 1] = sem rotação = forward em +X.

    CORREÇÃO CP-04: o código anterior usava [fwd.x, fwd.y, fwd.z, 0.0] que
    NÃO é um quaternion válido (w=0 implica rotação de 180° pura, não a direção
    desejada). Este função produz o quaternion correto para qualquer direção 2D.
    """
    # Extrai apenas componentes XY para rotação em torno de Z (plano de navegação)
    dx, dy = float(fwd[0]), float(fwd[1])
    norm_2d = np.sqrt(dx * dx + dy * dy)
    if norm_2d < 1e-6:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # identidade
    theta = np.arctan2(dy, dx)
    half  = theta * 0.5
    return np.array([0.0, 0.0, np.sin(half), np.cos(half)], dtype=np.float32)


def _build_action_trajectory(direction: list, speed: float, n_samples: int = 6, dt: float = 0.1) -> Trajectory:
    """
    Gera trajetória sintética a partir de ação RL (direction + speed).
    Usada pelo build_motion_response para preencher RefinedTrajectory.

    CORREÇÃO CP-04: facing agora é um quaternion UE5 (XYZW) válido calculado
    a partir da direção de movimento, não um vetor de direção com w=0 que
    representava uma rotação de 180° em vez da orientação desejada.
    """
    samples = []
    pos = np.zeros(3, dtype=np.float32)
    dir_arr = np.array(direction, dtype=np.float32)
    norm = float(np.linalg.norm(dir_arr))
    if norm > 1e-6:
        dir_arr = dir_arr / norm

    facing = _direction_to_quat_xyzw(dir_arr)  # quaternion válido, calculado uma vez

    for i in range(n_samples):
        t = (i + 1) * dt
        p = pos + dir_arr * speed * t
        vel = dir_arr * speed
        samples.append(TrajectorySample(
            position=p.copy(), linear_velocity=vel.copy(),
            angular_velocity=np.zeros(3, dtype=np.float32),
            facing=facing.copy(), time_in_seconds=t, speed=float(speed),
        ))
    return Trajectory(samples=samples)

class ClientSession:
    def __init__(
        self,
        reader:    asyncio.StreamReader,
        writer:    asyncio.StreamWriter,
        service:  "MotionInferenceService",
        session_id: str,
    ) -> None:
        self.reader             = reader
        self.writer             = writer
        self.service            = service
        self.session_id         = session_id
        self.requests_processed = 0
        self._obs_dim:  int     = 0
        self._emb_dim:  int     = 0
        # Percepção mais recente enviada pelo NPC (lista de entidades) e
        # vocabulário ensinado pelo líder. Usados pela recompensa e pela
        # camada de decisão para agir com base no ambiente.
        self.last_perception: list = []
        self.action_vocabulary: dict = {}   # action_index → label
        self.current_verb: str = "idle"

    async def handle(self) -> None:
        try:
            await self._do_handshake()
            while True:
                await self._process_message()
        except asyncio.IncompleteReadError:
            pass
        except ConnectionResetError:
            pass
        except Exception as exc:
            logger.warning(f"[{self.session_id}] erro: {exc}", exc_info=False)
        finally:
            try:
                self.writer.close()
                await self.writer.wait_closed()
            except Exception:
                pass
            self.service.session_registry.remove_session(self.session_id)
            logger.info(
                f"[{self.session_id}] desconectado "
                f"| reqs={self.requests_processed}"
            )

    async def _do_handshake(self) -> None:
        raw = await self.reader.readexactly(HEADER_SIZE)
        hdr = _parse_header(raw)
        if hdr is None:
            raise ConnectionResetError("handshake header inválido")
        payload_size = hdr["payload_size"]
        body = await self.reader.readexactly(payload_size) if payload_size else b""
        full = raw + body
        hs = parse_handshake(full)
        if hs is None:
            raise ConnectionResetError("handshake payload inválido")
        # BUG-1 FIX: parse_handshake() returns Tuple[int,int], not dict
        self._obs_dim, self._emb_dim = hs
        # BUG-1 FIX: build_handshake_ack() takes no arguments
        ack = build_handshake_ack()
        self.writer.write(ack)
        await self.writer.drain()
        logger.info(
            f"[{self.session_id}] handshake OK "
            f"| obs_dim={self._obs_dim} | emb_dim={self._emb_dim}"
        )

    async def _process_message(self) -> None:
        raw_hdr = await self.reader.readexactly(HEADER_SIZE)
        hdr     = _parse_header(raw_hdr)
        if hdr is None:
            return
        payload_size = hdr["payload_size"]
        body = await self.reader.readexactly(payload_size) if payload_size else b""
        full = raw_hdr + body
        mt   = hdr["msg_type"]
        sid  = hdr["seq_id"]

        if mt == MSG_MOTION_REQUEST:
            await self._handle_motion_request(full, sid)
        elif mt == MSG_AUTONOMOUS_REQUEST:
            await self._handle_autonomous_request(full, sid)
        elif mt == MSG_LEADER_SEQUENCE:
            await self._handle_leader_sequence(full)
        elif mt == MSG_POSE_FRAME:
            # BUG-5 FIX: parse_pose_frame(), not parse_motion_request()
            pose_frame = parse_pose_frame(full)
            if pose_frame is not None:
                asyncio.ensure_future(
                    self.service.store_experience(
                        self.session_id,
                        {"pose_frame": pose_frame, "blackboard": {}},
                    )
                )
        elif mt == MSG_PERCEPTION:
            self._handle_perception(full)
        elif mt == MSG_TEACH:
            self._handle_teach(full)
        elif mt == MSG_PING:
            self.writer.write(_build_frame(MSG_PONG, sid, b""))
            await self.writer.drain()
            self.service.session_registry.heartbeat(self.session_id)

    # ──────────────────────────────────────────────────────────────────────────
    # MSG_PERCEPTION → guarda entidades percebidas na sessão (usadas pela
    # recompensa de tarefa e pela camada de decisão).
    # ──────────────────────────────────────────────────────────────────────────
    def _handle_perception(self, data: bytes) -> None:
        parsed = parse_perception(data)
        if not parsed:
            return
        self.last_perception = parsed.get("entities", [])
        # Repassa para o serviço (a recompensa de tarefa lê daqui).
        self.service.update_perception(self.session_id, self.last_perception)
        if self.last_perception:
            top = self.last_perception[0]
            logger.debug(
                f"[{self.session_id}] PERCEPÇÃO: {len(self.last_perception)} "
                f"entidade(s). Mais próxima: {top['category_name']} "
                f"dist={top['distance']:.0f} reação_sugerida={top['reaction_name']} "
                f"ameaça={top['threat_weight']:.2f}"
            )

    # ──────────────────────────────────────────────────────────────────────────
    # MSG_TEACH → registra o vocabulário de ações ensinado pelo líder.
    # ──────────────────────────────────────────────────────────────────────────
    def _handle_teach(self, data: bytes) -> None:
        parsed = parse_teach(data)
        if not parsed:
            return
        vocab = parsed.get("vocabulary", [])
        self.action_vocabulary = {
            v["action_index"]: v["label"] for v in vocab if v["action_index"] >= 0
        }
        self.current_verb = parsed.get("current_verb_name", "idle")
        self.service.update_vocabulary(self.session_id, parsed)

        # APRENDIZADO POR DEMONSTRAÇÃO: se o líder está rotulando emoção+ação
        # agora, registra a tripla (percepção atual, emoção, ação) para treino.
        demo_emotion = parsed.get("current_emotion_name", "")
        demo_action = parsed.get("current_action_index", -1)
        if demo_emotion and demo_action is not None and int(demo_action) >= 0:
            from encoding.perception_features import perception_features
            pvec = perception_features(self.last_perception)
            self.service.demo_learner.add_demonstration(
                pvec, demo_emotion, int(demo_action))
        if not getattr(self, "_teach_logged", False):
            self._teach_logged = True
            verbs = ", ".join(v["verb_name"] for v in vocab)
            logger.info(
                f"[{self.session_id}] VOCABULÁRIO ensinado pelo líder "
                f"({len(vocab)} verbos): {verbs}"
            )

    # ──────────────────────────────────────────────────────────────────────────
    # MSG_MOTION_REQUEST → RL inference → MSG_MOTION_ACTION
    # ──────────────────────────────────────────────────────────────────────────

    async def _handle_motion_request(self, data: bytes, seq_id: int) -> None:
        t0      = time.perf_counter()
        req_dict = parse_motion_request(data)
        if req_dict is None:
            return

        pose_frame = req_dict.get("pose_frame")
        if pose_frame is None:
            return

        # Extrai NPCId do blackboard (0 = sessão TCP pura)
        bb     = req_dict.get("blackboard", {})
        npc_id = int(bb.get("npc_id", 0)) or hash(self.session_id) & 0x7FFFFFFF

        loop = asyncio.get_running_loop()

        # 1. Codifica pose → obs_enc (256-d)
        obs_enc, conf = await loop.run_in_executor(
            self.service.executor,
            self.service.learner.pose_encoder.encode_frame,
            pose_frame,
            self.service.config.device,
        )

        # 1b. AUMENTA o obs com a percepção (inimigo/ameaça/refém/direção).
        # É isto que faz a política ENXERGAR o ambiente e permite a decisão
        # de combate EMERGIR do RL, em vez de ser regra fixa.
        obs_enc = augment_obs(obs_enc, self.last_perception)

        # 2. Inferência RL: RSSM → Policy → ActionExecutor
        action_idx, direction, speed, entropy_val = await loop.run_in_executor(
            self.service.executor,
            self.service.rl_inference,
            npc_id, obs_enc, req_dict,
        )

        # ── APRENDIZADO POR DEMONSTRAÇÃO ──────────────────────────────────────
        # Treina as cabeças periodicamente com o que o líder já demonstrou.
        if (self.requests_processed % 10) == 0:
            self.service.demo_learner.train_step(batch_size=64)

        # Se o líder já ensinou o suficiente, o NPC usa o que APRENDEU
        # (percepção→emoção→ação) e generaliza. Caso contrário, segue para a
        # camada reativa (rede de segurança até haver aprendizado).
        learned_emotion = None
        if self.service.demo_learner.n_demonstrations() >= 64 and self.last_perception:
            from encoding.perception_features import perception_features
            pred = self.service.demo_learner.infer(
                perception_features(self.last_perception))
            learned_emotion = pred["emotion"]
            # Só sobrepõe a ação se a confiança for alta (senão deixa a política).
            if pred["action_conf"] >= 0.6:
                action_idx = pred["action_idx"]
                if (self.requests_processed % 30) == 0:
                    logger.info(
                        f"[{self.session_id}] APRENDIDO: emoção='{pred['emotion']}'"
                        f"(conf={pred['emotion_conf']:.2f}) → ação={action_idx}"
                        f"(conf={pred['action_conf']:.2f})")

        # 3. Controle de incerteza
        self.service.uncertainty_controller.update(
            entropy_nats=entropy_val,
            reward=conf,
        )
        mod = self.service.uncertainty_controller.get_action_modification(
            action_idx, speed
        )
        action_idx, speed, direction, _, _ = mod.apply(
            action_idx, speed, direction
        )

        # ── CAMADA REATIVA: estados físicos + reação a ameaça ─────────────────
        # Lê vida/ameaça do blackboard e movement_mode da pose. Pode SOBRESCREVER
        # a ação da política (ex.: morte → idle, queda → fall, fuga → run).
        movement_mode = int(getattr(pose_frame, "movement_mode", 0))
        rdec = self.service.reactive_layer.decide(
            npc_id=npc_id,
            blackboard=bb,
            movement_mode=movement_mode,
            policy_action=int(action_idx),
            perception=self.last_perception,
            profile_name=str(bb.get("profile", bb.get("training_category", ""))),
        )
        physical_state = rdec.physical_state
        if rdec.override:
            action_idx = rdec.action
            # Se morte/queda, zera locomoção (sem deslizar)
            if physical_state in ("dead", "falling", "landing"):
                speed = 0.0
                direction = [0.0, 0.0, 0.0]

        # 4. Recompensa intrínseca (curiosidade via z latente)
        session = self.service.npc_session_manager.get(npc_id)
        if session is not None:
            z_np  = session.z.squeeze(0).cpu().numpy()
            r_int = self.service.intrinsic_reward.compute(z_np)
        else:
            r_int = 0.0

        # 4b. Recompensa de TAREFA (percepção do ambiente) — ponto 3.
        # Faz o NPC decidir com base no que percebe (fugir de ameaça, ir pegar
        # objeto), separada da recompensa de imitação que o world model otimiza.
        r_task, r_breakdown = self.service.compute_task_reward(self.session_id)

        # Registra a recompensa total no NPCSession para o trainer usar.
        r_total = float(r_int) + float(r_task)
        if session is not None and hasattr(session, "record_reward"):
            try:
                session.record_reward(r_total)
            except Exception:
                pass

        # LOG DE RECOMPENSA POR PASSO (ponto 3) — a cada 30 requests (~1x/s a
        # 30Hz) para validar se a recompensa está alinhada ao comportamento.
        if (self.requests_processed % 30) == 0:
            logger.info(
                f"[{self.session_id}] RECOMPENSA passo={self.requests_processed} | "
                f"total={r_total:+.3f} | intrínseca={r_int:+.3f} | "
                f"tarefa={r_task:+.3f} "
                f"(ameaça={r_breakdown['threat_proximity']:+.2f} "
                f"pickup={r_breakdown['pickup_proximity']:+.2f} "
                f"facing={r_breakdown['enemy_facing']:+.2f}) | "
                f"ação={action_idx} verbo={self.current_verb}"
            )

        # ── LLM: hint de estilo de alto nível (ponto 2) ───────────────────────
        # gpt2 no CPU é lento; consultamos de forma esparsa (a cada ~60 passos,
        # ~2s a 30Hz) e fire-and-forget. A resposta volta pelo _llm_response_loop
        # e fica em session.llm_motion_style, lido pela política como hint.
        if (self.requests_processed % 60) == 0 and \
           self.service.llm_request_queue is not None:
            try:
                self.service.llm_request_queue.put_nowait(LLMRequest(
                    session_id=self.session_id,
                    health=float(bb.get("health", 100.0)),
                    stamina=float(bb.get("stamina", 100.0)),
                    alertness=float(bb.get("alertness", 0.0)),
                    fear_level=float(bb.get("fear_level", 0.0)),
                    aggression_level=float(bb.get("aggression_level", 0.0)),
                    current_state=int(bb.get("current_state", 0)),
                ))
            except Exception:
                pass  # fila cheia: ignora este ciclo (não bloqueia inferência)

        # 5. Armazena experiência no SequenceBuffer (não no FAISS)
        asyncio.ensure_future(
            self.service.store_experience(self.session_id, req_dict, obs_enc, action_idx)
        )

        # BUG-2 FIX: UE5 DeserializeResponse() only accepts MSG_MOTION_RESPONSE (0x02).
        # Encode RL action into the response fields:
        #   SelectedStyle = action_idx clamped to ECognitiveMotionStyle (0-8)
        #   Embedding.Values[0:3] = direction, [3] = speed  (read by AnimInstance)
        latency_ms    = (time.perf_counter() - t0) * 1000.0
        selected_style = int(action_idx) % 9  # clamp to ECognitiveMotionStyle

        emb_dim = self.service.config.encoder.embedding_dim
        embedding = np.zeros(emb_dim, dtype=np.float32)
        embedding[0] = float(direction[0])
        embedding[1] = float(direction[1])
        embedding[2] = float(direction[2])
        embedding[3] = float(speed)
        if obs_enc is not None:
            embedding[4:min(emb_dim, 4 + len(obs_enc))] = obs_enc[: emb_dim - 4]

        refined_traj = _build_action_trajectory(direction, speed)

        # ── POSES GERADAS PELO WORLD MODEL ────────────────────────────────────
        # O PoseDecoder gera a animação a partir do latente aprendido. Isso faz
        # o NPC ANIMAR sozinho (não copiar o líder). Se o decoder ainda não foi
        # treinado o bastante (pose vazia), cai no relay dos bones do líder.
        # Sincroniza o gate de geração de pose com o estado do treino. Só quando
        # o modelo amadureceu (pose_ready) é que generate_pose devolve poses;
        # caso contrário o NPC copia o líder (evita "virar bola").
        self.service.pose_generation_ready = bool(
            getattr(self.service.pipeline_stats, "pose_ready", False))

        generated_bones = self.service.generate_pose(npc_id)
        if generated_bones:
            leader_bones = generated_bones
        else:
            leader_bones = (
                self.service.latest_leader_bones
                or getattr(self, "_latest_leader_bones", [])
            )

        wire = build_motion_response(
            seq_id=seq_id,
            embedding=embedding,
            confidence=float(conf),
            refined_trajectory=refined_traj,
            selected_style=selected_style,
            latency_ms=latency_ms,
            valid=conf > 0.1,
            bone_transforms=leader_bones,
            physical_state=physical_state,
        )
        self.writer.write(wire)
        await self.writer.drain()
        self.requests_processed += 1

        # ── PIPELINE: contadores ao vivo + painel periódico transparente ─────
        ps = self.service.pipeline_stats
        ps.inc("responses_sent")
        ps.set(last_confidence=float(conf), last_latency_ms=float(latency_ms))
        try:
            ps.set(active_sessions=self.service.session_registry.session_count())
        except Exception:
            pass
        self.service.pipeline_logger.maybe_emit()

        # ── TELEMETRIA: o que o Python recebeu e o que enviou ─────────────────
        mon = getattr(self.service.continuous_trainer, "monitor", None)
        if mon is not None:
            mon.record_confidence(float(conf))
        _dbg("INFER",
             f"RECEBIDO seq={seq_id} | ENVIADO acao={selected_style} "
             f"conf={conf:.3f} bones={len(leader_bones)} estado={physical_state}")

        # 7. Registra no PolicyRegistry para monitoramento
        reward_signal = float(conf) + r_int
        normed        = self.service.reward_normalizer.update_and_normalize(reward_signal)
        self.service.policy_registry.record_step_reward(normed, n_steps=1)

        # 8. ELV tracking
        style_names = [
            "neutral", "aggressive", "relaxed", "injured",
            "fatigued", "stealth", "military", "civilian", "criminal",
        ]
        style_name = style_names[min(action_idx, len(style_names) - 1)]
        self.service.elv_estimator.record_training(TrainingRecord(
            skill_name=style_name,
            context=str(bb.get("current_state", 0)),
            reward_before=max(0.0, float(conf) - 0.1),
            reward_after=float(conf),
            n_steps=1,
        ))

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"[{self.session_id}] RL | seq={seq_id} | npc={npc_id} "
                f"| action={action_idx} | speed={speed:.2f} "
                f"| conf={conf:.3f} | lat={latency_ms:.1f}ms"
            )

    # ──────────────────────────────────────────────────────────────────────────
    # MSG_LEADER_SEQUENCE → SequenceBuffer + DreamerProcess queue
    # ──────────────────────────────────────────────────────────────────────────

    async def _handle_leader_sequence(self, data: bytes) -> None:
        payload = parse_leader_sequence(data)
        if not payload or not payload.get("raw_frames"):
            return

        leader_id   = payload["leader_npc_id"]
        follower_id = payload["follower_npc_id"]
        group_key   = f"{leader_id}_{follower_id}"
        raw_frames  = payload["raw_frames"]

        loop = asyncio.get_running_loop()
        embeddings = []

        pose_frames = []
        for raw_frame in raw_frames:
            try:
                pose_frame = parse_pose_frame(raw_frame)
                if pose_frame is None:
                    continue

                # Armazena bone transforms no SERVIÇO (não na sessão).
                # Isso garante que bones persistam mesmo após reconexão do NPC.
                if pose_frame.bone_transforms:
                    mode_key = getattr(self, "_current_mode_key", "Idle|Default")
                    self.service.leader_bones_by_mode[mode_key] = pose_frame.bone_transforms
                    self.service.latest_leader_bones = pose_frame.bone_transforms
                    self._latest_leader_bones = pose_frame.bone_transforms
                    self.service.pipeline_stats.inc("frames_received")
                    # Em Observing o ponto de inferência não é alcançado, então
                    # emitimos o painel transparente AQUI também (recepção do líder).
                    self.service.pipeline_logger.maybe_emit()
                    _dbg("BONES", f"líder: {len(pose_frame.bone_transforms)} bones recebidos | sid={self.session_id[:12]}")
                    logger.debug(
                        f"[{self.session_id}] leader bones atualizado: "
                        f"{len(pose_frame.bone_transforms)} bones | "
                        f"locomoção={mode_key} (coletando p/ treino — modo Observing)"
                    )

                emb, _ = await loop.run_in_executor(
                    self.service.executor,
                    self.service.learner.pose_encoder.encode_frame,
                    pose_frame,
                    self.service.config.device,
                )
                # AUMENTA com a percepção (mesma dim da inferência: 256+PERCEPTION_DIM).
                # Sem isto o obs de treino teria dim diferente do de inferência.
                emb = augment_obs(emb, self.last_perception)
                embeddings.append(emb)
                pose_frames.append(pose_frame)
            except Exception as exc:
                logger.debug(f"[{self.session_id}] encode leader frame: {exc}")

        if len(embeddings) < 2:
            return

        # ── CRÍTICO PARA AUTONOMIA: rotula a demonstração com ações + recompensas ──
        # Sem isso o actor-critic nunca aprende (action=0/reward=0 → gradiente nulo).
        # label_sequence infere a ação por dinâmica inversa (velocidade local) e
        # calcula recompensa de imitação por frame.
        obs_seq = np.stack(embeddings, axis=0)
        action_dim = self.service.config.actor_critic.action_dim
        action_seq, reward_seq = label_sequence(pose_frames, action_dim=action_dim)
        done_seq = np.zeros(len(embeddings), dtype=bool)
        done_seq[-1] = True

        # Garante alinhamento de tamanhos (encode pode pular frames inválidos)
        T = len(embeddings)
        action_seq = action_seq[:T]
        reward_seq = reward_seq[:T]

        # ── LIGA A PERCEPÇÃO AO TREINO ─────────────────────────────────────────
        # Soma a recompensa de TAREFA (fugir de ameaça / ir a pickup) ao reward
        # de imitação. Assim a política de locomoção aprende POSICIONAMENTO em
        # relação ao ambiente, não só a copiar o movimento. A percepção é um
        # instantâneo da sessão; aplica-se uniformemente sobre a janela.
        r_task, _ = self.service.compute_task_reward(self.session_id)
        if r_task != 0.0:
            reward_seq = reward_seq + np.float32(r_task)

        # ── ALVO DO POSE DECODER: poses reais dos 89 bones do líder ───────────
        # É isto que ensina o world model a GERAR a animação (não só a ação).
        num_bones = getattr(self.service.world_model, "num_bones", 89)
        pose_targets = []
        for pf in pose_frames[:T]:
            t = bones_to_target_tensor(pf.bone_transforms, num_bones, device="cpu")
            pose_targets.append(t.cpu().numpy())
        pose_seq = np.stack(pose_targets, axis=0).astype(np.float32) if pose_targets else None

        self.service.sequence_buffer.add_sequence(
            obs_seq, action_seq, reward_seq, done_seq, group_key, pose_seq=pose_seq
        )

        # Telemetria de aprendizado: contexto e contagem do que está chegando.
        # Alimenta AMBOS os monitores: o do trainer de VAE e — principalmente —
        # o do WorldModelTrainer, que é quem treina o DreamerV3.
        for _trainer_attr in ("continuous_trainer", "wm_trainer_thread"):
            _tr = getattr(self.service, _trainer_attr, None)
            _mon = getattr(_tr, "monitor", None)
            if _mon is not None:
                _mon.sequences_received += 1
                _mon.frames_received += int(T)
                _mon.set_context(training=group_key, mode="Observing")

        if _DEBUG:
            import numpy as _np
            acts = action_seq.argmax(axis=1)
            uniq, cnts = _np.unique(acts, return_counts=True)
            dist = ", ".join(f"a{int(u)}={int(c)}" for u, c in zip(uniq, cnts))
            _dbg("LABEL", f"seq rotulada | frames={T} | ações[{dist}] | reward_médio={float(reward_seq.mean()):.2f}")

        if self.service.dreamer_sequence_queue is not None:
            q = self.service.dreamer_sequence_queue
            payload = (obs_seq, action_seq, reward_seq, done_seq, group_key)
            # Contabiliza no painel transparente (corrige o "Sequências p/ treino: 0").
            self.service.pipeline_stats.inc("sequences_buffered")
            try:
                q.put_nowait(payload)
            except Exception:
                # Fila cheia: o trainer consome mais devagar (~2s/step) do que
                # a recepção produz (~30Hz). Em vez de jogar fora o dado NOVO,
                # descartamos o MAIS ANTIGO e inserimos o novo — assim o treino
                # sempre usa as poses mais recentes do líder. O trainer reamostra
                # o buffer, então não precisa de todas as sequências.
                try:
                    _ = q.get_nowait()        # remove a mais antiga
                except Exception:
                    pass
                try:
                    q.put_nowait(payload)     # insere a nova
                except Exception:
                    # Ainda cheia (corrida com outra thread) — só então loga e segue.
                    logger.debug("[TREINO] fila saturada; sequência mais recente adiada.")

        logger.debug(
            f"[{self.session_id}] LeaderSeq | group={group_key} "
            f"| frames={len(embeddings)}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # MSG_AUTONOMOUS_REQUEST → RSSM prior → Policy → MSG_MOTION_ACTION
    # ──────────────────────────────────────────────────────────────────────────

    async def _handle_autonomous_request(self, data: bytes, seq_id: int) -> None:
        t0  = time.perf_counter()
        req = parse_autonomous_request(data)
        if not req:
            return

        npc_id     = req.get("npc_id", 0)
        pose_frame = req.get("pose_frame")
        loop       = asyncio.get_running_loop()

        if pose_frame is not None:
            obs_enc, conf = await loop.run_in_executor(
                self.service.executor,
                self.service.learner.pose_encoder.encode_frame,
                pose_frame,
                self.service.config.device,
            )
        else:
            obs_enc = np.zeros(
                self.service.config.encoder.embedding_dim, dtype=np.float32
            )
            conf = 0.0

        # AUMENTA com percepção (mesma dim da inferência principal).
        obs_enc = augment_obs(obs_enc, self.last_perception)

        action_idx, direction, speed, entropy_val = await loop.run_in_executor(
            self.service.executor,
            self.service.rl_inference,
            npc_id, obs_enc, req,
        )

        # CORREÇÃO CP-05: _handle_motion_request passa reward=conf mas esta função
        # passava apenas entropy_val sem reward=, perdendo metade da atualização
        # do UncertaintyController. Adicionado reward=conf para consistência.
        self.service.uncertainty_controller.update(
            entropy_nats=entropy_val,
            reward=conf,
        )
        mod = self.service.uncertainty_controller.get_action_modification(
            action_idx, speed
        )
        action_idx, speed, direction, _, _ = mod.apply(action_idx, speed, direction)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        wire = build_motion_action(
            seq_id=seq_id, npc_id=npc_id,
            action_idx=action_idx, direction=direction,
            speed=speed, confidence=float(conf),
            latency_ms=latency_ms, valid=conf > 0.1,
        )
        self.writer.write(wire)
        await self.writer.drain()
        self.requests_processed += 1

        logger.debug(
            f"[{self.session_id}] Autonomous | npc={npc_id} "
            f"| action={action_idx} | lat={latency_ms:.1f}ms"
        )

    @staticmethod
    def _dict_to_trajectory(d: dict) -> Trajectory:
        # BUG-6 FIX: TrajectorySample uses linear_velocity and time_in_seconds
        t = Trajectory()
        for sd in d.get("samples", []):
            t.samples.append(TrajectorySample(
                position=np.array(sd.get("position", [0.0, 0.0, 0.0]), dtype=np.float32),
                linear_velocity=np.array(sd.get("linear_velocity", sd.get("velocity", [0.0, 0.0, 0.0])), dtype=np.float32),
                angular_velocity=np.array(sd.get("angular_velocity", [0.0, 0.0, 0.0]), dtype=np.float32),
                facing=np.array(sd.get("facing", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32),
                time_in_seconds=float(sd.get("time_in_seconds", sd.get("time_offset", 0.0))),
                speed=float(sd.get("speed", 0.0)),
            ))
        return t


# ──────────────────────────────────────────────────────────────────────────────
# MotionInferenceService
# ──────────────────────────────────────────────────────────────────────────────

class MotionInferenceService:
    def __init__(self, config: MotionIntelligenceConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        cfg         = config

        # Gate: só gera poses pelo modelo após treino suficiente. Começa em False
        # para que, sem treino, o NPC copie o líder (não vire bola). O trainer
        # liga quando atinge o nº mínimo de steps com pose loss saudável.
        self.pose_generation_ready = False

        # ── Log de pipeline transparente (estado ao vivo do servidor) ────────
        from runtime.pipeline_logger import PipelineStats, PipelineLogger
        self.pipeline_stats = PipelineStats()
        self.pipeline_logger = PipelineLogger(self.pipeline_stats, interval_s=5.0)

        # ── Encoders + VAE learner ───────────────────────────────────────────
        # OnlineImitationLearner accepts (encoder_config, learning_config, device).
        # memory_config is handled internally by the learner — not an init parameter.
        self.learner = OnlineImitationLearner(
            encoder_config=cfg.encoder,
            learning_config=cfg.learning,
            device=cfg.device,
        )

        # ── World Model (RSSM + decoder) ─────────────────────────────────────
        wm_cfg = cfg.world_model
        combined_dim = (
            wm_cfg.rssm_num_categories * wm_cfg.rssm_category_dim
            + wm_cfg.rssm_hidden_dim
        )
        self.world_model = WorldModel(
            obs_enc_dim=cfg.encoder.embedding_dim + PERCEPTION_DIM,
            action_dim=cfg.actor_critic.action_dim,
            hidden_dim=wm_cfg.rssm_hidden_dim,
            num_categories=wm_cfg.rssm_num_categories,
            category_dim=wm_cfg.rssm_category_dim,
            free_nats=wm_cfg.rssm_free_nats,
            kl_balance=wm_cfg.rssm_kl_balance,
            unimix=wm_cfg.rssm_unimix,
            use_block_gru=wm_cfg.use_block_gru,
            n_blocks=wm_cfg.n_blocks,
            num_bones=getattr(wm_cfg, "num_bones", 89),
        )

        # ── Policy (Actor + Critic) ──────────────────────────────────────────
        self.policy = Policy(
            combined_dim=combined_dim,
            action_dim=cfg.actor_critic.action_dim,
            hidden=256,
        )
        # Lock compartilhado: protege Policy.forward() contra race condition
        # entre WorldModelTrainerThread (escrita) e rl_inference (leitura)
        self.model_lock: threading.RLock = threading.RLock()

        # ── Action decoder ───────────────────────────────────────────────────
        self.action_executor       = ActionExecutor(action_dim=cfg.actor_critic.action_dim)
        self.uncertainty_controller = UncertaintyController(enabled=True)
        # Camada reativa: estados físicos (vida/morte/queda/natação) +
        # reação a ameaça. Roda antes de aceitar a ação da política.
        self.reactive_layer = ReactiveDecisionLayer(ReactiveConfig())

        # Aprendizado por demonstração: o líder rotula emoção+ação por Blueprint
        # e este learner aprende percepção→emoção→ação (generaliza fora da demo).
        from learning.demonstration_learning import DemonstrationLearner
        self.demo_learner = DemonstrationLearner(device=str(cfg.device))

        # ── Per-NPC state manager ────────────────────────────────────────────
        npc_cfg = cfg.npc_session
        self.npc_session_manager = NPCSessionManager(
            max_sessions=npc_cfg.max_sessions,
            timeout_s=npc_cfg.timeout_s,
            hidden_dim=wm_cfg.rssm_hidden_dim,
            stochastic_dim=wm_cfg.rssm_num_categories * wm_cfg.rssm_category_dim,
            device=cfg.device,
        )

        # ── Sequence buffer para treino RSSM ─────────────────────────────────
        self.sequence_buffer = SequenceBuffer(
            capacity=wm_cfg.sequence_buffer_capacity,
            obs_dim=cfg.encoder.embedding_dim + PERCEPTION_DIM,
            action_dim=cfg.actor_critic.action_dim,
            seq_len=wm_cfg.seq_len,
        )

        # ── Intrinsic reward (curiosidade via z latente) ─────────────────────
        self.intrinsic_reward = IntrinsicRewardModule(
            embedding_dim=wm_cfg.rssm_num_categories * wm_cfg.rssm_category_dim,
            beta=0.05,
            beta_decay=0.9999,
            k_neighbors=5,
        )

        # ── MotionMemoryBank — apenas para diagnóstico/fallback ──────────────
        self.memory_bank = MotionMemoryBank(
            config=cfg.memory,
            embedding_dim=cfg.encoder.embedding_dim,
        )

        # ── Memórias semântica e episódica ───────────────────────────────────
        self.semantic_memory = SemanticMemory()
        self.episodic_memory = EpisodicMemory(embedding_dim=cfg.encoder.embedding_dim)

        # ── Learner support stack ────────────────────────────────────────────
        from world_model.skeleton_profile import profile_for
        _skel_profile = profile_for(getattr(self.world_model, "num_bones", 89), "Default")
        self.policy_registry   = PolicyRegistry(
            save_dir=cfg.checkpoint_dir,
            min_reward_threshold=-5.0,
            keep_last_n=5,
            category="Default",                       # treino acumulativo por categoria
            skeleton_signature=_skel_profile.signature(),
        )
        self.reward_normalizer = RewardNormalizer()
        self.continuous_trainer = ContinuousTrainer(learner=self.learner)

        # ── DreamerTrainer (para WorldModel + Policy) ────────────────────────
        self.dreamer_trainer = DreamerTrainer(
            world_model=self.world_model,
            actor_critic=self.policy,
            sequence_buffer=self.sequence_buffer,
            policy_registry=self.policy_registry,
            config=cfg,
            device=cfg.device,
            model_lock=self.model_lock,
        )
        self.wm_trainer_thread = WorldModelTrainerThread(
            dreamer_trainer=self.dreamer_trainer,
            sequence_buffer=self.sequence_buffer,
            config=cfg,
            train_interval_s=wm_cfg.train_interval_s,
            pipeline_stats=self.pipeline_stats,
        )

        # ── Session / NPC registry ───────────────────────────────────────────
        self.session_registry = SessionRegistry(max_sessions=cfg.server.max_clients)
        self.elv_estimator    = ELVEstimator()
        self._behavior_controllers: Dict[str, NPCBehaviorController] = {}

        # ── Bone transforms persistentes (nível de serviço) ──────────────────
        # _latest_leader_bones estava no ClientSession (self) e era perdido
        # a cada reconexão (novo ClientSession = bones vazios = NPC para).
        # Armazenando no serviço, os bones persistem entre sessões.
        self.latest_leader_bones:    list = []
        self.leader_bones_by_mode:   dict = {}

        # ── Percepção e vocabulário por sessão ───────────────────────────────
        # Preenchidos por MSG_PERCEPTION e MSG_TEACH. A recompensa de tarefa
        # (compute_task_reward) e a camada de decisão leem daqui para agir com
        # base no ambiente, não só na dinâmica do corpo.
        self.perception_by_session: Dict[str, list] = {}
        self.vocabulary_by_session: Dict[str, dict] = {}

        # ── IPC → DreamerProcess ─────────────────────────────────────────────
        self.dreamer_sequence_queue: Optional[mp.Queue] = None
        self._dreamer_proc:          Optional[mp.Process] = None

        # ── LLM queue ────────────────────────────────────────────────────────
        self.llm_request_queue:  Optional[mp.Queue] = None
        self.llm_response_queue: Optional[mp.Queue] = None
        self._llm_proc:          Optional[mp.Process] = None

        # ── Asyncio handles ──────────────────────────────────────────────────
        self._server:       Optional[asyncio.AbstractServer] = None
        self._train_task:   Optional[asyncio.Task] = None
        self._stale_task:   Optional[asyncio.Task] = None
        self._llm_task:     Optional[asyncio.Task] = None
        self._running:      bool = False
        self.executor:      Optional[ThreadPoolExecutor] = None

    # ──────────────────────────────────────────────────────────────────────────
    # RL Inference (chamado via executor — thread-safe, não bloqueia event loop)
    # ──────────────────────────────────────────────────────────────────────────

    def generate_pose(self, npc_id: int) -> list:
        """
        Decodifica o latente atual do NPC → poses dos 89 bones (animação gerada
        pelo world model). Usado no modo Inferring para o NPC ANIMAR sozinho,
        sem copiar o líder em tempo real. Retorna lista de dicts de bone ou [].

        GATE DE QUALIDADE: só gera poses se o modelo tiver treinado o suficiente
        (pose_generation_ready). Sem isso, um modelo pouco treinado devolveria
        poses degeneradas (bones em zero → "vira bola") que sobrescreveriam o
        relay do líder. Abaixo do limiar, retorna [] e o NPC copia o líder.
        """
        if not getattr(self, "pose_generation_ready", False):
            return []
        session = self.npc_session_manager.get(npc_id)
        if session is None or getattr(session, "last_combined", None) is None:
            return []
        try:
            with self.model_lock:
                bones = self.world_model.pose_decoder.decode_to_transforms(
                    session.last_combined.to(self.config.device)
                )
            return bones
        except Exception as exc:
            logger.debug(f"generate_pose falhou: {exc}")
            return []

    def rl_inference(
        self,
        npc_id:   int,
        obs_enc:  np.ndarray,
        req_dict: dict,
    ) -> tuple[int, list, float, float]:
        """
        Inferência RL completa:
          1. obs_enc → RSSM.observe(h, z, a_prev, obs_enc) → (h_new, z_new)
          2. Policy.get_action(h_new, z_new) → action_idx
          3. ActionExecutor.decode(action_idx) → (idx, direction, speed)
          4. NPCSessionManager.update_state(npc_id, h_new, z_new)

        Retorna (action_idx, direction, speed, entropy_nats).
        """
        session = self.npc_session_manager.get_or_create(npc_id)
        dev     = self.config.device

        obs_t   = torch.from_numpy(obs_enc.astype(np.float32)).unsqueeze(0).to(dev)
        a_prev  = torch.zeros(1, self.config.actor_critic.action_dim, device=dev)

        # Usa last_action para a_prev (one-hot)
        if session.last_action > 0:
            a_prev[0, min(session.last_action, self.config.actor_critic.action_dim - 1)] = 1.0

        with self.model_lock:
            with torch.no_grad():
                h_new, z_new, _, _, _, _ = self.world_model.rssm.forward(
                    session.h, session.z, a_prev, obs_enc=obs_t
                )
                action_t, _, entropy_t = self.policy.act_with_entropy(h_new, z_new)

        action_idx = int(action_t.item())
        entropy    = float(entropy_t.mean().item())

        idx, direction, speed = self.action_executor.decode(action_idx)

        session.update_state(h_new, z_new)
        session.last_action = idx

        # Guarda o latente combinado para o PoseDecoder gerar a animação.
        with self.model_lock:
            with torch.no_grad():
                combined = torch.cat([z_new, h_new], dim=-1)
        session.last_combined = combined.detach()

        # Recompensa intrínseca registrada no EpisodicMemory
        z_np = z_new.squeeze(0).cpu().numpy()
        r_intrinsic = self.intrinsic_reward.compute(z_np)
        if r_intrinsic > 0.1:
            # store_episode expects a list of (embedding, reward, done) tuples
            self.episodic_memory.store_episode(
                episode=[(obs_enc, r_intrinsic, False)],
                session_id=str(npc_id),
                motion_style=idx,
                mean_reward=r_intrinsic,
            )

        return idx, direction, speed, entropy

    # ──────────────────────────────────────────────────────────────────────────
    # Percepção e vocabulário (MSG_PERCEPTION / MSG_TEACH)
    # ──────────────────────────────────────────────────────────────────────────

    def update_perception(self, session_id: str, entities: list) -> None:
        self.perception_by_session[session_id] = entities

    def update_vocabulary(self, session_id: str, teach: dict) -> None:
        self.vocabulary_by_session[session_id] = teach

    def get_perception(self, session_id: str) -> list:
        return self.perception_by_session.get(session_id, [])

    def compute_task_reward(self, session_id: str) -> Tuple[float, dict]:
        """
        Recompensa de TAREFA baseada na percepção do ambiente — separada da
        recompensa de imitação (pose) que o world model otimiza. Esta é a
        recompensa que faz o NPC *decidir* em vez de só imitar.

        Termos (todos em [-1, +1] aprox., somados):
          - threat_proximity: penaliza estar perto de ameaça (incentiva fugir)
          - pickup_proximity:  premia estar perto de objeto pegável (incentiva pegar)
          - enemy_facing:      penaliza inimigo nas costas (incentiva defender/virar)
        Retorna (reward, breakdown) para logging por passo.
        """
        entities = self.perception_by_session.get(session_id, [])
        breakdown = {"threat_proximity": 0.0, "pickup_proximity": 0.0,
                     "enemy_facing": 0.0}
        if not entities:
            return 0.0, breakdown

        for e in entities:
            dist = max(float(e.get("distance", 1e6)), 1.0)
            prox = float(np.clip(600.0 / dist, 0.0, 1.0))  # 1 perto, →0 longe
            cat  = e.get("category_name", "unknown")
            disp = e.get("disposition_name", "neutral")
            threat = float(e.get("threat_weight", 0.0))

            if cat == "hazard" or disp == "enemy" or threat > 0.5:
                breakdown["threat_proximity"] -= prox * max(threat, 0.5)
                # inimigo atrás (DirX negativo no espaço local = atrás)
                if e.get("direction", [0, 0, 0])[0] < 0.0:
                    breakdown["enemy_facing"] -= 0.3 * prox
            elif cat in ("pickup", "weapon"):
                breakdown["pickup_proximity"] += prox * 0.5

        # Limita cada termo para não dominar a recompensa de pose.
        for k in breakdown:
            breakdown[k] = float(np.clip(breakdown[k], -1.0, 1.0))
        total = float(sum(breakdown.values()))
        return total, breakdown

    # ──────────────────────────────────────────────────────────────────────────
    # Experience storage (alimenta SequenceBuffer, não FAISS)
    # ──────────────────────────────────────────────────────────────────────────

    async def store_experience(
        self,
        session_id: str,
        request:    dict,
        obs_enc:    Optional[np.ndarray] = None,
        action_idx: Optional[int]        = None,
    ) -> None:
        pose_frame = request.get("pose_frame")
        if pose_frame is None:
            return

        loop = asyncio.get_running_loop()

        if obs_enc is None:
            try:
                obs_enc, _ = await loop.run_in_executor(
                    self.executor,
                    self.learner.pose_encoder.encode_frame,
                    pose_frame,
                    self.config.device,
                )
            except Exception:
                return
            # Gerado do zero aqui → aumenta com percepção (consistência de dim).
            obs_enc = augment_obs(obs_enc, self.get_perception(session_id))

        act = action_idx if action_idx is not None else 0
        act_oh = np.zeros(self.config.actor_critic.action_dim, dtype=np.float32)
        act_oh[min(act, self.config.actor_critic.action_dim - 1)] = 1.0

        bb       = request.get("blackboard", {})
        reward   = float(bb.get("reward", 0.0))
        done     = bool(bb.get("done", False))
        group_key = f"session_{hash(session_id) & 0xFFFF}"

        # Single-frame sequence (buffer acumula por sessão)
        self.sequence_buffer.add_sequence(
            obs_seq=obs_enc[None, :],
            action_seq=act_oh[None, :],
            reward_seq=np.array([reward], dtype=np.float32),
            done_seq=np.array([done], dtype=bool),
            group_key=group_key,
        )

        # Semântica: associa estado NPC ao estilo de ação
        npc_state = bb.get("current_state", 0)
        self.semantic_memory.learn(
            subject=f"state_{npc_state}",
            relation=Relations.MOTION_STYLE_FOR,
            object_val=act,
            confidence_delta=0.02,
            source=session_id,
        )

        # Treino do VAE (PoseEncoder) — mantido para compatibilidade de embeddings
        self.learner.replay_buffer.add(
            embedding=obs_enc,
            style=act,
            confidence=float(bb.get("confidence", 0.5)),
        )
        if self.learner.replay_buffer.size() % 50 == 0:
            self.continuous_trainer.request_train()

    # ──────────────────────────────────────────────────────────────────────────
    # Background tasks
    # ──────────────────────────────────────────────────────────────────────────

    async def _stale_session_loop(self) -> None:
        while self._running:
            await asyncio.sleep(30.0)
            loop = asyncio.get_running_loop()
            n = await loop.run_in_executor(
                self.executor, self.npc_session_manager.remove_stale
            )
            if n:
                logger.debug(f"StaleLoop | removidas {n} NPC sessions inativas")

    async def _llm_response_loop(self) -> None:
        while self._running:
            await asyncio.sleep(0.1)
            if self.llm_response_queue is None:
                continue
            while not self.llm_response_queue.empty():
                try:
                    resp = self.llm_response_queue.get_nowait()
                    session = self.session_registry.get_session(resp.session_id)
                    if session:
                        session.llm_motion_style = resp.selected_style
                except Exception as e:
                    logger.debug("[STYLE] Falha ao definir motion_style: %s", e)

    async def _client_connected(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        peer       = writer.get_extra_info("peername", ("unknown", 0))
        session_id = f"{peer[0]}:{peer[1]}"
        self.session_registry.create_session(session_id, peer_addr=str(peer))

        bc = NPCBehaviorController(session_id=session_id, behavior_config=self.config.behavior)
        self._behavior_controllers[session_id] = bc

        logger.info(f"Nova conexão: {session_id}")
        client = ClientSession(reader, writer, self, session_id)
        try:
            await client.handle()
        finally:
            self._behavior_controllers.pop(session_id, None)

    # ──────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────────

    def _start_dreamer_process(self) -> None:
        if not self.config.dreamer.enable_dreamer_process:
            return
        self.dreamer_sequence_queue = mp.Queue(
            maxsize=self.config.dreamer.ipc_queue_maxsize
        )
        self._dreamer_proc = mp.Process(
            target=dreamer_worker_process,
            args=(
                self.dreamer_sequence_queue,
                {},
                self.config.production.checkpoint_dir,
                self.config.dreamer.dreamer_device,
            ),
            daemon=True,
            name="DreamerProcess",
        )
        self._dreamer_proc.start()
        logger.info(f"DreamerProcess PID={self._dreamer_proc.pid}")

    def _start_llm_process(self) -> None:
        if not self.config.multiprocess.enable_llm_process:
            return
        self.llm_request_queue  = mp.Queue(maxsize=256)
        self.llm_response_queue = mp.Queue(maxsize=256)
        self._llm_proc = mp.Process(
            target=llm_worker_process,
            args=(
                self.llm_request_queue,
                self.llm_response_queue,
                self.config.llm.model_name,
                self.config.llm.device,
                self.config.llm.cache_dir,
            ),
            daemon=True,
            name="LLMProcess",
        )
        self._llm_proc.start()
        logger.info(f"LLMProcess PID={self._llm_proc.pid}")

    async def start(self) -> None:
        from concurrent.futures import ThreadPoolExecutor as TPE
        workers = max(4, self.config.multiprocess.n_inference_workers * 2)
        self.executor = TPE(max_workers=workers, thread_name_prefix="inference")

        # Inicia processos dedicados
        self._start_dreamer_process()
        self._start_llm_process()

        # Pré-carrega dataset de interações no SequenceBuffer (em background)
        if self.config.dataset.load_on_start:
            from datasets.dataset_registry import DatasetRegistry, DatasetConfig
            _ds_cfg = DatasetConfig(
                scale=self.config.dataset.scale,
                seed=self.config.dataset.seed,
                obs_dim=self.config.encoder.embedding_dim + PERCEPTION_DIM,
                action_dim=self.config.actor_critic.action_dim,
                enable_weapons=self.config.dataset.enable_weapons,
                enable_ball=self.config.dataset.enable_ball,
                enable_threats=self.config.dataset.enable_threats,
                enable_vehicles=self.config.dataset.enable_vehicles,
                enable_mounts=self.config.dataset.enable_mounts,
                enable_traffic=self.config.dataset.enable_traffic,
            )
            _ds_reg = DatasetRegistry(_ds_cfg)
            _ds_reg.load_into_buffer_async(
                self.sequence_buffer,
                obs_dim=self.config.encoder.embedding_dim + PERCEPTION_DIM
            )
            logger.info("DatasetRegistry | carregamento em background iniciado")

        # Inicia WorldModelTrainer thread
        self.wm_trainer_thread.start()

        # Inicia ContinuousTrainer (VAE)
        self.continuous_trainer.start()

        self._running = True
        self._stale_task = asyncio.ensure_future(self._stale_session_loop())
        if self.config.multiprocess.enable_llm_process:
            self._llm_task = asyncio.ensure_future(self._llm_response_loop())

        srv = self.config.server
        self._server = await asyncio.start_server(
            self._client_connected,
            host=srv.host,
            port=srv.port,
            limit=2 ** 20,
        )
        logger.info(
            f"MotionInferenceService | {srv.host}:{srv.port} | "
            f"device={self.config.device} | "
            f"dreamer={'on' if self.config.dreamer.enable_dreamer_process else 'off'}"
        )
        async with self._server:
            await self._server.serve_forever()

    async def stop(self) -> None:
        self._running = False

        # Para tarefas asyncio
        for task in (self._stale_task, self._train_task, self._llm_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Para threads
        self.wm_trainer_thread.stop(timeout=5.0)
        self.continuous_trainer.stop()

        # Para processos filhos
        for proc, queue, name in [
            (self._dreamer_proc, self.dreamer_sequence_queue, "Dreamer"),
            (self._llm_proc,     self.llm_request_queue,     "LLM"),
        ]:
            if proc and proc.is_alive():
                if queue:
                    try:
                        queue.put_nowait(None)
                    except Exception:
                        pass
                proc.join(timeout=5.0)
                if proc.is_alive():
                    proc.kill()
                logger.info(f"{name}Process encerrado")

        if self._server:
            self._server.close()

        if self.executor:
            self.executor.shutdown(wait=False)

        logger.info("MotionInferenceService parado")

    # ──────────────────────────────────────────────────────────────────────────
    # Stats
    # ──────────────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        return {
            "active_sessions":    self.session_registry.session_count(),
            "active_npcs":        self.npc_session_manager.session_count(),
            "sequence_buffer":    self.sequence_buffer.summary(),
            "wm_trainer":         self.wm_trainer_thread.get_stats(),
            "policy_registry": {
                "current_version": self.policy_registry.current_version,
                "mean_reward":     self.policy_registry.mean_reward if self.policy_registry.has_reward_data else 0.0,
                "reward_count":    self.policy_registry.reward_count,
                "can_publish":     self.policy_registry.can_publish(),
            },
            "reward_normalizer":  {
                "mean": self.reward_normalizer.mean,
                "std":  self.reward_normalizer.std,
            },
            "intrinsic_reward":   self.intrinsic_reward.get_diagnostics(),
            "uncertainty":        self.uncertainty_controller.get_diagnostics(),
        }
