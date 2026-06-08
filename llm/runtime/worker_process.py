"""
worker_process.py
==================
Worker de inferência RL — substitui paradigma FAISS por RSSM + Policy.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    session_id:     str
    seq_id:         int
    request_dict:   dict
    llm_style_hint: int = 0
    npc_id:         int = 0


@dataclass
class InferenceResponse:
    session_id:              str
    seq_id:                  int
    action_idx:              int
    direction:               list
    speed:                   float
    confidence:              float
    entropy:                 float
    latency_ms:              float
    valid:                   bool
    embedding:               np.ndarray = field(
        default_factory=lambda: np.zeros(256, dtype=np.float32)
    )
    refined_trajectory_dict: dict = field(default_factory=dict)
    selected_style:          int  = 0


def inference_worker_process(
    worker_id:              int,
    request_queue:          mp.Queue,
    response_queue:         mp.Queue,
    config_dict:            dict,
    checkpoint_dir:         str,
    weight_sync_interval_s: float = 30.0,
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s | W{worker_id:02d} | %(levelname)-8s | %(message)s",
    )
    log = logging.getLogger(f"worker_{worker_id:02d}")
    log.info(f"InferenceWorker {worker_id} iniciado")

    import torch
    from config import DEFAULT_CONFIG
    from world_model.world_model import WorldModel
    from planning.policy import Policy
    from planning.action_executor import ActionExecutor
    from learning.policy_registry import PolicyRegistry
    from encoding.pose_encoder import PoseEncoder

    config   = DEFAULT_CONFIG
    wm_cfg   = config.world_model
    ac_cfg   = config.actor_critic
    device   = config.device

    combined_dim = (
        wm_cfg.rssm_num_categories * wm_cfg.rssm_category_dim
        + wm_cfg.rssm_hidden_dim
    )

    world_model = WorldModel(
        obs_enc_dim=config.encoder.embedding_dim,
        action_dim=ac_cfg.action_dim,
        hidden_dim=wm_cfg.rssm_hidden_dim,
        num_categories=wm_cfg.rssm_num_categories,
        category_dim=wm_cfg.rssm_category_dim,
        free_nats=wm_cfg.rssm_free_nats,
        kl_balance=wm_cfg.rssm_kl_balance,
        num_bones=getattr(wm_cfg, "num_bones", 89),
    )

    policy       = Policy(combined_dim=combined_dim, action_dim=ac_cfg.action_dim, hidden=256)
    executor     = ActionExecutor(action_dim=ac_cfg.action_dim)
    pose_encoder = PoseEncoder(config=config.encoder, device=device)
    policy_reg   = PolicyRegistry(
        save_dir=checkpoint_dir, min_reward_threshold=-5.0, keep_last_n=5,
    )

    session_states: Dict[str, tuple] = {}

    def get_state(sid: str):
        if sid not in session_states:
            h = torch.zeros(1, wm_cfg.rssm_hidden_dim, device=device)
            z = torch.zeros(
                1, wm_cfg.rssm_num_categories * wm_cfg.rssm_category_dim, device=device
            )
            session_states[sid] = (h, z, 0)
        return session_states[sid]

    def sync_weights() -> None:
        ckpt = policy_reg.get_latest_path()
        if not ckpt:
            return
        try:
            payload = torch.load(ckpt, map_location=device, weights_only=False)
            state   = payload.get("state_dict", payload)
            if "rssm"    in state: world_model.rssm.load_state_dict(state["rssm"],    strict=False)
            if "decoder" in state: world_model.decoder.load_state_dict(state["decoder"], strict=False)
            if "pose_decoder" in state and hasattr(world_model, "pose_decoder"):
                world_model.pose_decoder.load_state_dict(state["pose_decoder"], strict=False)
            if "actor"   in state: policy.actor.load_state_dict(state["actor"],   strict=False)
            if "critic"  in state: policy.critic.load_state_dict(state["critic"],  strict=False)
        except Exception as exc:
            log.debug(f"sync falhou: {exc}")

    last_sync = time.time()
    log.info(f"InferenceWorker {worker_id} pronto")

    while True:
        if (time.time() - last_sync) >= weight_sync_interval_s:
            sync_weights()
            last_sync = time.time()

        try:
            req: InferenceRequest = request_queue.get(timeout=0.1)
        except Exception:
            continue

        if req is None:
            log.info(f"W{worker_id} shutdown")
            break

        t0 = time.perf_counter()
        try:
            pose_frame = req.request_dict.get("pose_frame")
            if pose_frame is None:
                raise ValueError("pose_frame ausente")

            obs_enc, conf = pose_encoder.encode_frame(pose_frame, device)
            obs_t  = torch.from_numpy(obs_enc.astype(np.float32)).unsqueeze(0).to(device)

            h, z, last_action = get_state(req.session_id)
            a_prev = torch.zeros(1, ac_cfg.action_dim, device=device)
            if last_action > 0:
                a_prev[0, min(last_action, ac_cfg.action_dim - 1)] = 1.0

            with torch.no_grad():
                h_new, z_new, _, _, _, _ = world_model.rssm.forward(
                    h, z, a_prev, obs_enc=obs_t
                )
                action_t, _, entropy_t = policy.act_with_entropy(h_new, z_new)

            action_idx = int(action_t.item())
            entropy    = float(entropy_t.mean().item())
            idx, direction, speed = executor.decode(action_idx)
            session_states[req.session_id] = (h_new, z_new, idx)

            resp = InferenceResponse(
                session_id=req.session_id, seq_id=req.seq_id,
                action_idx=idx, direction=direction, speed=speed,
                confidence=float(conf), entropy=entropy,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                valid=True, embedding=obs_enc, selected_style=idx,
            )
        except Exception as exc:
            log.warning(f"W{worker_id} erro: {exc}")
            resp = InferenceResponse(
                session_id=req.session_id, seq_id=req.seq_id,
                action_idx=0, direction=[0.0, 0.0, 0.0], speed=0.0,
                confidence=0.0, entropy=3.0,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                valid=False,
            )
        try:
            response_queue.put_nowait(resp)
        except Exception:
            pass
