"""
world_model/dreamer_trainer.py
================================
DreamerTrainer — treina WorldModel + Policy via imaginação (DreamerV3).

Única classe de política: planning.policy.Policy
  - inference usa Policy (main process, com lock de leitura)
  - training usa Policy (WorldModelTrainerThread, com lock de escrita)
  - DreamerProcess usa cópia independente de Policy, sincroniza via checkpoint

Thread-safety:
  - DreamerTrainer aceita um model_lock (threading.RLock)
  - train_world_model / update_actor_critic adquirem o lock
  - rl_inference no serviço também adquire o mesmo lock
"""
from __future__ import annotations

import logging
import math
import multiprocessing as mp
import threading
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim

logger = logging.getLogger(__name__)


class DreamerTrainer:
    """
    Treinamento intercalado:
      1. train_world_model()  — RSSM + decoder loss sobre sequências reais
      2. update_actor_critic() — lambda returns sobre rollout imaginado
      3. maybe_publish()       — checkpoint via PolicyRegistry
    """

    def __init__(
        self,
        world_model,
        actor_critic,         # planning.policy.Policy
        sequence_buffer,
        policy_registry,
        config,
        device:     str = "cpu",
        model_lock: Optional[threading.RLock] = None,
    ) -> None:
        self.wm              = world_model.to(device)
        self.ac              = actor_critic.to(device)   # Policy
        self.seq_buffer      = sequence_buffer
        self.policy_registry = policy_registry
        self.config          = config
        self.device          = device
        self.model_lock      = model_lock or threading.RLock()

        # Perfil do skeleton em treino (para gravar/verificar compatibilidade).
        # Default = humanoide 89 bones. Pode ser sobrescrito pelo serviço quando
        # o Unreal informar o skeleton real do NPC.
        try:
            from world_model.skeleton_profile import profile_for
            self.skeleton_profile = profile_for(self.wm.num_bones, "Default")
        except Exception:
            self.skeleton_profile = None

        cfg  = config.world_model
        acfg = config.actor_critic

        wm_params = (
            list(self.wm.rssm.parameters()) +
            list(self.wm.decoder.parameters()) +
            list(self.wm.pose_decoder.parameters())  # cabeça generativa de pose
        )
        self.wm_optim     = optim.Adam(wm_params,                    lr=cfg.learning_rate,  eps=1e-8)
        self.actor_optim  = optim.Adam(self.ac.actor.parameters(),   lr=cfg.learning_rate, eps=1e-8)
        self.critic_optim = optim.Adam(self.ac.critic.parameters(),  lr=cfg.learning_rate, eps=1e-8)

        self._wm_steps:     int = 0
        self._ac_steps:     int = 0
        self._last_publish: int = 0
        self._rew_mean: float = 0.0
        self._rew_m2:   float = 0.0
        self._rew_cnt:  int   = 0

        logger.info(
            f"DreamerTrainer | device={device} "
            f"| wm_lr={cfg.learning_rate} | ac_lr={acfg.learning_rate}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # World Model
    # ──────────────────────────────────────────────────────────────────────────

    def train_world_model(self) -> Optional[Dict[str, float]]:
        cfg = self.config.world_model
        batch = self.seq_buffer.sample_sequence(cfg.batch_size, cfg.seq_len)
        if batch is None:
            return None

        obs_s  = batch["obs"].to(self.device)
        act_s  = batch["action"].to(self.device)
        rew_s  = batch["reward"].to(self.device)
        done_s = batch["done"].to(self.device)
        mask   = batch["mask"].to(self.device)
        B, T   = obs_s.shape[:2]

        self._update_reward_stats(rew_s, mask)
        rew_norm = self._normalize_reward(rew_s)

        with self.model_lock:
            rollout = self.wm.rollout_sequence(obs_s, act_s)

            recon     = rollout["recon"]
            rec_loss  = (F.mse_loss(recon, obs_s, reduction="none").mean(-1) * mask).sum() / (mask.sum() + 1e-8)

            # ── PERDA DE POSE: decoder aprende a gerar as poses reais do líder ──
            # As poses do líder vêm em ESCALA DE MUNDO (cm): loc em centenas,
            # quaternion em [-1,1]. Um MSE direto faz a parte de localização
            # dominar tudo (loss ~1200) e afogar o aprendizado do world model
            # (rec/kl), travando a loss e deixando kl≈0. Corrigimos normalizando
            # as posições por uma escala de esqueleto e separando loc de quat.
            pose_loss = torch.zeros((), device=self.device)
            pose_tgt  = batch.get("pose")
            if pose_tgt is not None and pose_tgt.dim() == 3 and pose_tgt.shape[-1] == self.wm.pose_decoder.out_dim:
                pose_tgt = pose_tgt.to(self.device)
                pose_pred = self.wm.pose_decoder(rollout["combined"])  # (B,T,num_bones*7)
                pose_pred = self.wm.pose_decoder.normalize_quaternions(
                    pose_pred, self.wm.pose_decoder.num_bones)

                NB = self.wm.pose_decoder.num_bones
                POSE_SCALE = float(getattr(cfg, "pose_position_scale", 100.0))  # cm → ~1
                # Separa [loc(3), quat(4)] por bone
                pp = pose_pred.view(*pose_pred.shape[:-1], NB, 7)
                pt = pose_tgt.view(*pose_tgt.shape[:-1], NB, 7)
                loc_pred, quat_pred = pp[..., :3] / POSE_SCALE, pp[..., 3:]
                loc_tgt,  quat_tgt  = pt[..., :3] / POSE_SCALE, pt[..., 3:]
                # Huber na posição (robusto a outliers), MSE no quaternion
                loc_err  = F.huber_loss(loc_pred,  loc_tgt,  reduction="none", delta=1.0).mean(-1)
                quat_err = F.mse_loss(quat_pred, quat_tgt, reduction="none").mean(-1)
                per_bone = (loc_err + quat_err).mean(-1)  # média sobre bones → (B,T)
                pose_loss = (per_bone * mask).sum() / (mask.sum() + 1e-8)
            kl_loss   = (torch.clamp(rollout["kl"] - cfg.rssm_free_nats, min=0.0) * mask).sum() / (mask.sum() + 1e-8)
            rew_loss  = (F.mse_loss(rollout["reward"], rew_norm, reduction="none") * mask).sum() / (mask.sum() + 1e-8)
            done_loss = (F.binary_cross_entropy(rollout["done"], done_s.float(), reduction="none") * mask).sum() / (mask.sum() + 1e-8)
            ov_loss   = self._overshooting_loss(
                rollout["combined"].detach(), rollout["z"], rollout["h"], act_s, mask,
                ov_steps=getattr(cfg, "overshooting_steps", 2),
            )

            total = (
                rec_loss  * getattr(cfg, "recon_weight",  1.0) +
                kl_loss   * getattr(cfg, "kl_weight",     1.0) +
                rew_loss  * getattr(cfg, "reward_weight", 1.0) +
                done_loss * getattr(cfg, "done_weight",   0.5) +
                pose_loss * getattr(cfg, "pose_weight",   2.0) +
                ov_loss
            )

            self.wm_optim.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.wm.rssm.parameters()) + list(self.wm.decoder.parameters())
                    + list(self.wm.pose_decoder.parameters()),
                max_norm=getattr(cfg, "grad_clip_norm", 100.0),
            )
            self.wm_optim.step()

        self._wm_steps += 1
        if self._wm_steps % 200 == 0:
            logger.info(
                f"DreamerTrainer | wm_step={self._wm_steps} "
                f"| loss={total.item():.4f} | rec={rec_loss.item():.4f} "
                f"| kl={kl_loss.item():.4f}"
            )
        return {
            "wm/loss":      float(total.item()),
            "wm/rec":       float(rec_loss.item()),
            "wm/kl":        float(kl_loss.item()),
            "wm/reward":    float(rew_loss.item()),
            "wm/done":      float(done_loss.item()),
            "wm/pose":      float(pose_loss.item()),
            "wm/overshoot": float(ov_loss.item()),
            "wm/steps":     self._wm_steps,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Actor-Critic via imaginação
    # ──────────────────────────────────────────────────────────────────────────

    def update_actor_critic(self) -> Optional[Dict[str, float]]:
        cfg  = self.config.world_model
        acfg = self.config.actor_critic

        starts = self._sample_start_states(cfg.imagination_batch_size)
        if starts is None:
            return None

        h, z  = starts
        H     = acfg.imagination_horizon

        rewards_t, values_t, dones_t = [], [], []
        log_probs_t, entropies_t     = [], []

        with self.model_lock:
            for _ in range(H):
                logits, val = self.ac.forward(h, z)
                dist    = torch.distributions.Categorical(logits=logits)
                action  = dist.sample()
                log_p   = dist.log_prob(action)
                entropy = dist.entropy()
                act_oh  = F.one_hot(action, acfg.action_dim).float()

                with torch.no_grad():
                    h_n, z_n, r, d, _, _ = self.wm.imagine_step(h, z, act_oh)

                rewards_t.append(r.detach())
                values_t.append(val)
                dones_t.append(d.detach())
                log_probs_t.append(log_p)
                entropies_t.append(entropy)
                h = h_n.detach()
                z = z_n.detach()

            with torch.no_grad():
                _, last_val = self.ac.forward(h, z)
            values_t.append(last_val)

            rewards   = torch.stack(rewards_t)    # (H, B)
            values    = torch.stack(values_t)     # (H+1, B)
            dones     = torch.stack(dones_t)
            log_probs = torch.stack(log_probs_t)
            entropies = torch.stack(entropies_t)

            returns = self._lambda_returns(rewards, values, dones, acfg.discount, acfg.lambda_gae)
            ret_std = returns.std()
            ret_n   = (returns - returns.mean()) / (ret_std + 1e-8) if ret_std > 1e-6 else returns - returns.mean()
            adv     = ret_n - values[:-1].detach()

            actor_loss  = -(log_probs * adv.detach()).mean() - acfg.entropy_weight * entropies.mean()
            critic_loss = F.mse_loss(values[:-1], returns.detach())

            self.actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ac.actor.parameters(), acfg.grad_clip_norm)
            self.actor_optim.step()

            self.critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ac.critic.parameters(), acfg.grad_clip_norm)
            self.critic_optim.step()

        self._ac_steps += 1

        # CRÍTICO: alimenta o PolicyRegistry com o retorno imaginado.
        # Sem isso _rewards fica vazio, can_publish()=False e NENHUM checkpoint
        # é salvo (era a causa da pasta checkpoints sempre vazia).
        mean_return = float(returns.mean().item())
        if self.policy_registry is not None:
            try:
                self.policy_registry.record_reward(mean_return)
            except Exception as e:
                logger.debug("[CKPT] Falha ao registrar reward: %s", e)

        return {
            "ac/actor_loss":   float(actor_loss.item()),
            "ac/critic_loss":  float(critic_loss.item()),
            "ac/entropy":      float(entropies.mean().item()),
            "ac/returns_mean": mean_return,
            "ac/steps":        self._ac_steps,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Checkpoint publish
    # ──────────────────────────────────────────────────────────────────────────

    def build_checkpoint_state(self) -> dict:
        """Monta o dict aninhado de submódulos (formato único de checkpoint)."""
        with self.model_lock:
            return {
                "rssm":         self.wm.rssm.state_dict(),
                "decoder":      self.wm.decoder.state_dict(),
                "pose_decoder": self.wm.pose_decoder.state_dict(),
                "actor":        self.ac.actor.state_dict(),
                "critic":       self.ac.critic.state_dict(),
            }

    def load_checkpoint_state(self, state: dict) -> None:
        """
        Restaura os submódulos a partir do dict aninhado salvo por
        build_checkpoint_state(). Carrega cada parte no submódulo correto —
        é o inverso exato do save, evitando o erro de 'chaves inesperadas'.
        """
        with self.model_lock:
            if "rssm" in state:
                self.wm.rssm.load_state_dict(state["rssm"])
            if "decoder" in state:
                self.wm.decoder.load_state_dict(state["decoder"])
            if "pose_decoder" in state:
                self.wm.pose_decoder.load_state_dict(state["pose_decoder"])
            if "actor" in state:
                self.ac.actor.load_state_dict(state["actor"])
            if "critic" in state:
                self.ac.critic.load_state_dict(state["critic"])

    def maybe_publish(self, min_interval_steps: int = 500) -> None:
        if (self._wm_steps - self._last_publish) < min_interval_steps:
            return
        if not self.policy_registry.can_publish():
            return

        state = self.build_checkpoint_state()

        v = self.policy_registry.publish(state, extra_metadata={
            "wm_steps": self._wm_steps,
            "ac_steps": self._ac_steps,
        })
        if v > 0:
            self._last_publish = self._wm_steps
            logger.info(f"DreamerTrainer | checkpoint v{v} publicado | wm={self._wm_steps}")

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers internos
    # ──────────────────────────────────────────────────────────────────────────

    def _sample_start_states(self, n: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        batch = self.seq_buffer.sample_sequence(n, seq_len=1)
        if batch is None:
            return None
        obs_enc = batch["obs"][:, 0].to(self.device)
        act     = batch["action"][:, 0].to(self.device)
        with torch.no_grad():
            h, z = self.wm.init_state(n, torch.device(self.device))
            h, z, _, _, _, _ = self.wm.rssm.forward(h, z, act, obs_enc=obs_enc)
        return h.detach(), z.detach()

    @staticmethod
    def _lambda_returns(
        rewards: torch.Tensor, values: torch.Tensor,
        dones: torch.Tensor, gamma: float, lam: float,
    ) -> torch.Tensor:
        H, B    = rewards.shape
        returns = torch.zeros(H, B, device=rewards.device)
        G       = values[-1]
        for t in reversed(range(H)):
            nd = 1.0 - dones[t].float()
            G  = rewards[t] + gamma * nd * ((1.0 - lam) * values[t + 1] + lam * G)
            returns[t] = G
        return returns

    def _overshooting_loss(
        self,
        combined_tgt: torch.Tensor,
        z_states:     torch.Tensor,
        h_states:     torch.Tensor,
        action_seq:   torch.Tensor,
        mask:         torch.Tensor,
        ov_steps:     int = 2,
    ) -> torch.Tensor:
        B, T, _ = combined_tgt.shape
        total   = torch.tensor(0.0, device=self.device)
        count   = 0
        for step in range(1, min(ov_steps + 1, T)):
            for t in range(T - step):
                ht, zt = h_states[:, t], z_states[:, t]
                for s in range(step):
                    ht, zt, _, _, _, _ = self.wm.rssm.forward(
                        ht, zt, action_seq[:, t + s], obs_enc=None
                    )
                pred = torch.cat([zt, ht], dim=-1)
                tgt  = combined_tgt[:, t + step]
                err  = F.mse_loss(pred, tgt, reduction="none").mean(-1)
                m    = mask[:, t]
                total += (err * m).sum() / (m.sum() + 1e-8)
                count += 1
        return total / max(count, 1)

    def _update_reward_stats(self, rew: torch.Tensor, mask: torch.Tensor) -> None:
        vals = rew[mask > 0].detach().cpu().float().tolist()
        for r in vals:
            self._rew_cnt += 1
            d = r - self._rew_mean
            self._rew_mean += d / self._rew_cnt
            self._rew_m2   += d * (r - self._rew_mean)

    def _normalize_reward(self, rew: torch.Tensor) -> torch.Tensor:
        if self._rew_cnt < 2:
            return rew
        std = math.sqrt(max(self._rew_m2 / (self._rew_cnt - 1), 0.0) + 1e-8)
        return (rew - self._rew_mean) / std


# ──────────────────────────────────────────────────────────────────────────────
# dreamer_worker_process — usa Policy (não DreamerActorCritic)
# ──────────────────────────────────────────────────────────────────────────────

def dreamer_worker_process(
    seq_buffer_queue: mp.Queue,
    config_dict:      dict,
    checkpoint_dir:   str,
    device:           str = "cpu",
) -> None:
    """
    Entry-point do processo DreamerV3 dedicado.
    Usa planning.policy.Policy — mesma classe do serviço principal.
    Publica checkpoints compatíveis com o worker_process.py.
    """
    import logging as _logging
    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s | DREAMER | %(levelname)-8s | %(message)s",
    )
    _log = _logging.getLogger("dreamer_worker")
    _log.info(f"DreamerWorker iniciado | device={device}")

    from config import DEFAULT_CONFIG
    from world_model.world_model import WorldModel
    from planning.policy import Policy
    from runtime.sequence_buffer import SequenceBuffer
    from learning.policy_registry import PolicyRegistry
    from encoding.perception_features import PERCEPTION_DIM

    config = DEFAULT_CONFIG
    wm_cfg = config.world_model
    ac_cfg = config.actor_critic

    world_model = WorldModel(
        obs_enc_dim=config.encoder.embedding_dim + PERCEPTION_DIM,
        action_dim=ac_cfg.action_dim,
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

    combined_dim = world_model.combined_dim
    policy = Policy(
        combined_dim=combined_dim,
        action_dim=ac_cfg.action_dim,
        hidden=256,
    )

    seq_buf = SequenceBuffer(
        capacity=wm_cfg.sequence_buffer_capacity,
        obs_dim=config.encoder.embedding_dim + PERCEPTION_DIM,
        action_dim=ac_cfg.action_dim,
        seq_len=wm_cfg.seq_len,
    )
    policy_reg = PolicyRegistry(
        save_dir=checkpoint_dir, min_reward_threshold=-5.0, keep_last_n=5,
    )
    trainer = DreamerTrainer(
        world_model=world_model, actor_critic=policy,
        sequence_buffer=seq_buf, policy_registry=policy_reg,
        config=config, device=device,
    )

    last_train = time.time()
    _log.info("DreamerWorker | loop iniciado")

    while True:
        # Drena queue IPC
        drained = 0
        while not seq_buffer_queue.empty():
            item = seq_buffer_queue.get_nowait()
            if item is None:
                _log.info("DreamerWorker | shutdown")
                return
            try:
                obs_seq, action_seq, reward_seq, done_seq, group_key = item
                import numpy as np
                obs_arr = np.asarray(obs_seq, dtype=np.float32)
                act_arr = np.zeros((len(obs_arr), ac_cfg.action_dim), dtype=np.float32) \
                    if action_seq is None else np.asarray(action_seq, dtype=np.float32)
                rew_arr  = np.zeros(len(obs_arr), dtype=np.float32) \
                    if reward_seq is None else np.asarray(reward_seq, dtype=np.float32)
                done_arr = np.zeros(len(obs_arr), dtype=bool) \
                    if done_seq is None else np.asarray(done_seq, dtype=bool)
                if len(obs_arr) > 1:
                    done_arr[-1] = True
                seq_buf.add_sequence(obs_arr, act_arr, rew_arr, done_arr, group_key)
                drained += 1
            except Exception as exc:
                _log.warning(f"DreamerWorker | seq inválida: {exc}")

        now = time.time()
        if (now - last_train) >= getattr(wm_cfg, "train_interval_s", 5.0):
            last_train = now
            trainer.train_world_model()
            if trainer._wm_steps > getattr(wm_cfg, "warmup_wm_steps", 50):
                trainer.update_actor_critic()
            trainer.maybe_publish(
                min_interval_steps=getattr(config.dreamer, "publish_interval_steps", 500)
            )
        else:
            time.sleep(0.05)
