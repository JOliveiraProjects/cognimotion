from __future__ import annotations

import argparse
import logging
import time
from typing import List, Optional

import numpy as np
import torch
import torch.optim as optim

from envs.motion_env import MotionEnv, ACTION_DIM
from world_model.rssm import RSSM
from world_model.world_model import WorldModel
from world_model.symlog import symlog, EMANormalizer
from planning.policy import Policy
from planning.action_executor import ActionExecutor
from planning.uncertainty_controller import UncertaintyController
from runtime.sequence_buffer import SequenceBuffer
from memory.intrinsic_reward import IntrinsicRewardModule

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | EnvRunner | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

OBS_DIM        = 256
HIDDEN_DIM     = 256
NUM_CATEGORIES = 16
CATEGORY_DIM   = 16
STOCH_DIM      = NUM_CATEGORIES * CATEGORY_DIM   # 256
COMBINED_DIM   = STOCH_DIM + HIDDEN_DIM           # 512
WM_LR          = 3e-4
AC_LR          = 1e-4
BATCH_SIZE     = 16
SEQ_LEN        = 8
IMAGINATION_H  = 10
GAMMA          = 0.99
LAMBDA         = 0.95
ENTROPY_W      = 0.01
GRAD_CLIP      = 100.0
FREE_NATS      = 1.0
TRAIN_EVERY    = 20          # passos de env antes de treinar
MAX_EPISODES   = 500
DEVICE         = "cpu"


class EnvRunner:
    """
    Treina o pipeline DreamerV3 completo sobre o MotionEnv 2D.
    Valida toda a stack (RSSM + WorldModel + Policy + SequenceBuffer)
    sem depender do Unreal Engine.

    Fluxo por episódio:
      reset() → collect T steps → add_sequence → train_wm → train_ac
    """

    def __init__(self, device: str = DEVICE, seed: int = 42) -> None:
        self.device = device
        self.env    = MotionEnv(seed=seed)

        # World Model
        self.world_model = WorldModel(
            obs_enc_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            hidden_dim=HIDDEN_DIM,
            num_categories=NUM_CATEGORIES,
            category_dim=CATEGORY_DIM,
            free_nats=FREE_NATS,
            kl_balance=0.8,
            unimix=0.01,
            use_block_gru=False,  # blocos menores para CPU
        ).to(device)

        # Policy (Actor + Critic)
        self.policy = Policy(
            combined_dim=COMBINED_DIM,
            action_dim=ACTION_DIM,
            hidden=128,
        ).to(device)

        # Optimizers
        wm_params = (
            list(self.world_model.rssm.parameters()) +
            list(self.world_model.decoder.parameters())
        )
        self.wm_optim     = optim.Adam(wm_params,               lr=WM_LR,  eps=1e-8)
        self.actor_optim  = optim.Adam(self.policy.actor.parameters(),  lr=AC_LR, eps=1e-8)
        self.critic_optim = optim.Adam(self.policy.critic.parameters(), lr=AC_LR, eps=1e-8)

        # Sequence buffer
        self.seq_buf = SequenceBuffer(
            capacity=50_000,
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            seq_len=SEQ_LEN,
        )

        # Executor / action selector
        self.executor = ActionExecutor(action_dim=ACTION_DIM)
        self.unc      = UncertaintyController(enabled=True)
        self.intr     = IntrinsicRewardModule(embedding_dim=STOCH_DIM, beta=0.05)

        # EMA reward normalizer
        self.rew_ema = EMANormalizer(decay=0.99, clip=5.0)

        # Per-episode state (h, z)
        self._h = torch.zeros(1, HIDDEN_DIM, device=device)
        self._z = torch.zeros(1, STOCH_DIM,  device=device)

        # Metrics
        self._ep_rewards: List[float] = []
        self._wm_losses:  List[float] = []
        self._ac_losses:  List[float] = []
        self._step_count:  int = 0
        self._train_count: int = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Run
    # ──────────────────────────────────────────────────────────────────────────

    def run(self, max_episodes: int = MAX_EPISODES) -> None:
        logger.info(f"EnvRunner | episodes={max_episodes} | device={self.device}")

        for ep in range(1, max_episodes + 1):
            ep_reward = self._run_episode()
            self._ep_rewards.append(ep_reward)

            if ep % 20 == 0:
                recent = self._ep_rewards[-20:]
                mean_r = sum(recent) / len(recent)
                mean_wm = sum(self._wm_losses[-20:]) / max(len(self._wm_losses[-20:]), 1)
                mean_ac = sum(self._ac_losses[-20:]) / max(len(self._ac_losses[-20:]), 1)
                logger.info(
                    f"Ep {ep:4d}/{max_episodes} | "
                    f"R_mean={mean_r:.2f} | "
                    f"WM_loss={mean_wm:.4f} | "
                    f"AC_loss={mean_ac:.4f} | "
                    f"buf={self.seq_buf.total_transitions()} | "
                    f"trains={self._train_count}"
                )

        logger.info("EnvRunner | treinamento concluído")

    # ──────────────────────────────────────────────────────────────────────────
    # Episode
    # ──────────────────────────────────────────────────────────────────────────

    def _run_episode(self) -> float:
        obs  = self.env.reset()
        done = False
        total_r  = 0.0

        # Reset per-episode RSSM state
        self._h = torch.zeros(1, HIDDEN_DIM, device=self.device)
        self._z = torch.zeros(1, STOCH_DIM,  device=self.device)

        # Collect episode trajectory
        obs_list, act_list, rew_list, done_list = [], [], [], []

        while not done:
            action, log_p = self._select_action(obs)

            # Intrinsic reward from z state
            z_np   = self._z.squeeze(0).detach().cpu().numpy()
            r_intr = self.intr.compute(z_np)

            next_obs, r_ext, done, info = self.env.step(action)
            reward = float(r_ext) + r_intr
            total_r += float(r_ext)

            # One-hot action for RSSM
            act_oh = np.zeros(ACTION_DIM, dtype=np.float32)
            act_oh[action] = 1.0

            obs_list.append(obs.copy())
            act_list.append(act_oh)
            rew_list.append(reward)
            done_list.append(done)

            obs = next_obs
            self._step_count += 1

            # Train every N steps
            if self._step_count % TRAIN_EVERY == 0:
                self._maybe_train()

        # Add episode to buffer
        if len(obs_list) >= 2:
            self.seq_buf.add_sequence(
                obs_seq=np.stack(obs_list),
                action_seq=np.stack(act_list),
                reward_seq=np.array(rew_list, dtype=np.float32),
                done_seq=np.array(done_list, dtype=bool),
                group_key="sim",
            )

        return total_r

    # ──────────────────────────────────────────────────────────────────────────
    # Action selection
    # ──────────────────────────────────────────────────────────────────────────

    def _select_action(self, obs: np.ndarray) -> tuple[int, float]:
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        act_t = torch.zeros(1, ACTION_DIM, device=self.device)

        with torch.no_grad():
            h_new, z_new, _, _, _, _ = self.world_model.rssm.forward(
                self._h, self._z, act_t, obs_enc=obs_t
            )
            action_t = self.policy.get_action(h_new, z_new, deterministic=False)
            entropy  = self.policy.entropy(h_new, z_new)

        self._h = h_new
        self._z = z_new

        action   = int(action_t.item())
        log_prob = float(entropy.mean().item())

        # Uncertainty modulation
        uc_state = self.unc.update(float(entropy.mean().item()))
        mod      = self.unc.get_action_modification(action)
        action, _, _, _, _ = mod.apply(action, 0.5, [1.0, 0.0, 0.0])

        return action, log_prob

    # ──────────────────────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────────────────────

    def _maybe_train(self) -> None:
        if not self.seq_buf.ready_sequence(BATCH_SIZE):
            return

        wm_loss = self._train_world_model()
        ac_loss = self._train_actor_critic()

        if wm_loss is not None:
            self._wm_losses.append(wm_loss)
        if ac_loss is not None:
            self._ac_losses.append(ac_loss)

        self._train_count += 1

    def _train_world_model(self) -> Optional[float]:
        import torch.nn.functional as F

        batch = self.seq_buf.sample_sequence(BATCH_SIZE, SEQ_LEN)
        if batch is None:
            return None

        obs_s   = batch["obs"].to(self.device)      # (B, T, D)
        act_s   = batch["action"].to(self.device)   # (B, T, A)
        rew_s   = batch["reward"].to(self.device)   # (B, T)
        done_s  = batch["done"].to(self.device)     # (B, T)
        mask    = batch["mask"].to(self.device)     # (B, T)
        B, T, _ = obs_s.shape

        # Normaliza rewards com EMA + symlog
        r_sym  = symlog(rew_s)
        r_norm = self.rew_ema.update_and_normalize(r_sym)

        rollout = self.world_model.rollout_sequence(obs_s, act_s)

        # Reconstruction loss
        recon     = rollout["recon"]
        rec_loss  = (F.mse_loss(recon, obs_s, reduction="none").mean(-1) * mask).sum() / (mask.sum() + 1e-8)

        # KL loss (free nats)
        kl_seq   = rollout["kl"]
        kl_loss  = (torch.clamp(kl_seq - FREE_NATS, min=0.0) * mask).sum() / (mask.sum() + 1e-8)

        # Reward loss
        rew_pred = rollout["reward"]
        rew_loss = (F.mse_loss(rew_pred, r_norm, reduction="none") * mask).sum() / (mask.sum() + 1e-8)

        # Done loss
        done_pred = rollout["done"]
        done_loss = (F.binary_cross_entropy(done_pred, done_s.float(), reduction="none") * mask).sum() / (mask.sum() + 1e-8)

        total = rec_loss + kl_loss + rew_loss + 0.5 * done_loss

        self.wm_optim.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.world_model.rssm.parameters()) +
            list(self.world_model.decoder.parameters()),
            GRAD_CLIP,
        )
        self.wm_optim.step()
        return float(total.item())

    def _train_actor_critic(self) -> Optional[float]:
        import torch.nn.functional as F

        batch = self.seq_buf.sample_sequence(BATCH_SIZE, seq_len=1)
        if batch is None:
            return None

        obs_enc = batch["obs"][:, 0].to(self.device)
        act_in  = batch["action"][:, 0].to(self.device)

        with torch.no_grad():
            h, z = self.world_model.init_state(BATCH_SIZE, torch.device(self.device))
            h, z, _, _, _, _ = self.world_model.rssm.forward(h, z, act_in, obs_enc=obs_enc)

        # Imagination rollout
        rewards_t, values_t, dones_t = [], [], []
        log_probs_t, entropies_t     = [], []

        h_cur, z_cur = h.detach(), z.detach()
        for _ in range(IMAGINATION_H):
            logits, val = self.policy.forward(h_cur, z_cur)
            dist    = torch.distributions.Categorical(logits=logits)
            action  = dist.sample()
            log_p   = dist.log_prob(action)
            entropy = dist.entropy()
            act_oh  = F.one_hot(action, ACTION_DIM).float()

            with torch.no_grad():
                h_n, z_n, r, d, _, _ = self.world_model.imagine_step(h_cur, z_cur, act_oh)

            rewards_t.append(r.detach())
            values_t.append(val)
            dones_t.append(d.detach())
            log_probs_t.append(log_p)
            entropies_t.append(entropy)
            h_cur = h_n.detach()
            z_cur = z_n.detach()

        with torch.no_grad():
            _, last_val = self.policy.forward(h_cur, z_cur)
        values_t.append(last_val)

        rewards  = torch.stack(rewards_t)    # (H, B)
        values   = torch.stack(values_t)     # (H+1, B)
        dones    = torch.stack(dones_t)      # (H, B)
        log_probs = torch.stack(log_probs_t) # (H, B)
        entropies = torch.stack(entropies_t) # (H, B)

        # Lambda returns
        returns = torch.zeros(IMAGINATION_H, BATCH_SIZE, device=self.device)
        G = values[-1]
        for t in reversed(range(IMAGINATION_H)):
            nd = 1.0 - dones[t].float()
            G  = rewards[t] + GAMMA * nd * ((1.0 - LAMBDA) * values[t + 1] + LAMBDA * G)
            returns[t] = G

        ret_std = returns.std()
        ret_n   = (returns - returns.mean()) / (ret_std + 1e-8) if ret_std > 1e-6 else returns - returns.mean()
        adv     = ret_n - values[:-1].detach()

        actor_loss  = -(log_probs * adv.detach()).mean() - ENTROPY_W * entropies.mean()
        critic_loss = F.mse_loss(values[:-1], returns.detach())

        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), GRAD_CLIP)
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), GRAD_CLIP)
        self.critic_optim.step()

        return float(actor_loss.item() + critic_loss.item())


def main() -> None:
    parser = argparse.ArgumentParser(description="DreamerV3 local sim trainer")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--device",   type=str, default="cpu")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    runner = EnvRunner(device=args.device, seed=args.seed)
    runner.run(max_episodes=args.episodes)


if __name__ == "__main__":
    main()
