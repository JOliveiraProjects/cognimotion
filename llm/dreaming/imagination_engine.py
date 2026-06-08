"""
dreaming/imagination_engine.py
================================
ImaginationEngine — gera rollouts imaginados e atualiza ator-crítico via lambda returns.

Adaptado de worldmodel_dreaming.zip/dreaming/imagination_engine.py:
  - Remove core.logger → logging padrão
  - Mantém imagine_batch, lambda_returns, update, sample_start_states, run_mode
  - Compatível com planning.policy.Policy
  - Compatível com WorldModel.imagine_step() (world_model/world_model.py)
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

LIGHT_HORIZON:    int = 5
LIGHT_BATCH_SIZE: int = 32
HEAVY_HORIZON:    int = 15
HEAVY_BATCH_SIZE: int = 256


class ImaginationEngine:
    """
    Imagination Engine DreamerV3.

    Ciclo de atualização:
      1. sample_start_states(buffer, encoder) → (h_starts, z_starts)
      2. imagine_batch(h, z, horizon)         → rollout dict
      3. lambda_returns(rewards, values, dones)
      4. update actor via policy gradient
      5. update critic via MSE

    Compatível com:
      - world_model.WorldModel.imagine_step()
      - planning.Policy
    """

    def __init__(
        self,
        config,           # MotionIntelligenceConfig com .actor_critic e .world_model
        world_model,      # world_model.WorldModel
        actor_critic,     # planning.policy.Policy
        device:    str  = "cpu",
        use_amp:   bool = False,
    ) -> None:
        self.config      = config
        self.wm          = world_model
        self.ac          = actor_critic
        self.device      = device
        self.use_amp     = use_amp and (device != "cpu")
        self.scaler      = torch.cuda.amp.GradScaler() if self.use_amp else None
        self._imagine_count = 0

        acfg = config.actor_critic
        logger.info(
            f"ImaginationEngine | horizon={acfg.imagination_horizon} "
            f"| action_dim={acfg.action_dim} | amp={use_amp}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # run_mode — atalho para light / heavy imagination
    # ──────────────────────────────────────────────────────────────────────────

    def run_mode(
        self,
        mode:            str,
        buffer,
        encoder,
        actor_optimizer:  "torch.optim.Optimizer",
        critic_optimizer: "torch.optim.Optimizer",
    ) -> Dict[str, float]:
        if mode == "light":
            horizon, batch_size = LIGHT_HORIZON, LIGHT_BATCH_SIZE
        else:
            horizon, batch_size = HEAVY_HORIZON, HEAVY_BATCH_SIZE

        starts = self.sample_start_states(buffer, encoder, n_starts=batch_size)
        if starts is None:
            return {"mode": mode, "skipped": True}

        h_s, z_s = starts
        metrics  = self.update(h_s, z_s, actor_optimizer, critic_optimizer,
                               horizon_override=horizon)
        metrics.update({"mode": mode, "horizon": horizon, "batch_size": batch_size})
        return metrics

    # ──────────────────────────────────────────────────────────────────────────
    # imagine_batch — rollout imaginado de H passos
    # ──────────────────────────────────────────────────────────────────────────

    def imagine_batch(
        self,
        h_batch: torch.Tensor,
        z_batch: torch.Tensor,
        horizon: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Executa H passos de imaginação usando o world model (prior).
        Retorna tensores (H, B) para rewards, values, dones, log_probs, entropies.
        """
        H = horizon or self.config.actor_critic.imagination_horizon
        h, z = h_batch.detach(), z_batch.detach()

        rewards_l: List[torch.Tensor] = []
        values_l:  List[torch.Tensor] = []
        dones_l:   List[torch.Tensor] = []
        lp_l:      List[torch.Tensor] = []
        ent_l:     List[torch.Tensor] = []

        for _ in range(H):
            action_logits, value = self.ac(h, z)
            dist      = torch.distributions.Categorical(logits=action_logits)
            action    = dist.sample()
            log_prob  = dist.log_prob(action)
            entropy   = dist.entropy()
            action_oh = F.one_hot(action, self.config.actor_critic.action_dim).float()

            with torch.no_grad():
                h_next, z_next, reward, done, _, _ = self.wm.imagine_step(h, z, action_oh)

            rewards_l.append(reward.detach())
            values_l.append(value)
            dones_l.append(done.detach())
            lp_l.append(log_prob)
            ent_l.append(entropy)

            h = h_next.detach()
            z = z_next.detach()

        with torch.no_grad():
            _, last_val = self.ac(h, z)
        values_l.append(last_val)

        return {
            "rewards":   torch.stack(rewards_l),         # (H, B)
            "values":    torch.stack(values_l),           # (H+1, B)
            "dones":     torch.stack(dones_l),            # (H, B)
            "log_probs": torch.stack(lp_l),               # (H, B)
            "entropies": torch.stack(ent_l),              # (H, B)
        }

    # ──────────────────────────────────────────────────────────────────────────
    # lambda_returns — retornos suavizados (DreamerV3)
    # ──────────────────────────────────────────────────────────────────────────

    def lambda_returns(
        self,
        rewards: torch.Tensor,
        values:  torch.Tensor,
        dones:   torch.Tensor,
        gamma:   Optional[float] = None,
        lam:     Optional[float] = None,
    ) -> torch.Tensor:
        gamma = gamma if gamma is not None else self.config.actor_critic.discount
        lam   = lam   if lam   is not None else self.config.actor_critic.lambda_gae
        H, _  = rewards.shape

        returns = torch.zeros_like(rewards)
        G       = values[-1]
        for t in reversed(range(H)):
            not_done = 1.0 - dones[t].float()
            G        = rewards[t] + gamma * not_done * ((1.0 - lam) * values[t + 1] + lam * G)
            returns[t] = G
        return returns

    # ──────────────────────────────────────────────────────────────────────────
    # update — atualiza actor e critic a partir do rollout imaginado
    # ──────────────────────────────────────────────────────────────────────────

    def update(
        self,
        h_starts:        torch.Tensor,
        z_starts:        torch.Tensor,
        actor_optimizer:  "torch.optim.Optimizer",
        critic_optimizer: "torch.optim.Optimizer",
        n_batches:        int = 1,
        horizon_override: Optional[int] = None,
    ) -> Dict[str, float]:
        acfg = self.config.actor_critic
        tot_actor = tot_critic = tot_entropy = tot_returns = 0.0

        for _ in range(n_batches):
            rollout  = self.imagine_batch(h_starts, z_starts, horizon=horizon_override)
            rewards  = rollout["rewards"]
            values   = rollout["values"]
            dones    = rollout["dones"]
            log_probs = rollout["log_probs"]
            entropies = rollout["entropies"]

            returns = self.lambda_returns(rewards, values, dones)

            ret_std = returns.std()
            ret_norm = (
                (returns - returns.mean()) / (ret_std + 1e-8)
                if ret_std > 1e-6 else returns - returns.mean()
            )
            advantages = ret_norm - values[:-1].detach()

            actor_loss  = -(log_probs * advantages.detach()).mean()
            entropy_reg = entropies.mean()
            actor_loss  = actor_loss - acfg.entropy_weight * entropy_reg

            critic_loss = F.mse_loss(values[:-1], returns.detach())

            actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.ac.actor.parameters(), acfg.grad_clip_norm
            )
            actor_optimizer.step()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.ac.critic.parameters(), acfg.grad_clip_norm
            )
            critic_optimizer.step()

            tot_actor   += actor_loss.item()
            tot_critic  += critic_loss.item()
            tot_entropy += entropy_reg.item()
            tot_returns += returns.mean().item()

        self._imagine_count += n_batches
        n = max(n_batches, 1)
        return {
            "imagination/actor_loss":   tot_actor   / n,
            "imagination/critic_loss":  tot_critic  / n,
            "imagination/entropy":      tot_entropy / n,
            "imagination/returns_mean": tot_returns / n,
            "imagination/total_steps":  self._imagine_count,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # sample_start_states — amostra (h, z) reais do buffer
    # ──────────────────────────────────────────────────────────────────────────

    def sample_start_states(
        self,
        buffer,
        encoder,
        n_starts: int = 64,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        batch = buffer.sample_sequence(n_starts, seq_len=1)
        if batch is None:
            return None

        obs_enc = batch["obs"][:, 0].to(self.device)
        act     = batch["action"][:, 0].to(self.device)

        with torch.no_grad():
            h, z = self.wm.init_state(n_starts, torch.device(self.device))
            h, z, _, _, _, _ = self.wm.rssm.forward(h, z, act, obs_enc=obs_enc)

        return h.detach(), z.detach()

    # ──────────────────────────────────────────────────────────────────────────
    # imagine_for_skill — rollout para diagnóstico de skill
    # ──────────────────────────────────────────────────────────────────────────

    def imagine_for_skill(
        self,
        skill_name: str,
        context:    str,
        buffer,
        encoder,
        n_starts: int = 32,
        horizon:  int = 10,
    ) -> Optional[Dict]:
        starts = self.sample_start_states(buffer, encoder, n_starts=n_starts)
        if starts is None:
            return None

        h, z = starts
        with torch.no_grad():
            rollout    = self.imagine_batch(h, z, horizon=horizon)
            pred_error = float(rollout["rewards"].std().item())

        return {
            "skill_name":      skill_name,
            "context":         context,
            "rollout":         rollout,
            "prediction_error": pred_error,
            "n_steps":         horizon * n_starts,
        }
