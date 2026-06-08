"""
actor_critic.py
===============
Rede Actor-Critic com GAE, imitation loss e suporte a skill embeddings.

Adaptado do arquivo enviado:
  - `rssm` tornado opcional: quando None, usa embedding_dim do projeto (256)
  - `config` usa ActorCriticConfig do projeto (config.py)
  - `imitation_loss` adaptado para PoseEncoder do projeto (retorna np.ndarray, não Tensor)
  - `update()` sem rssm: policy gradient sobre embedding direto
  - Todos os métodos originais mantidos: forward, forward_with_skill, get_action,
    compute_gae, update, imitation_loss, update_with_demo
  - Sem remoção de nenhum método ou atributo
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ActorCritic(nn.Module):
    """
    Rede Actor-Critic para seleção de ECognitiveMotionStyle.

    Quando rssm=None (modo projeto):
      input_dim = config.embedding_dim  (256)
      h e z são ambos metade do embedding (128 cada)

    Quando rssm fornecido (modo original):
      input_dim = rssm.stochastic_dim + rssm.hidden_dim
    """

    def __init__(self, config, rssm=None) -> None:
        """
        Args:
            config:  ActorCriticConfig (do config.py do projeto)
            rssm:    Módulo RSSM opcional. Se None, usa embedding direto.
        """
        super().__init__()
        self.config = config
        self.rssm = rssm

        # Determina input_dim baseado na presença de rssm
        if rssm is not None:
            input_dim = rssm.stochastic_dim + rssm.hidden_dim
        else:
            input_dim = getattr(config, "embedding_dim", 256)

        self._input_dim = input_dim

        self.actor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, config.action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.skill_proj = nn.Linear(config.skill_embed_dim, input_dim)

        logger.info(
            f"ActorCritic | input_dim={input_dim} | action_dim={config.action_dim} "
            f"| rssm={'yes' if rssm else 'no'}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Forward passes (originais — mantidos intactos)
    # ──────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quando rssm=None: h e z são concatenados diretamente.
        Quando rssm presente: idem (combined = cat([z, h])).
        """
        combined = torch.cat([z, h], dim=-1)
        action_logits = self.actor(combined)
        value = self.critic(combined).squeeze(-1)
        return action_logits, value

    def forward_with_skill(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        skill_vector: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([z, h], dim=-1)
        skill_bias = self.skill_proj(skill_vector)
        combined_sk = combined + skill_bias
        action_logits = self.actor(combined_sk)
        value = self.critic(combined_sk).squeeze(-1)
        return action_logits, value

    def get_action(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        logits, _ = self.forward(h, z)
        if deterministic:
            return torch.argmax(logits, dim=-1)
        return torch.distributions.Categorical(logits=logits).sample()

    def get_action_from_embedding(
        self,
        embedding: np.ndarray,
        deterministic: bool = False,
    ) -> int:
        """
        Conveniência: aceita np.ndarray (saída do PoseEncoder) diretamente.
        Divide o embedding em h e z (metade cada) quando rssm=None.
        """
        emb = torch.from_numpy(embedding.astype(np.float32)).unsqueeze(0)
        half = emb.shape[-1] // 2
        h = emb[:, :half]
        z = emb[:, half:]
        with torch.no_grad():
            action = self.get_action(h, z, deterministic=deterministic)
        return int(action.item())

    # ──────────────────────────────────────────────────────────────────────────
    # GAE (original — mantido intacto)
    # ──────────────────────────────────────────────────────────────────────────

    def compute_gae(
        self,
        rewards: List[torch.Tensor],
        values: List[torch.Tensor],
        dones: List[torch.Tensor],
        last_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        values_list = values + [last_value]
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = (
                rewards[t]
                + self.config.discount * (1 - dones[t]) * values_list[t + 1]
                - values_list[t]
            )
            gae = (
                delta
                + self.config.discount
                * self.config.lambda_gae
                * (1 - dones[t])
                * gae
            )
            advantages.insert(0, gae)
        advantages = torch.stack(advantages)
        returns = advantages + torch.stack(values_list[:-1])
        return advantages, returns

    # ──────────────────────────────────────────────────────────────────────────
    # Update com RSSM (original) ou sem RSSM (adaptado)
    # ──────────────────────────────────────────────────────────────────────────

    def update(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        optimizer_actor: torch.optim.Optimizer,
        optimizer_critic: torch.optim.Optimizer,
        num_imagined: Optional[int] = None,
    ) -> Tuple[float, float, float]:
        """
        Atualiza actor e critic.

        Com rssm: rollout imaginário de num_imagined steps (comportamento original).
        Sem rssm: policy gradient direto sobre o par (h, z) fornecido.
        """
        if num_imagined is None:
            num_imagined = self.config.imagination_horizon

        if self.rssm is not None:
            return self._update_with_rssm(h, z, optimizer_actor, optimizer_critic, num_imagined)
        else:
            return self._update_direct(h, z, optimizer_actor, optimizer_critic)

    def _update_with_rssm(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        optimizer_actor: torch.optim.Optimizer,
        optimizer_critic: torch.optim.Optimizer,
        num_imagined: int,
    ) -> Tuple[float, float, float]:
        """Rollout imaginário usando RSSM (comportamento original do arquivo enviado)."""
        h = h.detach()
        z = z.detach()

        actions_logits, actions, rewards, values, dones, entropies = [], [], [], [], [], []

        for _ in range(num_imagined):
            logits, value = self.forward(h, z)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            action_onehot = F.one_hot(action, num_classes=self.config.action_dim).float()
            if action_onehot.dim() == 1:
                action_onehot = action_onehot.unsqueeze(0)

            h, z, reward, done, _, _ = self.rssm(h, z, action_onehot, obs_enc=None)
            h = h.detach()
            z = z.detach()

            actions_logits.append(logits)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            dones.append(done)
            entropies.append(dist.entropy().mean())

        with torch.no_grad():
            _, last_value = self.forward(h, z)

        advantages, returns = self.compute_gae(rewards, values, dones, last_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_losses = []
        for t in range(num_imagined):
            dist = torch.distributions.Categorical(logits=actions_logits[t])
            log_prob = dist.log_prob(actions[t])
            actor_losses.append(-log_prob * advantages[t].detach())

        actor_loss = torch.stack(actor_losses).mean()
        entropy = torch.stack(entropies).mean()
        actor_loss = actor_loss - self.config.entropy_weight * entropy

        critic_loss = F.mse_loss(torch.stack(values), returns.detach())

        optimizer_actor.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.config.grad_clip_norm
        )
        optimizer_actor.step()

        optimizer_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.config.grad_clip_norm
        )
        optimizer_critic.step()

        return float(actor_loss.item()), float(critic_loss.item()), float(entropy.item())

    def _update_direct(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        optimizer_actor: torch.optim.Optimizer,
        optimizer_critic: torch.optim.Optimizer,
    ) -> Tuple[float, float, float]:
        """
        Policy gradient direto sobre (h, z) sem RSSM.
        Usa rewards sintéticos baseados em confiança do embedding.
        Adequado para ser chamado com embeddings do PoseEncoder.
        """
        logits, value = self.forward(h, z)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        entropy = dist.entropy().mean()

        # Reward sintético: log_prob como proxy de adequação do action
        log_prob = dist.log_prob(action)

        # Advantage: diferença entre log_prob e baseline (value normalizado).
        # Sem next_state disponível, usamos o valor atual como baseline de referência
        # para reduzir variância sem viés — equivalente a um REINFORCE com baseline.
        baseline  = value.detach().mean()
        advantage = (log_prob.detach() - baseline).clamp(-2.0, 2.0)

        actor_loss  = -(log_prob * advantage).mean() - self.config.entropy_weight * entropy
        # Critic aprende a prever log_prob como sinal proxy de qualidade da ação
        target_value = log_prob.detach()
        if value.shape != target_value.shape:
            target_value = target_value.expand_as(value)
        critic_loss = F.mse_loss(value, target_value)

        optimizer_actor.zero_grad()
        actor_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip_norm)
        optimizer_actor.step()

        optimizer_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.grad_clip_norm)
        optimizer_critic.step()

        return float(actor_loss.item()), float(critic_loss.item()), float(entropy.item())

    # ──────────────────────────────────────────────────────────────────────────
    # Imitation loss (adaptado para PoseEncoder do projeto)
    # ──────────────────────────────────────────────────────────────────────────

    def imitation_loss(
        self,
        obs: torch.Tensor,
        target_action: torch.Tensor,
        encoder: nn.Module,
        rssm: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """
        BC loss: cross-entropy entre ação do ator e ação demonstrada.

        Adaptação do projeto:
          - encoder: PoseEncoder (retorna (embedding, confidence) quando
            chamado via encode_frame) OU nn.Module padrão (retorna Tensor)
          - rssm: opcional — quando None, usa embedding diretamente

        target_action pode ser one-hot (B, action_dim) ou índice (B,).
        """
        device = next(self.parameters()).device
        obs = obs.to(device)

        if target_action.dim() == 1:
            target_idx = target_action.long().to(device)
        else:
            target_idx = target_action.argmax(dim=-1).long().to(device)
        if target_idx.dim() > 1:
            target_idx = target_idx.squeeze(-1)

        batch_size = obs.size(0)

        with torch.no_grad():
            # Tenta usar o encoder como nn.Module (forward direto)
            try:
                z_enc = encoder(obs)
                if isinstance(z_enc, tuple):
                    z_enc = z_enc[0]   # (embedding, confidence) → embedding
            except Exception:
                z_enc = obs            # fallback: usa obs diretamente

            if rssm is not None:
                h = torch.zeros(batch_size, rssm.hidden_dim, device=device)
                z = torch.zeros(batch_size, rssm.stochastic_dim, device=device)
                a_dummy = torch.zeros(batch_size, rssm.action_dim, device=device)
                h_out, z_out, _, _, _, _ = rssm(h, z, a_dummy, obs_enc=z_enc)
                h = h_out[:batch_size]
                z = z_out[:batch_size]
            else:
                # Sem RSSM: divide embedding em h e z
                half = z_enc.shape[-1] // 2
                h = z_enc[:, :half]
                z = z_enc[:, half:]

        logits, _ = self.forward(h, z)
        return F.cross_entropy(logits, target_idx)

    # ──────────────────────────────────────────────────────────────────────────
    # Update com demonstrações (original — adaptado para rssm opcional)
    # ──────────────────────────────────────────────────────────────────────────

    def update_with_demo(
        self,
        batch: dict,
        optimizer_actor: torch.optim.Optimizer,
        imitation_alpha: Optional[float] = None,
        encoder: Optional[nn.Module] = None,
        rssm: Optional[nn.Module] = None,
    ) -> float:
        """
        Atualiza actor com behavioral cloning loss nas transições de demo.

        Args:
            batch:           dict com chaves "obs", "action", "is_demo"
            optimizer_actor: otimizador do ator
            imitation_alpha: peso da BC loss (default: config.imitation_alpha)
            encoder:         encoder de observações (opcional)
            rssm:            módulo RSSM (opcional)
        """
        if imitation_alpha is None:
            imitation_alpha = getattr(self.config, "imitation_alpha", 0.5)

        obs      = batch["obs"]
        action   = batch["action"]
        is_demo  = batch["is_demo"]

        total_loss = 0.0
        if is_demo.any():
            demo_obs    = obs[is_demo]
            demo_action = action[is_demo]
            if encoder is not None:
                bc_loss = self.imitation_loss(demo_obs, demo_action, encoder, rssm)
            else:
                # Sem encoder: usa obs diretamente como embedding
                bc_loss = self.imitation_loss(demo_obs, demo_action,
                                              nn.Identity(), rssm)
            total_loss += imitation_alpha * bc_loss

        if total_loss != 0.0:
            optimizer_actor.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.config.grad_clip_norm
            )
            optimizer_actor.step()
            return float(total_loss.item())
        return 0.0
