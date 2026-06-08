from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ActorNet(nn.Module):
    def __init__(self, combined_dim: int, action_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(combined_dim, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),       nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, combined: torch.Tensor) -> torch.Tensor:
        return self.net(combined)


class CriticNet(nn.Module):
    def __init__(self, combined_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(combined_dim, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),       nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, 1),
        )

    def forward(self, combined: torch.Tensor) -> torch.Tensor:
        return self.net(combined).squeeze(-1)


class Policy(nn.Module):
    """
    Actor-Critic para DreamerV3.
    Recebe combined = cat([z, h]) e retorna logits + value.
    Suporta:
      - forward: ação + valor
      - get_action: amostragem / determinístico
      - entropy: entropia da distribuição de ação
      - imitation_loss: cross-entropy com ação demonstrada
    """

    def __init__(
        self,
        combined_dim:    int,
        action_dim:      int,
        hidden:          int   = 256,
        skill_embed_dim: int   = 64,
    ) -> None:
        super().__init__()
        self.combined_dim    = combined_dim
        self.action_dim      = action_dim

        self.actor  = ActorNet(combined_dim, action_dim, hidden)
        self.critic = CriticNet(combined_dim, hidden)
        self.skill_proj = nn.Linear(skill_embed_dim, combined_dim)

        logger.info(f"Policy | combined={combined_dim} | actions={action_dim} | hidden={hidden}")

    def _combined(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.cat([z, h], dim=-1)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        c = self._combined(h, z)
        return self.actor(c), self.critic(c)

    def forward_combined(self, combined: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.actor(combined), self.critic(combined)

    def forward_with_skill(
        self, h: torch.Tensor, z: torch.Tensor, skill: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        c = self._combined(h, z) + self.skill_proj(skill)
        return self.actor(c), self.critic(c)

    @torch.no_grad()
    def get_action(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        deterministic: bool = False,
        temperature:   float = 1.0,
    ) -> torch.Tensor:
        logits, _ = self.forward(h, z)
        if deterministic:
            return logits.argmax(dim=-1)
        if temperature != 1.0:
            logits = logits / max(temperature, 1e-6)
        return torch.distributions.Categorical(logits=logits).sample()

    @torch.no_grad()
    def act_with_entropy(
        self, h: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(h, z)
        dist          = torch.distributions.Categorical(logits=logits)
        action        = dist.sample()
        return action, value, dist.entropy()

    @torch.no_grad()
    def entropy(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(h, z)
        return torch.distributions.Categorical(logits=logits).entropy()

    def imitation_loss(
        self, h: torch.Tensor, z: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        idx = target.argmax(-1).long() if target.dim() > 1 else target.long()
        logits, _ = self.forward(h, z)
        return F.cross_entropy(logits, idx.to(logits.device))

    def policy_gradient_loss(
        self,
        log_probs:  torch.Tensor,
        advantages: torch.Tensor,
        entropy:    torch.Tensor,
        entropy_weight: float = 0.01,
    ) -> torch.Tensor:
        return -(log_probs * advantages.detach()).mean() - entropy_weight * entropy.mean()

    def critic_loss(
        self, h: torch.Tensor, z: torch.Tensor, returns: torch.Tensor
    ) -> torch.Tensor:
        _, value = self.forward(h, z)
        return F.mse_loss(value, returns.detach())
