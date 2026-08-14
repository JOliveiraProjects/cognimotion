"""
world_model/rssm.py
===================
Recurrent State Space Model (RSSM) — DreamerV3 style.

Adaptado de worldmodel_dreaming.zip / cognitive_brain.zip:
  - Remove dependência de core.logger → logging padrão
  - Adiciona unimix_categorical no prior e posterior
  - Adiciona suporte a symlog nas predições de reward/done
  - Mantém 100% da lógica de KL-balanced, categorical z (32×32)
  - Adiciona BlockGRU opcional (divide estado em blocos independentes)
  - Expose: forward, imagine_step, prior_logits, observe
"""
from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from world_model.symlog import symlog, unimix_categorical

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# BlockGRUCell — divide o estado em blocos independentes (DreamerV3)
# ──────────────────────────────────────────────────────────────────────────────

class BlockGRUCell(nn.Module):
    """
    GRUCell particionado em `n_blocks` blocos independentes.
    Cada bloco tem dimensão hidden_dim // n_blocks.
    Não há atenção entre blocos — paralelismo puro.
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_blocks: int = 8) -> None:
        super().__init__()
        assert hidden_dim % n_blocks == 0, (
            f"hidden_dim ({hidden_dim}) deve ser divisível por n_blocks ({n_blocks})"
        )
        self.hidden_dim = hidden_dim
        self.n_blocks   = n_blocks
        self.block_dim  = hidden_dim // n_blocks

        # Um GRUCell por bloco (processamento paralelo via vmap equivalente manual)
        self.cells = nn.ModuleList([
            nn.GRUCell(input_dim, self.block_dim) for _ in range(n_blocks)
        ])

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        x: (B, input_dim)
        h: (B, hidden_dim)
        → h_new: (B, hidden_dim)

        Iteração scriptável (TorchScript não suporta zip sobre ModuleList):
        percorre self.cells por índice e fatia h manualmente por bloco.
        """
        h_new_blocks: List[torch.Tensor] = []
        i = 0
        for cell in self.cells:
            h_b = h[:, i * self.block_dim : (i + 1) * self.block_dim]
            h_new_blocks.append(cell(x, h_b))
            i += 1
        return torch.cat(h_new_blocks, dim=-1)


# ──────────────────────────────────────────────────────────────────────────────
# RSSM
# ──────────────────────────────────────────────────────────────────────────────

class RSSM(nn.Module):
    """
    Recurrent State Space Model com:
      - Estado determinístico h (GRU / BlockGRU)
      - Estado estocástico z (categorical 32×32 + straight-through + unimix)
      - Prior: P(z | h)
      - Posterior: Q(z | h, obs_enc)
      - KL-balanceado: alpha * KL(post_sg || prior) + (1-alpha) * KL(post || prior_sg)
      - Reward head + Done head (symlog)
      - Imagine step (prior only, sem observação)
    """

    def __init__(
        self,
        obs_enc_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        num_categories: int = 32,
        category_dim: int = 32,
        free_nats: float = 3.0,
        kl_balance: float = 0.8,
        unimix: float = 0.01,
        use_block_gru: bool = True,
        n_blocks: int = 8,
    ) -> None:
        super().__init__()

        self.obs_enc_dim    = obs_enc_dim
        self.action_dim     = action_dim
        self.hidden_dim     = hidden_dim
        self.num_categories = num_categories
        self.category_dim   = category_dim
        self.stochastic_dim = num_categories * category_dim   # 32*32 = 1024
        self.free_nats      = free_nats
        self.kl_balance     = kl_balance
        self.unimix         = unimix

        # Quando True, _sample_with_unimix usa argmax determinístico em vez de
        # torch.multinomial. multinomial depende do gerador global de RNG do
        # torch, que NÃO é inicializado quando o .pt roda via LibTorch embarcada
        # no Unreal — causando access violation/fastfail no forward. Para
        # inferência (export) ligamos esta flag; o treino mantém False (amostra).
        self.deterministic_sampling = False

        gru_input_dim = self.stochastic_dim + action_dim

        if use_block_gru:
            self.gru: nn.Module = BlockGRUCell(gru_input_dim, hidden_dim, n_blocks)
        else:
            self.gru = nn.GRUCell(gru_input_dim, hidden_dim)

        # Prior P(z | h)
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.stochastic_dim),
        )

        # Posterior Q(z | h, obs_enc)
        self.obs_proj       = nn.Linear(obs_enc_dim, hidden_dim)
        self.posterior_net  = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.stochastic_dim),
        )

        combined_dim = self.stochastic_dim + hidden_dim

        # Reward head (prediz symlog(reward))
        self.reward_head = nn.Sequential(
            nn.Linear(combined_dim, 256), nn.SiLU(),
            nn.Linear(256, 256),          nn.SiLU(),
            nn.Linear(256, 1),
        )

        # Done head (probabilidade de término)
        self.done_head = nn.Sequential(
            nn.Linear(combined_dim, 256), nn.SiLU(),
            nn.Linear(256, 1),
        )

        logger.info(
            f"RSSM | h={hidden_dim} | z=({num_categories}×{category_dim}={self.stochastic_dim}) | "
            f"obs_enc={obs_enc_dim} | action={action_dim} | "
            f"gru={'Block' if use_block_gru else 'Standard'} | unimix={unimix}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Forward completo (posterior — usa observação)
    # ──────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        prev_h:   torch.Tensor,         # (B, hidden_dim)
        prev_z:   torch.Tensor,         # (B, stochastic_dim)
        action:   torch.Tensor,         # (B, action_dim)
        obs_enc:  Optional[torch.Tensor] = None,  # (B, obs_enc_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            h       (B, hidden_dim)
            z       (B, stochastic_dim)
            reward  (B,)  — symlog scale
            done    (B,)  — sigmoid
            kl      (B,)  — KL divergência
            combined (B, stochastic_dim + hidden_dim)
        """
        gru_in = torch.cat([prev_z, action], dim=-1)
        h      = self.gru(gru_in, prev_h)

        prior_logits = self.prior_net(h)

        if obs_enc is not None:
            obs_h      = self.obs_proj(obs_enc)
            post_in    = torch.cat([h, obs_h], dim=-1)
            post_logits = self.posterior_net(post_in)

            z   = self._sample_with_unimix(post_logits)
            kl  = self._kl_balanced(post_logits, prior_logits)
        else:
            z   = self._sample_with_unimix(prior_logits)
            kl  = torch.zeros(h.size(0), device=h.device)

        combined = torch.cat([z, h], dim=-1)
        reward   = self.reward_head(combined).squeeze(-1)
        done     = torch.sigmoid(self.done_head(combined)).squeeze(-1)

        return h, z, reward, done, kl, combined

    # ──────────────────────────────────────────────────────────────────────────
    # imagine_step — prior puro (sem observação)
    # ──────────────────────────────────────────────────────────────────────────

    def imagine_step(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rollout imaginado: sem obs_enc → usa prior."""
        return self.forward(h, z, action, obs_enc=None)

    # ──────────────────────────────────────────────────────────────────────────
    # Sampling com straight-through + unimix
    # ──────────────────────────────────────────────────────────────────────────

    def _sample_with_unimix(self, logits: torch.Tensor) -> torch.Tensor:
        """
        1. Aplica unimix ao prior/posterior (evita colapso categórico)
        2. Amostra com Categorical
        3. Aplica straight-through para gradiente fluir
        """
        B = logits.size(0)
        # Mistura com uniforme
        probs_mixed = unimix_categorical(logits, self.unimix)           # (..., stoch_dim)
        probs_r     = probs_mixed.view(B, self.num_categories, self.category_dim)

        # Amostra categórica (índices) — implementação scriptável (sem
        # torch.distributions, que não é suportado pelo TorchScript). Usa
        # amostragem por CDF inversa, equivalente a Categorical.sample().
        flat_probs = probs_r.reshape(B * self.num_categories, self.category_dim)
        flat_probs = flat_probs / (flat_probs.sum(dim=-1, keepdim=True) + 1e-8)
        if self.deterministic_sampling:
            # Inferência: argmax (sem RNG). Evita o crash do torch.multinomial
            # sob LibTorch no Unreal (gerador global de RNG não inicializado).
            samples_flat = torch.argmax(flat_probs, dim=-1)
        else:
            samples_flat = torch.multinomial(flat_probs, num_samples=1).squeeze(-1)
        samples  = samples_flat.view(B, self.num_categories)            # (B, num_cat)
        one_hot  = F.one_hot(samples, self.category_dim).float()        # (B, num_cat, cat_dim)
        oh_flat  = one_hot.view(B, -1)                                  # (B, stoch_dim)
        p_flat   = probs_r.view(B, -1)

        # Straight-through: gradiente pela distribuição, valor amostrado
        z = oh_flat + (p_flat - p_flat.detach())
        return z

    # ──────────────────────────────────────────────────────────────────────────
    # KL-balanced (DreamerV3)
    # ──────────────────────────────────────────────────────────────────────────

    def _kl_balanced(
        self,
        post_logits:  torch.Tensor,
        prior_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        KL balanceado:
          loss = alpha * KL(post_sg || prior) + (1-alpha) * KL(post || prior_sg)

        Aplica free_nats por categoria para estabilidade.
        """
        B     = post_logits.size(0)
        shape = (B, self.num_categories, self.category_dim)

        pq = F.softmax(post_logits.view(*shape), dim=-1)
        pp = F.softmax(prior_logits.view(*shape), dim=-1)

        log_pq = F.log_softmax(post_logits.view(*shape), dim=-1)
        log_pp = F.log_softmax(prior_logits.view(*shape), dim=-1)

        # KL prior: gradiente flui para o prior (fixa post)
        kl_prior = (pq.detach() * (log_pq.detach() - log_pp)).sum(-1)
        # KL post: gradiente flui para o posterior (fixa prior)
        kl_post  = (pq          * (log_pq          - log_pp.detach())).sum(-1)

        kl_bal   = self.kl_balance * kl_prior + (1.0 - self.kl_balance) * kl_post
        kl_clip  = torch.clamp(kl_bal, min=self.free_nats / self.num_categories)
        return kl_clip.sum(-1)   # (B,) — soma sobre categorias

    # ──────────────────────────────────────────────────────────────────────────
    # Utilitários
    # ──────────────────────────────────────────────────────────────────────────

    def prior_logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.prior_net(h)

    def init_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retorna (h, z) zerados para inicialização de episódio."""
        h = torch.zeros(batch_size, self.hidden_dim,     device=device)
        z = torch.zeros(batch_size, self.stochastic_dim, device=device)
        return h, z

    def observe(
        self,
        h:       torch.Tensor,
        z:       torch.Tensor,
        action:  torch.Tensor,
        obs_enc: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Atualiza estado latente com observação real.
        Retorna (h_new, z_new, kl).
        """
        h_new, z_new, _, _, kl, _ = self.forward(h, z, action, obs_enc)
        return h_new, z_new, kl
