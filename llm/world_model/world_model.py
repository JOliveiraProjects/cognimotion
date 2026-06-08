"""
world_model/world_model.py
==========================
WorldModel completo: RSSM + Observation Decoder + Reward/Done heads.

Arquitetura DreamerV3:
  PoseEncoder(obs) → obs_enc (256-d)
  RSSM.posterior(h, z, a, obs_enc) → h_new, z_new, reward, done, kl
  ObsDecoder(combined=[z,h]) → reconstructed obs_enc (256-d)

O decoder reconstrói o obs_enc (embedding 256-d do PoseEncoder), não a observação
raw, o que mantém compatibilidade com a stack de encoders existente.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from world_model.pose_decoder import PoseDecoder
import torch.nn.functional as F

from world_model.rssm import RSSM
from world_model.symlog import symlog, symexp, EMANormalizer

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Observation Decoder
# ──────────────────────────────────────────────────────────────────────────────

class ObservationDecoder(nn.Module):
    """
    Decodificador MLP: estado latente combinado → obs_enc reconstruído.
    Recebe combined = cat([z, h]) e reconstrói o obs_enc (256-d do PoseEncoder).
    """

    def __init__(self, combined_dim: int, obs_enc_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, obs_enc_dim),
        )

    def forward(self, combined: torch.Tensor) -> torch.Tensor:
        return self.net(combined)


# ──────────────────────────────────────────────────────────────────────────────
# WorldModel
# ──────────────────────────────────────────────────────────────────────────────

class WorldModel(nn.Module):
    """
    Encapsula RSSM + ObsDecoder + reward/done heads.

    Responsabilidades:
      1. Codificar obs_enc via RSSM posterior (observe)
      2. Imaginar rollouts com prior (imagine_step)
      3. Decodificar estado latente → obs_enc reconstruído
      4. Predizer reward e done a partir do estado latente

    O WorldModel NÃO treina por si só — isso é feito pelo DreamerTrainer.
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
        num_bones: int = 89,
    ) -> None:
        super().__init__()
        self.num_bones = num_bones

        self.rssm = RSSM(
            obs_enc_dim=obs_enc_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_categories=num_categories,
            category_dim=category_dim,
            free_nats=free_nats,
            kl_balance=kl_balance,
            unimix=unimix,
            use_block_gru=use_block_gru,
            n_blocks=n_blocks,
        )

        combined_dim = self.rssm.stochastic_dim + hidden_dim
        self.decoder = ObservationDecoder(combined_dim, obs_enc_dim)

        # Cabeça generativa de POSE: estado latente → poses dos 89 bones.
        # É isto que faz o world model GERAR animação (não só decidir ação).
        self.pose_decoder = PoseDecoder(combined_dim, num_bones=num_bones)

        # Normalização EMA para rewards (DreamerV3)
        self._reward_ema = EMANormalizer(decay=0.99, clip=10.0)

        logger.info(
            f"WorldModel | combined_dim={combined_dim} "
            f"| obs_enc_dim={obs_enc_dim} | action_dim={action_dim}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Properties de dimensão (conveniência para o trainer)
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def hidden_dim(self) -> int:
        return self.rssm.hidden_dim

    @property
    def stochastic_dim(self) -> int:
        return self.rssm.stochastic_dim

    @property
    def combined_dim(self) -> int:
        return self.rssm.stochastic_dim + self.rssm.hidden_dim

    # ──────────────────────────────────────────────────────────────────────────
    # Core APIs
    # ──────────────────────────────────────────────────────────────────────────

    def observe(
        self,
        h:       torch.Tensor,
        z:       torch.Tensor,
        action:  torch.Tensor,
        obs_enc: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Atualiza estado com observação real.
        Returns: (h_new, z_new, reward, done, kl, combined)
        """
        return self.rssm.forward(h, z, action, obs_enc=obs_enc)

    def imagine_step(
        self,
        h:      torch.Tensor,
        z:      torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Passo de imaginação (sem observação real)."""
        return self.rssm.imagine_step(h, z, action)

    def decode(self, combined: torch.Tensor) -> torch.Tensor:
        """Decodifica estado combinado → obs_enc reconstruído."""
        return self.decoder(combined)

    def init_state(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.rssm.init_state(batch_size, device)

    # ──────────────────────────────────────────────────────────────────────────
    # Sequence rollout (usado pelo trainer para computar loss em T passos)
    # ──────────────────────────────────────────────────────────────────────────

    def rollout_sequence(
        self,
        obs_enc_seq: torch.Tensor,    # (B, T, obs_enc_dim)
        action_seq:  torch.Tensor,    # (B, T, action_dim)
        init_h:      Optional[torch.Tensor] = None,
        init_z:      Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Rollout completo de T passos usando observações reais (posterior).

        Returns dict com tensores (B, T, *):
            h_seq, z_seq, combined_seq, reward_seq, done_seq, kl_seq,
            recon_seq (decodificado)
        """
        B, T, _ = obs_enc_seq.shape
        device  = obs_enc_seq.device

        h = init_h if init_h is not None else torch.zeros(B, self.hidden_dim, device=device)
        z = init_z if init_z is not None else torch.zeros(B, self.stochastic_dim, device=device)

        h_list, z_list, comb_list = [], [], []
        rew_list, done_list, kl_list = [], [], []

        for t in range(T):
            h, z, rew, done, kl, combined = self.rssm.forward(
                h, z, action_seq[:, t], obs_enc=obs_enc_seq[:, t]
            )
            h_list.append(h);        z_list.append(z)
            comb_list.append(combined)
            rew_list.append(rew);    done_list.append(done)
            kl_list.append(kl)

        combined_seq = torch.stack(comb_list, dim=1)   # (B, T, combined_dim)
        recon_seq    = self.decoder(combined_seq)       # (B, T, obs_enc_dim)

        return {
            "h":        torch.stack(h_list, dim=1),
            "z":        torch.stack(z_list, dim=1),
            "combined": combined_seq,
            "reward":   torch.stack(rew_list,  dim=1),
            "done":     torch.stack(done_list, dim=1),
            "kl":       torch.stack(kl_list,   dim=1),
            "recon":    recon_seq,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Reward normalizer
    # ──────────────────────────────────────────────────────────────────────────

    def normalize_reward(self, reward: torch.Tensor) -> torch.Tensor:
        """Aplica symlog + EMA normalization para estabilizar o reward."""
        r_sym = symlog(reward)
        return self._reward_ema.update_and_normalize(r_sym)
