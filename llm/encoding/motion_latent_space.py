from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from config import EncoderConfig


class MotionLatentSpace(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        pose_dim = config.embedding_dim
        traj_dim = config.trajectory_dim
        latent_dim = config.embedding_dim

        self.fusion = nn.Sequential(
            nn.Linear(pose_dim + traj_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        self.style_condition = nn.Embedding(9, 64)
        self.style_proj = nn.Linear(latent_dim + 64, latent_dim)

        self.mu_head    = nn.Linear(latent_dim, latent_dim)
        self.logvar_head = nn.Linear(latent_dim, latent_dim)

        self.decode_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, pose_dim),
            nn.Tanh(),
        )

        self.style_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, 9),
        )

        self.trajectory_decode = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, 6 * 10),
        )

    def encode(
        self,
        pose_emb: torch.Tensor,
        traj_emb: torch.Tensor,
        style: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fused = self.fusion(torch.cat([pose_emb, traj_emb], dim=-1))
        if style is not None:
            style_emb = self.style_condition(style)
            fused = self.style_proj(torch.cat([fused, style_emb], dim=-1))
        mu     = self.mu_head(fused)
        logvar = self.logvar_head(fused)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pose_out = self.decode_head(z)
        traj_out = self.trajectory_decode(z).view(-1, 6, 10)
        return pose_out, traj_out

    def classify_style(self, z: torch.Tensor) -> torch.Tensor:
        return self.style_head(z)

    def forward(
        self,
        pose_emb: torch.Tensor,
        traj_emb: torch.Tensor,
        style: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(pose_emb, traj_emb, style)
        z = self.reparameterize(mu, logvar)
        pose_recon, traj_recon = self.decode(z)
        return z, mu, logvar, pose_recon, traj_recon

    @staticmethod
    def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
