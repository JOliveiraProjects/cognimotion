from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from config import EncoderConfig
from data.pose_frame import Trajectory


class TrajectoryEncoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config  # expose config for callers (e.g. online_imitation_learner)
        sample_dim = config.sample_dim
        hidden = 128

        self.sample_proj = nn.Sequential(
            nn.Linear(sample_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            dim_feedforward=hidden * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden, config.trajectory_dim),
            nn.LayerNorm(config.trajectory_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = self.sample_proj(x)
        ctx  = self.transformer(proj)
        return self.output_proj(ctx.mean(dim=1))

    @torch.no_grad()
    def encode_trajectory(self, traj: Trajectory, n_samples: int = 6,
                           device: str = "cpu") -> np.ndarray:
        arr = traj.to_numpy(n_samples)
        t   = torch.from_numpy(arr).unsqueeze(0).to(device)
        out = self.forward(t)
        return out.squeeze(0).cpu().numpy()
