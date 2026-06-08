from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from config import EncoderConfig
from data.pose_frame import PoseFrame


class BoneEncoder(nn.Module):
    def __init__(self, bone_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(bone_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalTransformerEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int,
                 max_seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        x = x + self.pos_embed(positions)
        x = self.transformer(x, src_key_padding_mask=mask)
        return self.output_norm(x[:, -1])


class PoseEncoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config

        root_input_dim = 3 + 4 + 3 + 3
        self.root_proj = nn.Linear(root_input_dim, config.pose_dim // 2)

        self.bone_encoder = BoneEncoder(config.bone_dim, 32)
        bone_total = 32 * config.bone_count
        self.bone_proj = nn.Linear(bone_total, config.pose_dim // 2)

        self.root_norm = nn.LayerNorm(config.pose_dim // 2)
        self.bone_norm = nn.LayerNorm(config.pose_dim // 2)

        self.temporal_encoder = TemporalTransformerEncoder(
            d_model=config.pose_dim,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        )

        self.embedding_head = nn.Sequential(
            nn.Linear(config.pose_dim, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
            nn.Tanh(),
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(config.pose_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.style_head = nn.Sequential(
            nn.Linear(config.pose_dim, 64),
            nn.GELU(),
            nn.Linear(64, 9),
        )

    def encode_single_frame(
        self,
        root_features: torch.Tensor,
        bone_features: torch.Tensor,
    ) -> torch.Tensor:
        r = self.root_norm(self.root_proj(root_features))
        b = self.bone_norm(self.bone_proj(bone_features))
        return torch.cat([r, b], dim=-1)

    def forward(
        self,
        root_seq: torch.Tensor,
        bone_seq: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = root_seq.shape

        # bone_seq: [B, T, bone_count * bone_dim] — reshape to per-bone for BoneEncoder
        bone_per_bone = bone_seq.view(B * T, self.config.bone_count, self.config.bone_dim)
        # Apply BoneEncoder to each bone independently, then flatten → [B*T, bone_count*32]
        bone_encoded = self.bone_encoder(bone_per_bone)          # [B*T, bone_count, 32]
        bone_flat_2d = bone_encoded.view(B * T, -1)              # [B*T, bone_count*32 = 288]

        root_seq_2d = root_seq.view(B * T, -1)
        root_proj_2d = self.root_proj(root_seq_2d)
        bone_proj_2d = self.bone_proj(bone_flat_2d)
        seq = torch.cat([root_proj_2d, bone_proj_2d], dim=-1).view(B, T, -1)

        context = self.temporal_encoder(seq, mask=mask)
        embedding = self.embedding_head(context)
        confidence = self.confidence_head(context)
        style_logits = self.style_head(context)

        return embedding, confidence.squeeze(-1), style_logits

    @torch.no_grad()
    def encode_frame(self, frame: PoseFrame, device: str = "cpu") -> tuple[np.ndarray, float]:
        obs = frame.to_observation_vector()
        if not np.isfinite(obs).all():
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        tensor = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

        root_dim = 3 + 4 + 3 + 3
        bone_dim = self.config.bone_count * self.config.bone_dim

        root_t = tensor[:, :, :root_dim]
        bone_t = tensor[:, :, root_dim:root_dim + bone_dim]

        self.eval()
        embedding, confidence, _ = self.forward(root_t, bone_t)
        return (embedding.squeeze(0).cpu().numpy().astype(np.float32),
                float(np.clip(confidence.item(), 0.0, 1.0)))
