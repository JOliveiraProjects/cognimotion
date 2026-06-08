from __future__ import annotations
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Dict, Any
from config import LearningConfig, EncoderConfig, MemoryConfig
from encoding.pose_encoder import PoseEncoder
from encoding.trajectory_encoder import TrajectoryEncoder
from encoding.motion_latent_space import MotionLatentSpace
from encoding.trajectory_encoder import TrajectoryEncoder
from memory.motion_replay_buffer import MotionReplayBuffer


class OnlineImitationLearner:
    def __init__(
        self,
        encoder_config: EncoderConfig,
        learning_config: LearningConfig,
        device: str = "cpu",
    ):
        self.config  = learning_config
        self.device  = torch.device(device)

        self.pose_encoder      = PoseEncoder(encoder_config).to(self.device)
        self.trajectory_encoder = TrajectoryEncoder(encoder_config).to(self.device)
        self.latent_space      = MotionLatentSpace(encoder_config).to(self.device)

        replay_cfg = MemoryConfig(
            replay_capacity=100000,
            priority_alpha=0.6,
            priority_beta=0.4,
            confidence_weight=0.3,
            recency_weight=0.4,
            diversity_weight=0.3,
        )
        self.replay_buffer = MotionReplayBuffer(
            config=replay_cfg,
            embedding_dim=encoder_config.embedding_dim,
        )

        all_params = (
            list(self.pose_encoder.parameters()) +
            list(self.trajectory_encoder.parameters()) +
            list(self.latent_space.parameters())
        )
        self.optimizer = optim.AdamW(
            all_params,
            lr=learning_config.learning_rate,
            weight_decay=1e-4,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=1000, T_mult=2)

        self._step = 0
        self._train_count = 0
        self._metrics: Dict[str, float] = {}

    def train_step(self, batch_size: int = 64) -> Optional[Dict[str, float]]:
        batch = self.replay_buffer.sample(batch_size)
        if batch is None:
            return None

        embeddings = torch.from_numpy(
            np.stack([e.embedding for e in batch], axis=0)
        ).to(self.device)

        styles = torch.tensor(
            [e.style for e in batch], dtype=torch.long, device=self.device)

        pose_emb = embeddings[:, :self.pose_encoder.config.embedding_dim]
        # ReplayEntry stores pose embeddings only — no raw trajectory.
        # Use zero vector as neutral prior for trajectory conditioning.
        traj_emb = torch.zeros(
            batch_size,
            self.trajectory_encoder.config.trajectory_dim,
            device=self.device,
            dtype=torch.float32)

        z, mu, logvar, pose_recon, traj_recon = self.latent_space(
            pose_emb, traj_emb, styles)

        recon_loss = nn.functional.mse_loss(pose_recon, pose_emb)
        kl_loss    = MotionLatentSpace.kl_loss(mu, logvar)
        style_logits = self.latent_space.classify_style(z)
        style_loss = nn.functional.cross_entropy(style_logits, styles)

        confidences = torch.tensor(
            [e.confidence for e in batch], dtype=torch.float32, device=self.device)
        weights = confidences / (confidences.sum() + 1e-8)
        weighted_recon = (recon_loss * weights.unsqueeze(-1)).sum()

        entropy_loss = -torch.mean(torch.distributions.Normal(mu, (0.5 * logvar).exp()).entropy())

        total_loss = (
            self.config.imitation_loss_weight  * weighted_recon +
            0.01                               * kl_loss +
            self.config.style_loss_weight      * style_loss +
            self.config.entropy_weight         * entropy_loss
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.latent_space.parameters(), self.config.gradient_clip)
        self.optimizer.step()
        self.scheduler.step()

        self._step += 1
        self._train_count += 1

        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_recon": float(recon_loss.item()),
            "loss_kl": float(kl_loss.item()),
            "loss_entropy": float(entropy_loss.item()),
            "step": self._step,
            "lr": float(self.optimizer.param_groups[0]["lr"]),
        }
        self._metrics = metrics
        return metrics

    def infer(
        self,
        pose_embedding: np.ndarray,
        traj_embedding: np.ndarray,
        style: int = 0,
    ) -> tuple[np.ndarray, float]:
        self.pose_encoder.eval()
        self.trajectory_encoder.eval()
        self.latent_space.eval()

        with torch.no_grad():
            pe = torch.from_numpy(pose_embedding.astype(np.float32)).unsqueeze(0).to(self.device)
            te = torch.from_numpy(traj_embedding.astype(np.float32)).unsqueeze(0).to(self.device)
            st = torch.tensor([style], dtype=torch.long, device=self.device)

            mu, logvar = self.latent_space.encode(pe, te, st)
            z = mu

            pose_recon, _ = self.latent_space.decode(z)
            emb_norm = float(mu.norm(dim=-1).item())
            confidence = float(1.0 / (1.0 + max(emb_norm - 1.0, 0.0) * 0.1))
            confidence = float(np.clip(confidence, 0.0, 1.0))

        return pose_recon.squeeze(0).cpu().numpy(), min(confidence, 1.0)

    def get_metrics(self) -> Dict[str, Any]:
        stats = self.replay_buffer.get_stats()
        return {**self._metrics, **stats, "train_count": self._train_count}

    def save(self, path: str) -> None:
        torch.save({
            "pose_encoder": self.pose_encoder.state_dict(),
            "trajectory_encoder": self.trajectory_encoder.state_dict(),
            "latent_space": self.latent_space.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self._step,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.pose_encoder.load_state_dict(ckpt["pose_encoder"])
        self.trajectory_encoder.load_state_dict(ckpt["trajectory_encoder"])
        self.latent_space.load_state_dict(ckpt["latent_space"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._step = ckpt.get("step", 0)
