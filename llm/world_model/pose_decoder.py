"""
world_model/pose_decoder.py
===========================
Cabeça decodificadora de POSE do world model.

O world model padrão (DreamerV3) só reconstrói o embedding 256-d. Isso permite
ao NPC DECIDIR ações, mas não GERAR animação — por isso, no modo autônomo, o
sistema antigo só repassava os bones congelados do líder.

Esta cabeça mapeia o estado latente do RSSM → as poses reais dos bones do
esqueleto (89 bones × loc(3)+quat(4) = 7 floats por bone). Treinada sobre as
poses do líder observadas, ela APRENDE a animação. No modo Inferring, o RSSM
roda para frente (imaginação) e esta cabeça decodifica a pose de cada frame —
o NPC gera o movimento sozinho, sem copiar o líder em tempo real.

Layout de saída por bone: [loc_x, loc_y, loc_z, q_x, q_y, q_z, q_w]
O quaternion é normalizado na decodificação. Escala assumida 1.0 (esqueleto
rígido) — pode ser estendida se necessário.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


BONE_POSE_DIM = 7  # loc(3) + quat(4)


class PoseDecoder(nn.Module):
    """
    Decodifica estado latente combinado [z, h] → poses de N bones.

    Saída: (B, num_bones * 7). Loc livre; quaternion normalizado.
    """

    def __init__(self, latent_dim: int, num_bones: int, hidden: int = 512):
        super().__init__()
        self.num_bones = num_bones
        self.out_dim   = num_bones * BONE_POSE_DIM

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.out_dim),
        )

    def forward(self, combined: torch.Tensor) -> torch.Tensor:
        """combined: (B, latent_dim) ou (B, T, latent_dim) → (..., num_bones*7)."""
        return self.net(combined)

    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def normalize_quaternions(pose_flat: torch.Tensor, num_bones: int) -> torch.Tensor:
        """
        Normaliza os quaternions de cada bone na saída achatada.
        pose_flat: (..., num_bones*7) → mesma forma, quats unitários.
        """
        shape = pose_flat.shape[:-1]
        p = pose_flat.view(*shape, num_bones, BONE_POSE_DIM)
        loc  = p[..., 0:3]
        quat = p[..., 3:7]
        quat = quat / (quat.norm(dim=-1, keepdim=True) + 1e-8)
        out = torch.cat([loc, quat], dim=-1)
        return out.view(*shape, num_bones * BONE_POSE_DIM)

    # ──────────────────────────────────────────────────────────────────────────
    def decode_to_transforms(self, combined: torch.Tensor) -> list:
        """
        Decodifica e converte para a lista de dicts que build_motion_response
        espera: [{"location":[x,y,z], "rotation":[x,y,z,w], "scale":[1,1,1]}, ...].
        Aceita combined de batch 1.
        """
        with torch.no_grad():
            raw = self.forward(combined)
            raw = self.normalize_quaternions(raw, self.num_bones)
        arr = raw.squeeze(0).cpu().numpy().astype(np.float32)
        arr = arr.reshape(self.num_bones, BONE_POSE_DIM)

        bones = []
        for i in range(self.num_bones):
            loc  = arr[i, 0:3].tolist()
            quat = arr[i, 3:7].tolist()  # x,y,z,w
            bones.append({
                "location": loc,
                "rotation": quat,
                "scale":    [1.0, 1.0, 1.0],
            })
        return bones


def bones_to_target_tensor(
    bone_transforms: list, num_bones: int, device: str = "cpu"
) -> torch.Tensor:
    """
    Converte a lista de bones do líder (matrizes 4×4 ou dicts) em um tensor
    alvo (num_bones*7) para treinar o PoseDecoder. Preenche/corta para num_bones.
    """
    out = np.zeros((num_bones, BONE_POSE_DIM), dtype=np.float32)
    out[:, 6] = 1.0  # quat w=1 (identidade) como padrão

    for i in range(min(len(bone_transforms), num_bones)):
        bt = bone_transforms[i]
        if isinstance(bt, np.ndarray) and bt.shape == (4, 4):
            loc = bt[:3, 3]
            quat = _mat3_to_quat_xyzw(bt[:3, :3])
        elif isinstance(bt, dict):
            loc  = np.asarray(bt.get("location", [0, 0, 0]), dtype=np.float32)
            quat = np.asarray(bt.get("rotation", [0, 0, 0, 1]), dtype=np.float32)
        else:
            continue
        out[i, 0:3] = loc
        out[i, 3:7] = quat

    return torch.from_numpy(out.reshape(-1)).to(device)


def _mat3_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Matriz de rotação 3×3 → quaternion [x, y, z, w] (normaliza colunas)."""
    R = R.copy().astype(np.float64)
    for c in range(3):
        n = np.linalg.norm(R[:, c])
        if n > 1e-8:
            R[:, c] /= n
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float32)
    return q / (np.linalg.norm(q) + 1e-8)
