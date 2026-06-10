"""
export_torchscript.py
=====================
Exporta o pipeline de inferência completo (treinado em Python) para UM único
arquivo TorchScript (.pt) que o plugin C++ carrega via LibTorch e executa
nativamente dentro do Unreal Engine — SEM rede, SEM Python em runtime.

O que é exportado (somente o caminho de INFERÊNCIA; o treino continua em Python):
  - RSSM.forward  (passo recorrente do world model: h,z,ação,obs → h',z')
  - ActorNet      (latente → logits de ação)
  - PoseDecoder   (latente → 89 bones, a animação gerada)

Tudo é embrulhado numa classe `CognitiveInferenceModule` cujo `forward` faz o
passo completo de uma vez. O C++ chama um único `forward()` por frame.

Uso:
    python export_torchscript.py --checkpoint checkpoints/policy_v000002.pt \
                                 --output CognitiveModel.pt

O arquivo gerado vai em:  <Plugin>/Content/Models/CognitiveModel.pt
e é carregado pelo UCognitiveInferenceComponent (C++).
"""
from __future__ import annotations

import argparse
import torch
import torch.nn as nn
from typing import Tuple

from config import DEFAULT_CONFIG
from world_model.world_model import WorldModel
from planning.policy import Policy
from encoding.perception_features import PERCEPTION_DIM


# ─────────────────────────────────────────────────────────────────────────────
# Módulo de inferência unificado — scriptável (torch.jit.script)
# ─────────────────────────────────────────────────────────────────────────────
class CognitiveInferenceModule(nn.Module):
    """
    Encapsula um passo completo de inferência autônoma do NPC.

    Entrada do forward:
      prev_h   (1, hidden_dim)        estado recorrente anterior
      prev_z   (1, stochastic_dim)    estado estocástico anterior
      action   (1, action_dim)        ação anterior (one-hot)
      obs_enc  (1, obs_enc_dim)       embedding observado (256-d); pode ser zeros

    Saída:
      h_new       (1, hidden_dim)
      z_new       (1, stochastic_dim)
      action_idx  (1,)   índice da ação escolhida (argmax dos logits)
      pose        (1, num_bones*7)  poses dos bones (loc3+quat4 por bone)

    O C++ mantém h/z entre frames (estado recorrente do NPC) e aplica `pose`
    no esqueleto + traduz `action_idx` em movimento da cápsula.
    """

    def __init__(self, rssm: nn.Module, actor: nn.Module, pose_decoder: nn.Module,
                 num_bones: int):
        super().__init__()
        self.rssm = rssm
        self.actor = actor
        self.pose_decoder = pose_decoder
        self.num_bones = num_bones

    def forward(
        self,
        prev_h:  torch.Tensor,
        prev_z:  torch.Tensor,
        action:  torch.Tensor,
        obs_enc: torch.Tensor,
        use_obs: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. Passo do RSSM (com ou sem observação)
        if use_obs:
            h, z, _, _, _, combined = self.rssm.forward(prev_h, prev_z, action, obs_enc)
        else:
            h, z, _, _, _, combined = self.rssm.forward(prev_h, prev_z, action, None)

        # 2. Política → índice de ação (argmax determinístico p/ inferência)
        logits = self.actor(combined)
        action_idx = torch.argmax(logits, dim=-1)

        # 3. PoseDecoder → poses dos bones, com quaternions normalizados
        pose = self.pose_decoder(combined)
        pose = self._normalize_quats(pose)

        return h, z, action_idx, pose

    def _normalize_quats(self, pose_flat: torch.Tensor) -> torch.Tensor:
        # pose_flat: (B, num_bones*7) → normaliza o quaternion (índices 3:7) por bone
        B = pose_flat.size(0)
        p = pose_flat.view(B, self.num_bones, 7)
        loc = p[:, :, 0:3]
        quat = p[:, :, 3:7]
        quat = quat / (torch.norm(quat, p=2, dim=-1, keepdim=True) + 1e-8)
        out = torch.cat([loc, quat], dim=-1)
        return out.view(B, self.num_bones * 7)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="checkpoint .pt do treino")
    ap.add_argument("--output", default="CognitiveModel.pt", help="saída TorchScript")
    args = ap.parse_args()

    cfg = DEFAULT_CONFIG
    wm_cfg = cfg.world_model
    ac_cfg = cfg.actor_critic
    num_bones = getattr(wm_cfg, "num_bones", 89)

    # Reconstrói os módulos com as mesmas dimensões do treino
    world_model = WorldModel(
        obs_enc_dim=cfg.encoder.embedding_dim + PERCEPTION_DIM,
        action_dim=ac_cfg.action_dim,
        hidden_dim=wm_cfg.rssm_hidden_dim,
        num_categories=wm_cfg.rssm_num_categories,
        category_dim=wm_cfg.rssm_category_dim,
        num_bones=num_bones,
    )
    policy = Policy(
        combined_dim=world_model.combined_dim,
        action_dim=ac_cfg.action_dim,
        hidden=256,
    )

    # Carrega pesos treinados
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    if "rssm" in state:
        world_model.rssm.load_state_dict(state["rssm"], strict=False)
    if "pose_decoder" in state:
        world_model.pose_decoder.load_state_dict(state["pose_decoder"], strict=False)
    if "actor" in state:
        policy.actor.load_state_dict(state["actor"], strict=False)
    print(f"Pesos carregados de {args.checkpoint} (versão {ckpt.get('version','?')})")

    # Monta o módulo unificado e coloca em modo avaliação
    module = CognitiveInferenceModule(
        rssm=world_model.rssm,
        actor=policy.actor,
        pose_decoder=world_model.pose_decoder,
        num_bones=num_bones,
    )
    module.eval()

    # Script (não trace — preserva o if/else de use_obs)
    scripted = torch.jit.script(module)
    scripted.save(args.output)
    print(f"TorchScript salvo em: {args.output}")
    print(f"  hidden_dim={wm_cfg.rssm_hidden_dim}  "
          f"stochastic_dim={world_model.stochastic_dim}  "
          f"action_dim={ac_cfg.action_dim}  num_bones={num_bones}")
    print("Copie este arquivo para <Plugin>/Content/Models/CognitiveModel.pt")


if __name__ == "__main__":
    main()
