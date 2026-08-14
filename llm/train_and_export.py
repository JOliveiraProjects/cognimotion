"""
train_and_export.py — Gera o CognitiveModel.pt FUNCIONAL a partir de um
checkpoint de treino, pronto para o LibTorch carregar no C++.

Fluxo:
  1. Carrega o checkpoint acumulado do treino (checkpoints/policy_Default.pt),
     que contém os pesos APRENDIDOS (rssm, pose_decoder, actor).
  2. Embrulha no CognitiveInferenceModule (forward compatível com o C++:
     (h,z,action,obs,use_obs) → (h',z',action_idx,pose)).
  3. torch.jit.script + save no caminho que o C++ procura:
     <Plugin>/Content/Models/CognitiveModel.pt

Uso:
  python train_and_export.py --checkpoint checkpoints/policy_Default.pt \
                             --output ../CognitiveAgent/Content/Models/CognitiveModel.pt

Se não houver checkpoint (treino ainda não rodou), use --init para exportar
um modelo de pesos iniciais (estrutura correta, comportamento não treinado) —
útil só para validar o carregamento no C++.
"""
from __future__ import annotations

import argparse
import os
import torch

from config import DEFAULT_CONFIG
from world_model.world_model import WorldModel
from planning.policy import Policy
from encoding.perception_features import PERCEPTION_DIM
from export_torchscript import CognitiveInferenceModule


def build_modules():
    cfg = DEFAULT_CONFIG
    wm = WorldModel(
        obs_enc_dim=cfg.encoder.embedding_dim + PERCEPTION_DIM,
        action_dim=cfg.actor_critic.action_dim,
        hidden_dim=cfg.world_model.rssm_hidden_dim,
        num_categories=cfg.world_model.rssm_num_categories,
        category_dim=cfg.world_model.rssm_category_dim,
        num_bones=getattr(cfg.world_model, "num_bones", 89),
    )
    policy = Policy(combined_dim=wm.combined_dim,
                    action_dim=cfg.actor_critic.action_dim, hidden=256)
    return cfg, wm, policy


def load_trained_weights(wm, policy, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    loaded = []
    if "rssm" in state:
        wm.rssm.load_state_dict(state["rssm"], strict=False); loaded.append("rssm")
    if "pose_decoder" in state:
        wm.pose_decoder.load_state_dict(state["pose_decoder"], strict=False); loaded.append("pose_decoder")
    if "actor" in state:
        policy.actor.load_state_dict(state["actor"], strict=False); loaded.append("actor")
    return ckpt.get("version", "?"), loaded


def export(output_path, wm, policy, num_bones):
    module = CognitiveInferenceModule(
        rssm=wm.rssm, actor=policy.actor,
        pose_decoder=wm.pose_decoder, num_bones=num_bones)
    module.eval()
    # Liga argmax determinístico no RSSM: torch.multinomial crasha sob LibTorch
    # no Unreal (RNG global não inicializado → access violation/fastfail).
    wm.rssm.deterministic_sampling = True
    scripted = torch.jit.script(module)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    scripted.save(output_path)
    return output_path


def verify(output_path):
    """Carrega o .pt como o C++ faz e roda o forward para garantir que funciona."""
    m = torch.jit.load(output_path); m.eval()
    H = torch.zeros(1, 512); Z = torch.zeros(1, 1024)
    A = torch.zeros(1, 9); A[0][0] = 1.0
    Obs = torch.zeros(1, 256 + PERCEPTION_DIM)
    with torch.no_grad():
        h, z, idx, pose = m.forward(H, Z, A, Obs, True)
    ok = (tuple(h.shape) == (1, 512) and tuple(z.shape) == (1, 1024)
          and pose.numel() == num_bones_global * 7 and 0 <= int(idx.item()) < 9)
    return ok, int(idx.item()), pose.numel()


num_bones_global = 89


def main():
    global num_bones_global
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/policy_Default.pt")
    ap.add_argument("--output", default="CognitiveModel.pt")
    ap.add_argument("--init", action="store_true",
                    help="exporta pesos iniciais se não houver checkpoint")
    args = ap.parse_args()

    cfg, wm, policy = build_modules()
    num_bones_global = getattr(cfg.world_model, "num_bones", 89)

    if os.path.exists(args.checkpoint):
        ver, loaded = load_trained_weights(wm, policy, args.checkpoint)
        print(f"Pesos TREINADos carregados de {args.checkpoint} (v{ver}): {loaded}")
    elif args.init:
        print("Sem checkpoint — exportando PESOS INICIAIS (não treinado).")
    else:
        print(f"ERRO: checkpoint '{args.checkpoint}' não existe. Treine primeiro "
              f"ou use --init para exportar pesos iniciais.")
        return 1

    path = export(args.output, wm, policy, num_bones_global)
    ok, idx, nposes = verify(path)
    print(f"TorchScript salvo em: {path}")
    print(f"Verificação (carrega + forward): {'OK' if ok else 'FALHOU'} "
          f"| ação={idx} | pose_floats={nposes}")
    print(f"Copie para <Plugin>/Content/Models/CognitiveModel.pt")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
