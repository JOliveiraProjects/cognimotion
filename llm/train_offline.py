"""
train_offline.py — Pipeline de treino OFFLINE (sem o UE5 conectado).

Resolve a limitação: o treino normal depende de sequências chegando do Unreal
em tempo real. Para um jogo lançado, você treina offline e exporta o .pt.

Fontes de dados (escolha uma):
  --dataset           usa o dataset sintético de interações (armas, bola,
                      veículos, ameaças, trânsito) já incluído no projeto.
  --replay ARQ.npz    carrega sequências gravadas (obs, action, reward, done)
                      de um arquivo .npz (ver formato em save_replay()).

Ao final, treina o World Model + Policy por --steps e exporta CognitiveModel.pt.

Uso típico:
  python train_offline.py --dataset --steps 2000 \
      --output ../CognitiveAgent/Content/Models/CognitiveModel.pt

  python train_offline.py --replay gravacoes/sessao1.npz --steps 5000 \
      --output ../CognitiveAgent/Content/Models/CognitiveModel.pt
"""
from __future__ import annotations

import argparse
import os
import threading
import numpy as np
import torch

from config import DEFAULT_CONFIG
from world_model.world_model import WorldModel
from world_model.dreamer_trainer import DreamerTrainer
from planning.policy import Policy
from runtime.sequence_buffer import SequenceBuffer
from encoding.perception_features import PERCEPTION_DIM


def build_stack():
    cfg = DEFAULT_CONFIG
    obs_dim = cfg.encoder.embedding_dim + PERCEPTION_DIM
    num_bones = getattr(cfg.world_model, "num_bones", 89)
    wm = WorldModel(
        obs_enc_dim=obs_dim,
        action_dim=cfg.actor_critic.action_dim,
        hidden_dim=cfg.world_model.rssm_hidden_dim,
        num_categories=cfg.world_model.rssm_num_categories,
        category_dim=cfg.world_model.rssm_category_dim,
        num_bones=num_bones,
    )
    policy = Policy(combined_dim=wm.combined_dim,
                    action_dim=cfg.actor_critic.action_dim, hidden=256)
    buf = SequenceBuffer(
        capacity=cfg.world_model.sequence_buffer_capacity,
        obs_dim=obs_dim, action_dim=cfg.actor_critic.action_dim,
        seq_len=cfg.world_model.seq_len)
    from learning.policy_registry import PolicyRegistry
    registry = PolicyRegistry(save_dir="checkpoints", category="Default")
    lock = threading.RLock()
    trainer = DreamerTrainer(world_model=wm, actor_critic=policy,
                             sequence_buffer=buf, policy_registry=registry,
                             config=cfg, device=str(cfg.device), model_lock=lock)
    return cfg, wm, policy, buf, trainer, obs_dim, num_bones


def populate_from_dataset(buf, obs_dim):
    from datasets.dataset_registry import DatasetRegistry
    from config import DatasetConfig
    reg = DatasetRegistry(DatasetConfig())
    n = reg.load_into_buffer(buf, obs_dim=obs_dim)
    return n


def populate_from_replay(buf, path, obs_dim, action_dim):
    """Carrega sequências de um .npz. Espera arrays: obs, action, reward, done.
    obs: (T, obs_dim); action: (T, action_dim) ou (T,); reward: (T,); done: (T,)."""
    data = np.load(path, allow_pickle=True)
    obs = data["obs"].astype(np.float32)
    T = obs.shape[0]
    if obs.shape[1] != obs_dim:
        # ajusta: se vier só pose (256), zera a percepção; se vier maior, corta.
        fixed = np.zeros((T, obs_dim), np.float32)
        c = min(obs.shape[1], obs_dim)
        fixed[:, :c] = obs[:, :c]
        obs = fixed
    act = data["action"]
    if act.ndim == 1:  # índices → one-hot
        oh = np.zeros((T, action_dim), np.float32)
        for i, a in enumerate(act):
            oh[i, int(a) % action_dim] = 1.0
        act = oh
    rew = data["reward"].astype(np.float32) if "reward" in data else np.zeros(T, np.float32)
    done = data["done"].astype(bool) if "done" in data else np.zeros(T, bool)
    done[-1] = True
    buf.add_sequence(obs, act.astype(np.float32), rew, done, "Offline|replay")
    return T


def save_replay(path, obs, action, reward, done):
    """Utilitário para gravar sequências no formato que --replay lê."""
    np.savez_compressed(path, obs=obs, action=action, reward=reward, done=done)


def export_pt(output, wm, policy, num_bones):
    from export_torchscript import CognitiveInferenceModule
    module = CognitiveInferenceModule(rssm=wm.rssm, actor=policy.actor,
                                      pose_decoder=wm.pose_decoder, num_bones=num_bones)
    module.eval()
    scripted = torch.jit.script(module)
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    scripted.save(output)
    # auto-verifica carregando como o C++ faz
    m = torch.jit.load(output); m.eval()
    H = torch.zeros(1, wm.rssm.hidden_dim)
    Z = torch.zeros(1, wm.rssm.stochastic_dim)
    A = torch.zeros(1, wm.rssm.action_dim if hasattr(wm.rssm, "action_dim") else 9)
    A[0][0] = 1.0
    Obs = torch.zeros(1, wm.rssm.obs_enc_dim)
    with torch.no_grad():
        h, z, idx, pose = m.forward(H, Z, A, Obs, True)
    return pose.numel() == num_bones * 7


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", action="store_true",
                     help="usa o dataset sintético de interações")
    src.add_argument("--replay", type=str, help="arquivo .npz com sequências gravadas")
    ap.add_argument("--steps", type=int, default=2000, help="passos de treino do WM")
    ap.add_argument("--ac-every", type=int, default=4,
                    help="treina actor-critic a cada N passos de WM")
    ap.add_argument("--output", default="CognitiveModel.pt")
    ap.add_argument("--log-every", type=int, default=100)
    args = ap.parse_args()

    cfg, wm, policy, buf, trainer, obs_dim, num_bones = build_stack()
    print(f"Stack: obs_dim={obs_dim} action_dim={cfg.actor_critic.action_dim} "
          f"num_bones={num_bones}")

    if args.dataset:
        n = populate_from_dataset(buf, obs_dim)
        print(f"Dataset sintético → {n} sequências no buffer")
    else:
        n = populate_from_replay(buf, args.replay, obs_dim, cfg.actor_critic.action_dim)
        print(f"Replay '{args.replay}' → {n} frames no buffer")

    if not buf.ready_sequence(cfg.world_model.batch_size):
        print("AVISO: buffer insuficiente para treinar. Forneça mais dados.")
        return 1

    print(f"Treinando {args.steps} passos de World Model...")
    last = {}
    for step in range(1, args.steps + 1):
        m = trainer.train_world_model()
        if m:
            last = m
        if step % args.ac_every == 0:
            trainer.update_actor_critic()
        if step % args.log_every == 0:
            loss = last.get("wm/loss", last.get("loss", float("nan")))
            rec = last.get("wm/rec", 0.0)
            pose = last.get("wm/pose", 0.0)
            print(f"  step {step}/{args.steps}  wm_loss={loss:.4f} "
                  f"rec={rec:.4f} pose={pose:.4f}")

    ok = export_pt(args.output, wm, policy, num_bones)
    print(f"\n.pt exportado em: {args.output}")
    print(f"Verificação forward: {'OK' if ok else 'FALHOU'}")
    print(f"Copie para <Plugin>/Content/Models/CognitiveModel.pt")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
