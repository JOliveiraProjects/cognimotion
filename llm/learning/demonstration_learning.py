"""
demonstration_learning.py — Aprendizado por DEMONSTRAÇÃO ROTULADA.

O líder, durante a cena, diz por Blueprint:
  SetCurrentEmotion(Fear) / SetCurrentRelation(Enemy) / SetCurrentAction(Flee)

A cada frame demonstrado, o sistema recebe a tripla rotulada:
  (vetor_de_percepção, emoção_rotulada, ação_rotulada)

Duas cabeças neurais aprendem por supervisão (cross-entropy):
  EmotionHead:  percepção (20-d) ............ → emoção (N_EMOTIONS)
  ActionHead:   percepção (20-d) + emoção .... → ação (N_ACTIONS)

Em INFERÊNCIA, quando NÃO há rótulo do líder, as cabeças preveem
emoção e ação — é assim que o NPC GENERALIZA para situações novas com base
no que o líder ensinou. Isto NÃO substitui o world model (que aprende a
imitar o MOVIMENTO); é a camada cognitiva percepção→emoção→ação, aprendida
em vez de regrada.

Conjunto canônico (alinhado entre C++, wire e Python):
  EMOTIONS: calm, happy, alert, fear, anger, panic, confident, suspicious
  RELATIONS já vêm na percepção (disposition/role).
  ACTIONS: o mesmo action_dim de locomoção (0..8) + estados de combate.
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("demonstration_learning")

# ── Conjunto canônico de emoções (índice = valor no wire/C++) ────────────────
EMOTIONS = ["calm", "happy", "alert", "fear", "anger", "panic",
            "confident", "suspicious"]
N_EMOTIONS = len(EMOTIONS)
EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTIONS)}
IDX_TO_EMOTION = {i: e for i, e in enumerate(EMOTIONS)}

# ── Ações canônicas: locomoção (0..8) idêntica ao action_dim do executor ─────
# 0 idle, 1 fwd, 2 back, 3 left, 4 right, 5 run, 6 jump, 7 crouch, 8 stop
N_ACTIONS = 9

from encoding.perception_features import PERCEPTION_DIM


class EmotionHead(nn.Module):
    """percepção → distribuição sobre emoções."""
    def __init__(self, in_dim: int = PERCEPTION_DIM, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, N_EMOTIONS),
        )

    def forward(self, x):
        return self.net(x)


class ActionHead(nn.Module):
    """percepção + emoção(one-hot) → distribuição sobre ações."""
    def __init__(self, in_dim: int = PERCEPTION_DIM + N_EMOTIONS, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, N_ACTIONS),
        )

    def forward(self, x):
        return self.net(x)


class DemonstrationLearner:
    """
    Acumula demonstrações rotuladas do líder e treina as duas cabeças.
    Thread-safe: o servidor adiciona demonstrações enquanto um thread treina.
    """
    def __init__(self, device: str = "cpu", lr: float = 1e-3,
                 buffer_capacity: int = 50000):
        self.device = device
        self.emotion_head = EmotionHead().to(device)
        self.action_head = ActionHead().to(device)
        self.opt = torch.optim.Adam(
            list(self.emotion_head.parameters())
            + list(self.action_head.parameters()), lr=lr)

        self._lock = threading.Lock()
        self._cap = buffer_capacity
        self._perc: list = []      # vetores de percepção (20-d)
        self._emo: list = []       # índice de emoção rotulada
        self._act: list = []       # índice de ação rotulada
        self.train_steps = 0

    # ── Coleta ────────────────────────────────────────────────────────────────
    def add_demonstration(self, perception_vec: np.ndarray,
                          emotion: str, action_idx: int) -> None:
        """Registra uma tripla rotulada pelo líder neste frame."""
        emo_idx = EMOTION_TO_IDX.get(emotion, 0)
        act_idx = int(action_idx) if 0 <= int(action_idx) < N_ACTIONS else 0
        with self._lock:
            self._perc.append(np.asarray(perception_vec, dtype=np.float32))
            self._emo.append(emo_idx)
            self._act.append(act_idx)
            if len(self._perc) > self._cap:
                self._perc.pop(0); self._emo.pop(0); self._act.pop(0)

    def n_demonstrations(self) -> int:
        with self._lock:
            return len(self._perc)

    # ── Treino supervisionado ───────────────────────────────────────────────
    def train_step(self, batch_size: int = 64) -> Optional[dict]:
        with self._lock:
            n = len(self._perc)
            if n < batch_size:
                return None
            idx = np.random.choice(n, batch_size, replace=False)
            P = torch.tensor(np.stack([self._perc[i] for i in idx]),
                             device=self.device)
            E = torch.tensor([self._emo[i] for i in idx],
                             dtype=torch.long, device=self.device)
            A = torch.tensor([self._act[i] for i in idx],
                             dtype=torch.long, device=self.device)

        # Cabeça de emoção: percepção → emoção
        emo_logits = self.emotion_head(P)
        loss_emo = F.cross_entropy(emo_logits, E)

        # Cabeça de ação: percepção + emoção REAL (one-hot) → ação
        emo_onehot = F.one_hot(E, N_EMOTIONS).float()
        act_in = torch.cat([P, emo_onehot], dim=1)
        act_logits = self.action_head(act_in)
        loss_act = F.cross_entropy(act_logits, A)

        loss = loss_emo + loss_act
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.train_steps += 1

        # Acurácia (quanto já aprendeu a reproduzir o líder)
        with torch.no_grad():
            acc_emo = (emo_logits.argmax(1) == E).float().mean().item()
            acc_act = (act_logits.argmax(1) == A).float().mean().item()
        return {"loss": float(loss.item()),
                "loss_emotion": float(loss_emo.item()),
                "loss_action": float(loss_act.item()),
                "acc_emotion": acc_emo, "acc_action": acc_act,
                "n": n}

    # ── Inferência (generalização) ──────────────────────────────────────────
    @torch.no_grad()
    def infer(self, perception_vec: np.ndarray) -> dict:
        """Sem rótulo do líder: prevê emoção e ação a partir da percepção.
        É a generalização do que foi ensinado."""
        P = torch.tensor(np.asarray(perception_vec, dtype=np.float32),
                         device=self.device).unsqueeze(0)
        emo_logits = self.emotion_head(P)
        emo_idx = int(emo_logits.argmax(1).item())
        emo_onehot = F.one_hot(torch.tensor([emo_idx], device=self.device),
                               N_EMOTIONS).float()
        act_logits = self.action_head(torch.cat([P, emo_onehot], dim=1))
        act_idx = int(act_logits.argmax(1).item())
        emo_conf = float(F.softmax(emo_logits, dim=1).max().item())
        act_conf = float(F.softmax(act_logits, dim=1).max().item())
        return {"emotion": IDX_TO_EMOTION[emo_idx], "emotion_idx": emo_idx,
                "action_idx": act_idx,
                "emotion_conf": emo_conf, "action_conf": act_conf}

    # ── Persistência ──────────────────────────────────────────────────────────
    def state_dict(self) -> dict:
        return {"emotion_head": self.emotion_head.state_dict(),
                "action_head": self.action_head.state_dict(),
                "train_steps": self.train_steps}

    def load_state_dict(self, sd: dict) -> None:
        self.emotion_head.load_state_dict(sd["emotion_head"])
        self.action_head.load_state_dict(sd["action_head"])
        self.train_steps = sd.get("train_steps", 0)
