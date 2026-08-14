"""
teaching_supervisor.py — Treino & Ensino integrado ao pipeline cognitivo.

Recebe da tela "Cognitive Training Studio" (via protocol/binary_protocol):
  TrainingRegister  registra o vocabulário de um treino {tipo, movimento}
  TeachingScenario  cenário → decide o movimento (responde TeachingChoice)
  TeachingFeedback  correção → vira dado supervisionado, retreina a política

REUSA a infraestrutura existente em vez de duplicá-la:
  - encoding.perception_features.perception_features → vetor 20-d (mesmo do
    resto do sistema; NÃO inventa features próprias)
  - protocol.binary_protocol.VERB_NAMES → vocabulário de movimentos do world model
    (approach/attack/flee/hide/pickup/enter/wait/cross), o MESMO que a
    percepção e o BehaviorCatalog usam
  - o mesmo estilo de cabeça supervisionada do DemonstrationLearner
    (percepção → movimento, cross-entropy, treino incremental)

A política aqui decide a REAÇÃO tática do ensino. O world model (RSSM) de
locomoção continua sendo treinado pelo pipeline DreamerV3 — não é tocado.

O cenário de ensino é traduzido para o MESMO formato de entidades que a
percepção real produz, então perception_features() gera um vetor consistente
com o que o NPC vê em jogo.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding.perception_features import perception_features, PERCEPTION_DIM
from protocol.binary_protocol import VERB_NAMES

VERB_NAME_TO_IDX = {v: k for k, v in VERB_NAMES.items()}

logger = logging.getLogger("teaching_supervisor")

# Vocabulário canônico de MOVIMENTOS que o world model executa (VERB_NAMES):
# idle/walk/run/jump/crouch/crawl/vault/pickup/flee/hide/attack/defend.
# O ensino escolhe qual destes o NPC usa em cada situação; o world model
# gera o movimento contínuo daquele verbo. Não há clipe: é a ação que dirige
# a geração de pose do DreamerV3.
VERBS = [VERB_NAMES[i] for i in sorted(VERB_NAMES)]
N_VERBS = len(VERB_NAMES)


class _VerbHead(nn.Module):
    """percepção (20-d) → verbo de movimento (N_VERBS)."""

    def __init__(self, in_dim: int = PERCEPTION_DIM, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, N_VERBS),
        )

    def forward(self, x):
        return self.net(x)


class TeachingSupervisor:
    """
    Uma instância por serviço (compartilhada entre sessões, como demo_learner).
    Persistência: <checkpoint_dir>/teaching/…
    """

    def __init__(self, checkpoint_dir: str = "checkpoints",
                 device: str = "cpu", lr: float = 1e-3, capacity: int = 20000):
        self.dir = os.path.join(checkpoint_dir, "teaching")
        os.makedirs(self.dir, exist_ok=True)
        self.device = device
        self._lock = threading.Lock()

        # Catálogo de treinos: tipo → [ {reaction=verbo, notes} ]
        self.catalog: dict[str, list[dict]] = {}

        # Buffer supervisionado: (percepção 20-d, reaction_idx, peso)
        self._perc: list[np.ndarray] = []
        self._react: list[int] = []
        self._w: list[float] = []
        self._cap = capacity

        # Uma cabeça por tipo de treino (percepção → movimento/verbo)
        self.heads: dict[str, _VerbHead] = {}
        self.opts: dict[str, torch.optim.Optimizer] = {}
        self._buf_by_type: dict[str, list[int]] = {}   # tipo → índices no buffer

        # Cenários pendentes de feedback: scenario_id → {training_type, pvec}
        self.pending: dict[int, dict] = {}

        self.lr = lr
        self._feedback_since_train = 0
        self.retrain_every = 5

        self._load()

    # ── persistência ─────────────────────────────────────────────────────────
    @property
    def _catalog_path(self) -> str:
        return os.path.join(self.dir, "catalog.jsonl")

    @property
    def _dataset_path(self) -> str:
        return os.path.join(self.dir, "teaching_dataset.jsonl")

    def _head_path(self, ttype: str) -> str:
        safe = "".join(c if c.isalnum() else "_" for c in ttype)
        return os.path.join(self.dir, f"reaction_head_{safe}.pt")

    def _load(self) -> None:
        if os.path.exists(self._catalog_path):
            with open(self._catalog_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        t = json.loads(line)
                        self.catalog.setdefault(t["training_type"], []).append(t)
                    except json.JSONDecodeError:
                        pass
        # reconstrói dataset a partir do log e recarrega/retreina cabeças
        if os.path.exists(self._dataset_path):
            with open(self._dataset_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self._add_sample(r["training_type"],
                                     np.asarray(r["pvec"], dtype=np.float32),
                                     int(r["reaction_idx"]), float(r["w"]),
                                     persist=False)
        for ttype in list(self.catalog) + list(self._buf_by_type):
            path = self._head_path(ttype)
            if os.path.exists(path):
                head = _VerbHead().to(self.device)
                head.load_state_dict(torch.load(path, map_location=self.device))
                head.eval()
                self.heads[ttype] = head
                self.opts[ttype] = torch.optim.Adam(head.parameters(), lr=self.lr)
        n = sum(len(v) for v in self.catalog.values())
        logger.info("TeachingSupervisor: %d treino(s) em %d tipo(s); "
                    "%d amostra(s) de ensino; %d cabeça(s)",
                    n, len(self.catalog), len(self._perc), len(self.heads))

    def _append_jsonl(self, path: str, obj: dict) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # ── TrainingRegister ──────────────────────────────────────────────────────
    def register_training(self, parsed: dict) -> None:
        ttype = parsed.get("training_type", "").strip()
        react = parsed.get("reaction", "").strip()
        if not ttype or not react:
            return
        entry = {"training_type": ttype, "reaction": react,
                 "animation": parsed.get("animation", ""),
                 "notes": parsed.get("notes", "")}
        with self._lock:
            self.catalog.setdefault(ttype, []).append(entry)
            self._append_jsonl(self._catalog_path, entry)
        logger.info('[Training] + [%s] "%s" anim=%s',
                    ttype, react, entry["animation"] or "-")

    # ── conversão cenário → entidades (formato da percepção real) ─────────────
    @staticmethod
    def _scenario_to_entities(parsed: dict) -> list:
        """Traduz as contagens do cenário para o MESMO formato de entidade que
        parse_perception produz, para reusar perception_features()."""
        entities = []
        for e in parsed.get("entities", []):
            kind = e.get("kind", "")
            count = int(e.get("count", 0))
            facing = int(e.get("facing_me", 0))
            dist_cm = float(e.get("distance_m", 5.0)) * 100.0
            if kind == "enemy":
                disp, cat, threat = "enemy", "character", 0.8
            elif kind == "ally":
                disp, cat, threat = "ally", "character", 0.0
            elif kind == "danger":
                disp, cat, threat = "neutral", "hazard", 0.9
            else:
                disp, cat, threat = "neutral", "objective", 0.2
            for i in range(max(count, 0)):
                entities.append({
                    "category_name": cat, "disposition_name": disp,
                    "role_name": "none", "reaction_name": "none",
                    "distance": dist_cm,
                    # direção: quem encara aponta para o agente (−X local)
                    "direction": [-1.0 if i < facing else 1.0, 0.0, 0.0],
                    "threat_weight": threat,
                })
        return entities

    # ── TeachingScenario → escolha ────────────────────────────────────────────
    def decide(self, parsed: dict) -> tuple:
        """Retorna (chosen, confidence, rationale). Guarda pendência p/ feedback."""
        ttype = parsed.get("training_type", "").strip()
        scenario_id = int(parsed.get("scenario_id", 0))

        candidates = [c for c in parsed.get("candidates", []) if c]
        if not candidates:
            candidates = [t["reaction"] for t in self.catalog.get(ttype, [])]
        candidates = list(dict.fromkeys(candidates))  # únicos, ordem estável

        entities = self._scenario_to_entities(parsed)
        pvec = perception_features(entities)

        with self._lock:
            self.pending[scenario_id] = {"training_type": ttype, "pvec": pvec}

        if not candidates:
            return ("", 0.0,
                    f"Nenhum movimento registrado para o tipo '{ttype}'. "
                    f"Registre treinos primeiro.")

        head = self.heads.get(ttype)
        if head is None:
            # sem política treinada ainda: primeira candidata, transparente
            return (candidates[0], 1.0 / len(candidates),
                    "sem política treinada para este tipo — palpite; corrija-me")

        # máscara: só candidatas que estão no vocabulário canônico
        known = [c for c in candidates if c in VERB_NAME_TO_IDX]
        if not known:
            return (candidates[0], 1.0 / len(candidates),
                    "candidatas fora do vocabulário canônico — palpite")

        with torch.no_grad():
            x = torch.tensor(pvec, device=self.device).unsqueeze(0)
            probs = F.softmax(head(x)[0], dim=0)
        scored = [(c, float(probs[VERB_NAME_TO_IDX[c]].item())) for c in known]
        scored.sort(key=lambda t: t[1], reverse=True)
        best, best_p = scored[0]
        total = sum(p for _, p in scored) or 1.0
        return (best, best_p / total,
                "política neural treinada com suas correções")

    # ── buffer + treino ────────────────────────────────────────────────────────
    def _add_sample(self, ttype: str, pvec: np.ndarray, reaction_idx: int,
                    w: float, persist: bool = True) -> None:
        with self._lock:
            i = len(self._perc)
            self._perc.append(np.asarray(pvec, dtype=np.float32))
            self._react.append(int(reaction_idx))
            self._w.append(float(w))
            self._buf_by_type.setdefault(ttype, []).append(i)
            if len(self._perc) > self._cap:
                # descarte simples do mais antigo (mantém índices consistentes
                # reconstruindo o mapa por tipo)
                self._perc.pop(0); self._react.pop(0); self._w.pop(0)
                for k in self._buf_by_type:
                    self._buf_by_type[k] = [j - 1 for j in self._buf_by_type[k]
                                            if j - 1 >= 0]
        if persist:
            self._append_jsonl(self._dataset_path, {
                "training_type": ttype, "pvec": [float(x) for x in pvec],
                "reaction_idx": int(reaction_idx), "w": float(w)})

    def _train_type(self, ttype: str, epochs: int = 200) -> Optional[float]:
        idxs = self._buf_by_type.get(ttype, [])
        if len(idxs) < 4:
            return None
        head = self.heads.get(ttype)
        if head is None:
            head = _VerbHead().to(self.device)
            self.heads[ttype] = head
            self.opts[ttype] = torch.optim.Adam(head.parameters(), lr=self.lr)
        opt = self.opts[ttype]

        X = torch.tensor(np.stack([self._perc[i] for i in idxs]),
                         device=self.device)
        Y = torch.tensor([self._react[i] for i in idxs],
                         dtype=torch.long, device=self.device)
        W = torch.tensor([self._w[i] for i in idxs],
                         dtype=torch.float32, device=self.device)

        head.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss = (F.cross_entropy(head(X), Y, reduction="none") * W).mean()
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            acc = (head(X).argmax(1) == Y).float().mean().item()
        torch.save(head.state_dict(), self._head_path(ttype))
        logger.info("[Teaching] '%s': %d amostra(s), acc(dataset)=%.0f%%",
                    ttype, len(idxs), acc * 100)
        return acc

    # ── TeachingFeedback ──────────────────────────────────────────────────────
    def feedback(self, parsed: dict) -> None:
        scenario_id = int(parsed.get("scenario_id", 0))
        correct = bool(parsed.get("correct", False))
        chosen = parsed.get("chosen", "")
        suggested = parsed.get("suggested", [])

        base = self.pending.pop(scenario_id, None)
        if base is None:
            logger.warning("[Teaching] feedback de cenário desconhecido %d",
                           scenario_id)
            return
        ttype, pvec = base["training_type"], base["pvec"]

        if correct:
            idx = VERB_NAME_TO_IDX.get(chosen)
            if idx is not None:
                self._add_sample(ttype, pvec, idx, 1.0)
        else:
            for rank, sug in enumerate(suggested):
                idx = VERB_NAME_TO_IDX.get(sug)
                if idx is not None:
                    self._add_sample(ttype, pvec, idx, max(1.0 - 0.2 * rank, 0.2))

        self._append_jsonl(os.path.join(self.dir, "feedback_log.jsonl"), {
            "scenario_id": scenario_id, "training_type": ttype,
            "correct": correct, "chosen": chosen, "suggested": suggested,
            "comment": parsed.get("comment", "")})

        mark = "✔ correto" if correct else f"✘ errado; sugestões: {suggested}"
        logger.info("[Teaching] feedback cenário %d: %s", scenario_id, mark)

        self._feedback_since_train += 1
        if self._feedback_since_train >= self.retrain_every:
            self._feedback_since_train = 0
            self._train_type(ttype)

    def train_now(self, ttype: Optional[str] = None) -> None:
        """Força retreino (usado em testes / comando manual)."""
        types = [ttype] if ttype else list(self._buf_by_type)
        for t in types:
            self._train_type(t)
