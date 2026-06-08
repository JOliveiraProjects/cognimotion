"""
perception_features.py — Codifica a percepção num vetor de tamanho FIXO para
entrar na OBSERVAÇÃO do RSSM. É isto que permite a decisão de combate EMERGIR
do RL (em vez de ser regra): a política passa a "ver" inimigo/ameaça/refém no
seu obs e o actor-critic aprende, via recompensa de tarefa, a reagir.

O vetor tem tamanho fixo PERCEPTION_DIM e codifica a entidade mais relevante
(maior ameaça; empate → mais perto) mais um resumo agregado da cena.

Layout (PERCEPTION_DIM = 20):
  [0]      has_target        (0/1)
  [1]      threat            (0..1)
  [2]      distance_norm     (0..1; 1=colado, 0=longe)  = clip(600/dist,0,1)
  [3..5]   direction xyz     (espaço local do NPC)
  [6..9]   disposition onehot (neutral, friend, enemy, ally)
  [10..15] role onehot       (none, hostage, captor, civilian, wounded, leader)
  [16]     n_enemies_norm    (0..1; nº de inimigos / 5)
  [17]     n_threats_norm    (0..1; nº de ameaças > 0.5 / 5)
  [18]     hostage_present   (0/1)
  [19]     captor_present    (0/1)
"""
from __future__ import annotations

import numpy as np

PERCEPTION_DIM = 20

_DISP = {"neutral": 0, "friend": 1, "enemy": 2, "ally": 3}
_ROLE = {"none": 0, "hostage": 1, "captor": 2, "civilian": 3, "wounded": 4, "leader": 5}


def perception_features(entities: list) -> np.ndarray:
    """Converte a lista de entidades percebidas num vetor fixo (float32)."""
    v = np.zeros(PERCEPTION_DIM, dtype=np.float32)
    if not entities:
        return v

    # Entidade prioritária: maior ameaça; empate → mais perto.
    def prio(e):
        return (float(e.get("threat_weight", 0.0)),
                -float(e.get("distance", 1e9)))
    top = max(entities, key=prio)

    dist = max(float(top.get("distance", 1e9)), 1.0)
    v[0] = 1.0
    v[1] = float(np.clip(top.get("threat_weight", 0.0), 0.0, 1.0))
    v[2] = float(np.clip(600.0 / dist, 0.0, 1.0))
    d = top.get("direction", [0.0, 0.0, 0.0])
    if len(d) >= 3:
        v[3], v[4], v[5] = float(d[0]), float(d[1]), float(d[2])

    disp = _DISP.get(top.get("disposition_name", "neutral"), 0)
    v[6 + disp] = 1.0
    role = _ROLE.get(top.get("role_name", "none"), 0)
    v[10 + role] = 1.0

    n_enemies = sum(1 for e in entities
                    if e.get("disposition_name") == "enemy")
    n_threats = sum(1 for e in entities
                    if float(e.get("threat_weight", 0.0)) > 0.5)
    v[16] = float(np.clip(n_enemies / 5.0, 0.0, 1.0))
    v[17] = float(np.clip(n_threats / 5.0, 0.0, 1.0))
    v[18] = 1.0 if any(e.get("role_name") == "hostage" for e in entities) else 0.0
    v[19] = 1.0 if any(e.get("role_name") == "captor" for e in entities) else 0.0
    return v


def augment_obs(obs_enc: np.ndarray, entities: list) -> np.ndarray:
    """Concatena o vetor de pose (256) com o vetor de percepção (20).
    Resultado: obs aumentado de dimensão 256 + PERCEPTION_DIM."""
    pf = perception_features(entities)
    if obs_enc is None:
        return pf
    return np.concatenate([np.asarray(obs_enc, dtype=np.float32), pf], axis=0)
