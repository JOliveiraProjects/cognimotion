"""
expected_learning_value.py
==========================
Estimador de Expected Learning Value (ELV) por habilidade de movimento.

Adaptado do arquivo enviado:
  - Remove dependência de core.logger → usa logging padrão
  - Toda a lógica original (ELVEstimator, TrainingRecord) mantida intacta

No contexto do projeto:
  - skill_name  → nome do ECognitiveMotionStyle (ex: "aggressive", "stealth")
  - context     → contexto do NPC state (ex: "combat", "patrol")
  - Usado pelo ContinuousTrainer para priorizar qual estilo de movimento treinar

Uso:
    estimator = ELVEstimator()
    estimator.record_training(TrainingRecord(
        skill_name="aggressive",
        context="combat",
        reward_before=0.2,
        reward_after=0.45,
        n_steps=500,
    ))
    elv = estimator.estimate("aggressive", "combat", baseline_gap=0.3)
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# TrainingRecord (original — mantido intacto)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingRecord:
    skill_name: str
    context: str
    timestamp: float = field(default_factory=time.time)
    reward_before: float = 0.0
    reward_after: float = 0.0
    n_steps: int = 0
    difficulty: float = 0.5

    @property
    def improvement(self) -> float:
        return self.reward_after - self.reward_before

    @property
    def improvement_per_step(self) -> float:
        if self.n_steps <= 0:
            return 0.0
        return self.improvement / self.n_steps


# ──────────────────────────────────────────────────────────────────────────────
# ELVEstimator (original — mantido intacto, apenas logger adaptado)
# ──────────────────────────────────────────────────────────────────────────────

class ELVEstimator:
    """
    Estima o Expected Learning Value de treinar (skill_name, context).

    ELV ∈ [0, 1] — quanto se espera que o modelo melhore ao treinar
    esse par. Usado para priorizar o curriculum de treinamento.

    Componentes do ELV:
      0.40 × historical_potential  — média ponderada de melhorias passadas
      0.30 × degradation_pressure  — queda relativa ao baseline
      0.15 × rarity_bonus          — raridade do par no buffer
      0.15 × fresh_bonus           — amostras recentes disponíveis
      -     recency_penalty        — penalidade por treino muito recente
    """

    def __init__(
        self,
        history_window: int = 20,
        recency_penalty_s: float = 300.0,
        plateau_threshold: float = 0.01,
        min_history: int = 3,
    ) -> None:
        self.history_window = history_window
        self.recency_penalty_s = recency_penalty_s
        self.plateau_threshold = plateau_threshold
        self.min_history = min_history

        self._history: Dict[Tuple[str, str], deque] = defaultdict(
            lambda: deque(maxlen=history_window)
        )

        logger.info(
            f"ELVEstimator | history_window={history_window} "
            f"| recency_penalty={recency_penalty_s}s"
        )

    # ──────────────────────────────────────────────────────────────────────────

    def record_training(self, record: TrainingRecord) -> None:
        """Registra um episódio de treino para (skill, context)."""
        key = (record.skill_name, record.context)
        self._history[key].append(record)
        logger.debug(
            f"ELVEstimator | ({record.skill_name}, {record.context}) | "
            f"improvement={record.improvement:+.4f} | steps={record.n_steps}"
        )

    def estimate(
        self,
        skill_name: str,
        context: str,
        baseline_gap: float = 0.0,
        rarity_score: float = 0.0,
        fresh_samples: int = 0,
    ) -> float:
        """
        Estima o ELV de treinar (skill_name, context).

        Args:
            baseline_gap:   queda relativa ao melhor reward observado [0, 1]
            rarity_score:   raridade desse par no replay buffer [0, 1]
            fresh_samples:  número de amostras recentes disponíveis

        Returns:
            ELV ∈ [0, 1]
        """
        key = (skill_name, context)
        records = list(self._history[key])

        historical_potential  = self._compute_historical_potential(records)
        degradation_pressure  = float(np.clip(baseline_gap, 0.0, 1.0))
        rarity_bonus          = float(np.clip(rarity_score, 0.0, 1.0)) * 0.3
        fresh_bonus           = float(np.clip(fresh_samples / 1000.0, 0.0, 0.5))
        recency_penalty       = self._compute_recency_penalty(records)

        elv = (
            0.40 * historical_potential
            + 0.30 * degradation_pressure
            + 0.15 * rarity_bonus
            + 0.15 * fresh_bonus
            - recency_penalty
        )
        elv = float(np.clip(elv, 0.0, 1.0))

        logger.debug(
            f"ELV({skill_name}, {context}) = {elv:.4f} | "
            f"hist={historical_potential:.3f} | degr={degradation_pressure:.3f} | "
            f"rare={rarity_bonus:.3f} | fresh={fresh_bonus:.3f} | "
            f"recent_pen={recency_penalty:.3f}"
        )
        return elv

    # ──────────────────────────────────────────────────────────────────────────
    # Internos (original — mantidos intactos)
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_historical_potential(self, records: List[TrainingRecord]) -> float:
        if len(records) < self.min_history:
            return 0.5

        improvements = [r.improvement for r in records]

        # Plateau — sem aprendizado detectável
        if all(abs(imp) < self.plateau_threshold for imp in improvements):
            return 0.05

        n = len(improvements)
        x = np.arange(n, dtype=np.float64)
        y = np.array(improvements, dtype=np.float64)

        x_mean, y_mean = x.mean(), y.mean()
        slope = float(
            np.sum((x - x_mean) * (y - y_mean))
            / (np.sum((x - x_mean) ** 2) + 1e-10)
        )

        # Média ponderada com peso exponencial nos mais recentes
        weights = np.exp(np.linspace(0, 1, n))
        weights /= weights.sum()
        weighted_mean = float(np.dot(weights, y))

        potential = weighted_mean + slope * 2.0
        return float(np.clip((potential + 0.5), 0.0, 1.0))

    def _compute_recency_penalty(self, records: List[TrainingRecord]) -> float:
        if not records:
            return 0.0
        last_record = records[-1]
        age_s = time.time() - last_record.timestamp
        if age_s >= self.recency_penalty_s:
            return 0.0
        return 0.3 * (1.0 - age_s / self.recency_penalty_s)

    # ──────────────────────────────────────────────────────────────────────────
    # Consultas (original — mantidas intactas)
    # ──────────────────────────────────────────────────────────────────────────

    def get_history(self, skill_name: str, context: str) -> List[TrainingRecord]:
        return list(self._history.get((skill_name, context), []))

    def get_all_pairs(self) -> List[Tuple[str, str]]:
        return list(self._history.keys())

    def has_plateau(self, skill_name: str, context: str) -> bool:
        records = list(self._history.get((skill_name, context), []))
        if len(records) < self.min_history:
            return False
        improvements = [r.improvement for r in records[-self.min_history:]]
        return all(abs(imp) < self.plateau_threshold for imp in improvements)

    def get_diagnostics(self) -> Dict:
        return {
            "tracked_pairs": len(self._history),
            "history_window": self.history_window,
            "plateau_pairs": [
                f"{k[0]}/{k[1]}"
                for k in self._history
                if self.has_plateau(k[0], k[1])
            ],
        }

    def best_skill_to_train(self, context: str = "combat") -> Optional[str]:
        """
        Retorna o skill_name com maior ELV para o context dado.
        Útil para o ContinuousTrainer escolher o próximo estilo de treino.
        """
        # ECognitiveMotionStyle names (mesma ordem dos enums UE5)
        style_names = [
            "neutral", "aggressive", "relaxed", "injured",
            "fatigued", "stealth", "military", "civilian", "criminal",
        ]
        best_name: Optional[str] = None
        best_elv = -1.0
        for name in style_names:
            elv = self.estimate(name, context)
            if elv > best_elv:
                best_elv = elv
                best_name = name
        return best_name
