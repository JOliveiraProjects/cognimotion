"""
population_trainer.py
=====================
Population-Based Training (PBT) para hyperparameter tuning automático.

Adaptado de itens.zip — remove dependência de core.logger.
"""
from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PopulationMember:
    member_id: str
    config: Any
    score: float = 0.0
    generation: int = 0
    steps: int = 0

    score_history: List[float] = field(default_factory=list)

    @property
    def trend(self) -> float:
        if len(self.score_history) < 2:
            return 0.0
        recent = self.score_history[-5:]
        if len(recent) < 2:
            return 0.0
        return (recent[-1] - recent[0]) / max(len(recent) - 1, 1)

    def update_score(self, score: float) -> None:
        self.score = score
        self.score_history.append(score)
        if len(self.score_history) > 50:
            self.score_history.pop(0)

    def to_dict(self) -> dict:
        return {
            "id": self.member_id,
            "score": round(self.score, 3),
            "generation": self.generation,
            "steps": self.steps,
            "trend": round(self.trend, 4),
        }


class PopulationTrainer:
    """
    Population-Based Training — mantém N configurações em paralelo,
    periodicamente substitui as piores pelas melhores com perturbação.

    Hyperparâmetros tunados correspondem aos campos de LearningConfig.
    """

    # (min, max, scale)
    HYPERPARAMS: Dict[str, tuple] = {
        "learning_rate":   (1e-5, 1e-3, "log"),
        "entropy_weight":  (0.001, 0.05, "linear"),
        "gradient_clip":   (0.5, 5.0, "linear"),
        "dropout":         (0.05, 0.3, "linear"),
        "style_loss_weight":      (0.05, 0.5, "linear"),
        "imitation_loss_weight":  (0.1, 0.7, "linear"),
    }

    def __init__(
        self,
        base_config,
        n_members: int = 4,
        eval_interval: int = 10_000,
        exploit_ratio: float = 0.25,
        perturb_factor: float = 0.2,
    ) -> None:
        self.base_config = base_config
        self.n_members = n_members
        self.eval_interval = eval_interval
        self.exploit_ratio = exploit_ratio
        self.perturb_factor = perturb_factor

        self.population: List[PopulationMember] = []
        self._generation = 0

        for i in range(n_members):
            cfg = self._create_diverse_config(base_config, i)
            self.population.append(
                PopulationMember(member_id=f"pbt_{i:02d}", config=cfg)
            )

        logger.info(
            f"PopulationTrainer | n={n_members} | eval_interval={eval_interval} "
            f"| exploit_ratio={exploit_ratio}"
        )

    def _create_diverse_config(self, base_config, idx: int):
        cfg = copy.deepcopy(base_config)
        for hp_name, (lo, hi, scale) in self.HYPERPARAMS.items():
            if not hasattr(cfg, hp_name):
                continue
            if scale == "log":
                val = float(np.exp(np.random.uniform(np.log(lo), np.log(hi))))
            else:
                val = float(np.random.uniform(lo, hi))
            setattr(cfg, hp_name, val)
        return cfg

    def update_scores(self, scores: Dict[str, float]) -> None:
        for member in self.population:
            if member.member_id in scores:
                member.update_score(scores[member.member_id])

    def maybe_evolve(self, total_steps: int) -> List[str]:
        if total_steps % self.eval_interval != 0:
            return []
        return self.evolve()

    def evolve(self) -> List[str]:
        self._generation += 1
        n_replace = max(1, int(self.n_members * self.exploit_ratio))

        sorted_pop = sorted(self.population, key=lambda m: m.score, reverse=True)
        best_members = sorted_pop[: max(1, self.n_members - n_replace)]
        worst_members = sorted_pop[-n_replace:]

        replaced = []
        for worst in worst_members:
            best = random.choice(best_members)
            for hp_name in self.HYPERPARAMS:
                if hasattr(best.config, hp_name):
                    setattr(worst.config, hp_name, getattr(best.config, hp_name))
            self._perturb(worst.config)
            worst.generation += 1
            worst.score_history = []
            replaced.append(worst.member_id)

        logger.info(
            f"PBT gen={self._generation} | substituídos={replaced} "
            f"| best_score={sorted_pop[0].score:.3f} "
            f"| worst_score={sorted_pop[-1].score:.3f}"
        )
        return replaced

    def _perturb(self, config) -> None:
        for hp_name, (lo, hi, scale) in self.HYPERPARAMS.items():
            if not hasattr(config, hp_name):
                continue
            val = getattr(config, hp_name)
            if scale == "log":
                delta = np.random.choice([-1, 1]) * self.perturb_factor
                val = val * (1.0 + delta)
            else:
                delta = np.random.uniform(-self.perturb_factor, self.perturb_factor)
                val = val + (hi - lo) * delta
            setattr(config, hp_name, float(np.clip(val, lo, hi)))

    def get_config(self, member_id: str) -> Optional[Any]:
        for m in self.population:
            if m.member_id == member_id:
                return m.config
        return None

    def best_config(self) -> Any:
        if not self.population:
            return self.base_config
        return max(self.population, key=lambda m: m.score).config

    def leaderboard(self) -> List[dict]:
        return [
            m.to_dict()
            for m in sorted(self.population, key=lambda m: m.score, reverse=True)
        ]

    def population_summary(self) -> dict:
        scores = [m.score for m in self.population]
        return {
            "generation": self._generation,
            "n_members": self.n_members,
            "best_score": round(max(scores), 3) if scores else 0.0,
            "worst_score": round(min(scores), 3) if scores else 0.0,
            "avg_score": round(float(np.mean(scores)), 3) if scores else 0.0,
            "leaderboard": self.leaderboard(),
        }
