"""
semantic_memory.py
==================
Memória semântica compartilhada entre NPCs — fatos sobre o mundo com decay.

Adaptado de itens.zip:
  - Remove dependência de core.logger
  - Remove dependência de VectorStore para busca semântica (usa dict simples)
  - Mantém lógica de reinforcement/decay de fatos
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Tipos
# ──────────────────────────────────────────────────────────────────────────────

class Relations:
    """Constantes para relações semânticas entre entidades."""
    IS_DANGEROUS       = "is_dangerous"
    HAS_RESOURCE       = "has_resource"
    IS_ALLY_OF         = "is_ally_of"
    IS_ENEMY_OF        = "is_enemy_of"
    LEADS_TO           = "leads_to"
    BLOCKS             = "blocks"
    IS_EFFECTIVE_AGAINST = "is_effective_against"
    MOTION_STYLE_FOR   = "motion_style_for"


@dataclass
class KnowledgeFact:
    subject: str
    relation: str
    object_val: Any
    confidence: float = 0.5
    count: int = 1
    last_seen: float = field(default_factory=time.time)
    source: str = "experience"

    @property
    def key(self) -> str:
        return f"{self.subject}|{self.relation}|{self.object_val}"

    def reinforce(self, delta: float = 0.05) -> None:
        self.count += 1
        self.confidence = min(1.0, self.confidence + delta * (1.0 - self.confidence))
        self.last_seen = time.time()

    def decay(self, rate: float = 0.001) -> None:
        self.confidence = max(0.0, self.confidence - rate)

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "relation": self.relation,
            "object": str(self.object_val),
            "confidence": round(self.confidence, 3),
            "count": self.count,
        }


# ──────────────────────────────────────────────────────────────────────────────
# SemanticMemory
# ──────────────────────────────────────────────────────────────────────────────

class SemanticMemory:
    """
    Memória semântica compartilhada entre todos os NPCs da sessão.

    Armazena fatos do tipo (sujeito, relação, objeto) com confidência e decay.
    Pode ser consultada por sujeito, relação ou objeto.
    """

    def __init__(
        self,
        max_facts: int = 10_000,
        decay_rate: float = 0.001,
    ) -> None:
        self.max_facts = max_facts
        self.decay_rate = decay_rate

        self._facts: Dict[str, KnowledgeFact] = {}
        # Índices para busca rápida
        self._by_subject: Dict[str, List[str]] = defaultdict(list)
        self._by_relation: Dict[str, List[str]] = defaultdict(list)

        self._total_reinforcements = 0
        self._last_decay = time.time()

        logger.info(f"SemanticMemory | max_facts={max_facts} | decay_rate={decay_rate}")

    # ──────────────────────────────────────────────────────────────────────────
    # Inserção / reforço
    # ──────────────────────────────────────────────────────────────────────────

    def learn(
        self,
        subject: str,
        relation: str,
        object_val: Any,
        confidence_delta: float = 0.05,
        source: str = "experience",
    ) -> KnowledgeFact:
        """
        Aprende ou reforça um fato.
        Se o fato já existe, aumenta sua confidência.
        Se é novo, cria com confidência = confidence_delta.
        """
        fact = KnowledgeFact(
            subject=subject,
            relation=relation,
            object_val=object_val,
            confidence=confidence_delta,
            source=source,
        )
        key = fact.key

        if key in self._facts:
            self._facts[key].reinforce(confidence_delta)
        else:
            if len(self._facts) >= self.max_facts:
                self._prune()
            self._facts[key] = fact
            self._by_subject[subject].append(key)
            self._by_relation[relation].append(key)

        self._total_reinforcements += 1
        return self._facts[key]

    # ──────────────────────────────────────────────────────────────────────────
    # Consultas
    # ──────────────────────────────────────────────────────────────────────────

    def query_subject(
        self,
        subject: str,
        relation: Optional[str] = None,
        min_confidence: float = 0.1,
    ) -> List[KnowledgeFact]:
        keys = self._by_subject.get(subject, [])
        facts = [self._facts[k] for k in keys if k in self._facts]
        if relation:
            facts = [f for f in facts if f.relation == relation]
        return [f for f in facts if f.confidence >= min_confidence]

    def query_relation(
        self,
        relation: str,
        min_confidence: float = 0.1,
    ) -> List[KnowledgeFact]:
        keys = self._by_relation.get(relation, [])
        facts = [self._facts[k] for k in keys if k in self._facts]
        return [f for f in facts if f.confidence >= min_confidence]

    def get_fact(
        self,
        subject: str,
        relation: str,
        object_val: Any,
    ) -> Optional[KnowledgeFact]:
        key = f"{subject}|{relation}|{object_val}"
        return self._facts.get(key)

    def get_motion_style_for(self, npc_state: int) -> Optional[int]:
        """Recupera o motion_style sugerido pela memória semântica para um estado NPC."""
        facts = self.query_subject(
            subject=f"state_{npc_state}",
            relation=Relations.MOTION_STYLE_FOR,
            min_confidence=0.3,
        )
        if not facts:
            return None
        best = max(facts, key=lambda f: f.confidence)
        try:
            return int(best.object_val)
        except (ValueError, TypeError):
            return None

    # ──────────────────────────────────────────────────────────────────────────
    # Manutenção
    # ──────────────────────────────────────────────────────────────────────────

    def apply_decay(self) -> None:
        """Aplica decay em todos os fatos. Chamar periodicamente."""
        now = time.time()
        elapsed = now - self._last_decay
        self._last_decay = now
        rate = self.decay_rate * elapsed

        to_remove = []
        for key, fact in self._facts.items():
            fact.decay(rate)
            if fact.confidence <= 0.0:
                to_remove.append(key)

        for key in to_remove:
            self._remove_fact(key)

    def _prune(self) -> None:
        """Remove os 10% de fatos menos confiantes."""
        n_remove = max(1, len(self._facts) // 10)
        sorted_keys = sorted(self._facts, key=lambda k: self._facts[k].confidence)
        for key in sorted_keys[:n_remove]:
            self._remove_fact(key)

    def _remove_fact(self, key: str) -> None:
        fact = self._facts.pop(key, None)
        if fact:
            subj_list = self._by_subject.get(fact.subject, [])
            if key in subj_list:
                subj_list.remove(key)
            rel_list = self._by_relation.get(fact.relation, [])
            if key in rel_list:
                rel_list.remove(key)

    # ──────────────────────────────────────────────────────────────────────────

    def summary(self) -> Dict:
        if not self._facts:
            return {"fact_count": 0}
        confs = [f.confidence for f in self._facts.values()]
        return {
            "fact_count": len(self._facts),
            "mean_confidence": round(float(np.mean(confs)), 3),
            "total_reinforcements": self._total_reinforcements,
        }
