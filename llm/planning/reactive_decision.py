"""
planning/reactive_decision.py
==============================
Camada de decisão reativa que roda ANTES da política aprendida.

A política (DreamerV3) aprende movimento por imitação, mas certas situações
exigem reação determinística imediata, independente do que foi aprendido:

  1. ESTADOS FÍSICOS (vida, queda, natação) — prioridade máxima
     - Vida zerada  → MORTE (idle absoluto / animação de morte no UE5)
     - Caindo        → estado FALL
     - Nadando       → estado SWIM
     Estes são lidos do blackboard/pose e SOBRESCREVEM a ação da política.

  2. REAÇÕES SITUACIONAIS (percepção) — prioridade alta
     - Ameaça forte e perto → fugir
     - Inimigo atacável     → atacar
     - Semáforo vermelho    → esperar
     Estas modulam ou sobrescrevem a ação aprendida conforme a situação.

A política só decide livremente quando NENHUMA regra reativa dispara — assim o
NPC reage corretamente a levar tiro/soco/atropelamento e ao terreno, mas ainda
exibe o movimento aprendido no resto do tempo.

Semântica de ação canônica (espelhada no UE5 e no inverse_dynamics):
  0=idle 1=forward 2=backward 3=left 4=right 5=run 6=jump 7=crouch 8=stop
Estados físicos estendidos (não são ações de locomoção; sinalizam ao UE5 qual
animação tocar) são devolvidos no campo `physical_state`:
  "alive", "dead", "falling", "swimming", "landing"
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

try:
    from planning.behavior_catalog import decide_behavior
except Exception:  # import defensivo: catálogo é opcional
    decide_behavior = None

logger = logging.getLogger("reactive_decision")


# ECognitiveMovementMode no UE5 (ordem do enum). Usado para detectar queda/natação.
class MovementMode(IntEnum):
    IDLE     = 0
    WALKING  = 1
    RUNNING  = 2
    FALLING  = 3
    SWIMMING = 4
    FLYING   = 5
    CROUCHED = 6


# Ações canônicas
ACT_IDLE, ACT_FWD, ACT_BACK, ACT_LEFT, ACT_RIGHT, ACT_RUN, ACT_JUMP, ACT_CROUCH, ACT_STOP = range(9)


@dataclass
class ReactiveConfig:
    """Configurável por NPC (vem do blackboard/handshake)."""
    max_health: float = 100.0
    death_threshold: float = 0.0       # vida <= isto → morte
    flee_threat: float = 0.7           # ThreatWeight acima disso e perto → fugir
    flee_distance: float = 600.0
    enabled: bool = True


@dataclass
class ReactiveDecision:
    """Resultado da camada reativa."""
    override: bool                 # True = ignora a política, usa esta ação
    action: int                    # ação canônica (0-8)
    physical_state: str            # "alive"|"dead"|"falling"|"swimming"|"landing"
    reason: str                    # para debug


class ReactiveDecisionLayer:
    """
    Avalia o blackboard + percepção e decide se uma reação determinística
    deve sobrescrever a política. Stateless entre chamadas exceto por detecção
    de transição (pouso).
    """

    def __init__(self, config: Optional[ReactiveConfig] = None):
        self.cfg = config or ReactiveConfig()
        self._was_falling: dict[int, bool] = {}

    # ──────────────────────────────────────────────────────────────────────────
    def decide(
        self,
        npc_id: int,
        blackboard: dict,
        movement_mode: int,
        policy_action: int,
        perception: Optional[list] = None,
        profile_name: Optional[str] = None,
    ) -> ReactiveDecision:
        """
        Decide a reação. `policy_action` é o que a política sugeriu; pode ser
        mantido (override=False) ou substituído (override=True).

        `perception` é a lista de entidades percebidas (de MSG_PERCEPTION). Cada
        entidade traz uma `reaction_name` sugerida (attack/flee/hide/pickup/
        enter) calculada no lado C++ a partir de categoria/disposição/ameaça.
        """
        if not self.cfg.enabled:
            return ReactiveDecision(False, policy_action, "alive", "reactive_disabled")

        # ── PRIORIDADE 1: MORTE ───────────────────────────────────────────────
        health = float(blackboard.get("health", self.cfg.max_health))
        if health <= self.cfg.death_threshold:
            self._was_falling.pop(npc_id, None)
            return ReactiveDecision(
                override=True,
                action=ACT_IDLE,           # sem locomoção
                physical_state="dead",     # UE5 toca animação de morte
                reason=f"health={health:.0f} <= {self.cfg.death_threshold:.0f}",
            )

        # ── PRIORIDADE 2: ESTADO FÍSICO (queda / natação / pouso) ─────────────
        was_falling = self._was_falling.get(npc_id, False)

        if movement_mode == int(MovementMode.SWIMMING):
            return ReactiveDecision(
                override=True, action=policy_action,  # nada, mas estado = swimming
                physical_state="swimming", reason="movement_mode=swimming",
            )

        if movement_mode == int(MovementMode.FALLING):
            self._was_falling[npc_id] = True
            return ReactiveDecision(
                override=True, action=ACT_IDLE,
                physical_state="falling", reason="movement_mode=falling",
            )

        # Detecta pouso: estava caindo e agora não está mais
        if was_falling and movement_mode != int(MovementMode.FALLING):
            self._was_falling[npc_id] = False
            return ReactiveDecision(
                override=True, action=ACT_IDLE,
                physical_state="landing", reason="landed",
            )

        # ── PRIORIDADE 3: COMPORTAMENTO RICO (catálogo: emoção/relação/perfil)─
        # Consulta o catálogo comportamental, que entende perfil (urbano/militar/
        # esportivo/piloto/lutador), emoção (medo/raiva/pânico/feliz), relações
        # (amigo/inimigo/aliado/refém/sequestrador) e cenários complexos
        # (sequestrador+refém → atira só com ângulo limpo). Se ele decidir, sua
        # reação é convertida na ação/estado correspondente.
        fear = float(blackboard.get("fear_level", 0.0))
        aggr = float(blackboard.get("aggression_level", 0.0))

        if decide_behavior is not None and perception:
            intent = decide_behavior(perception, blackboard, profile_name)
            if intent is not None:
                rdec = self._reaction_to_decision(intent.reaction)
                if rdec is not None:
                    rdec.reason = f"[{intent.emotion.value}] {intent.reason}"
                    return rdec

        # ── PRIORIDADE 4: reação simples da percepção (fallback) ──────────────
        combat = self._decide_from_perception(perception, fear, aggr)
        if combat is not None:
            return combat

        # ── PRIORIDADE 5: AMEAÇA via blackboard (compat. legada) ──────────────
        threat = float(blackboard.get("threat_level", 0.0))
        if threat >= self.cfg.flee_threat and fear > aggr:
            return ReactiveDecision(
                override=True, action=ACT_RUN,
                physical_state="flee",
                reason=f"flee threat={threat:.2f} fear={fear:.2f}",
            )

        # ── Nenhuma regra disparou: a política decide livremente ──────────────
        return ReactiveDecision(
            override=False, action=policy_action,
            physical_state="alive", reason="policy",
        )

    # ──────────────────────────────────────────────────────────────────────────
    def _decide_from_perception(
        self, perception: Optional[list], fear: float, aggr: float
    ) -> Optional[ReactiveDecision]:
        """
        Converte a entidade percebida mais relevante numa decisão de combate.
        Retorna None se não há nada que justifique sobrepor a política.

        O COMBATE é expresso via physical_state (que o Unreal interpreta para
        tocar a animação certa — mesmo mecanismo de dead/falling); a `action`
        usa o espaço de locomoção existente (action_dim=9):
          "attack" → aproxima (ACT_FWD)    state="attack"  (se medo>agressão → foge)
          "flee"   → corre (ACT_RUN)         state="flee"
          "hide"   → agacha (ACT_CROUCH)     state="hide"
          "pickup"/"enter" → aproxima        state=reação
        """
        if not perception:
            return None

        # Entidade de maior prioridade: maior ameaça; empate → mais perto.
        def score(e):
            return (float(e.get("threat_weight", 0.0)),
                    -float(e.get("distance", 1e9)))
        top = max(perception, key=score)

        reaction = top.get("reaction_name", "none")
        dist     = float(top.get("distance", 1e9))

        # Só reage a algo dentro de um raio útil (pickup/enter podem ser longe).
        if dist > self.cfg.flee_distance and reaction not in ("pickup", "enter"):
            return None

        if reaction == "attack":
            # Medo dominante converte ataque em fuga (autopreservação).
            # Basta o medo superar a agressão — independe do limiar de ameaça.
            if fear > aggr:
                return ReactiveDecision(True, ACT_RUN, "flee",
                                        f"perception: fugiu de inimigo (fear={fear:.2f} > aggr={aggr:.2f})")
            return ReactiveDecision(True, ACT_FWD, "attack",
                                    f"perception: ataca alvo dist={dist:.0f}")
        if reaction == "flee":
            return ReactiveDecision(True, ACT_RUN, "flee",
                                    f"perception: foge de hazard dist={dist:.0f}")
        if reaction == "hide":
            return ReactiveDecision(True, ACT_CROUCH, "hide",
                                    f"perception: esconde dist={dist:.0f}")
        if reaction in ("pickup", "enter"):
            return ReactiveDecision(True, ACT_FWD, reaction,
                                    f"perception: {reaction} dist={dist:.0f}")
        return None

    # ──────────────────────────────────────────────────────────────────────────
    def _reaction_to_decision(self, reaction: str) -> Optional[ReactiveDecision]:
        """Converte uma reação canônica do catálogo numa ReactiveDecision,
        usando o espaço de locomoção (action_dim=9) + physical_state."""
        mapping = {
            "attack":   (ACT_FWD,    "attack"),
            "flee":     (ACT_RUN,    "flee"),
            "hide":     (ACT_CROUCH, "hide"),
            "approach": (ACT_FWD,    "alive"),
            "pickup":   (ACT_FWD,    "pickup"),
            "enter":    (ACT_FWD,    "enter"),
            "wait":     (ACT_IDLE,   "alive"),
        }
        if reaction not in mapping:
            return None
        action, state = mapping[reaction]
        return ReactiveDecision(True, action, state, reaction)
