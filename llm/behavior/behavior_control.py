"""
behavior_control.py
===================
Controladores de comportamento por NPC.

Adaptado do arquivo enviado:
  - Remove dependência de core.logger → usa logging padrão
  - PolicyController adaptado para BehaviorConfig do projeto
  - Mantém 100% da lógica original de GoalController e ExplorationController
  - Sem remoção de nenhum método existente
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# GoalController (original — mantido intacto)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GoalSpec:
    goal_str: str
    priority: float = 1.0
    max_steps: Optional[int] = None
    active: bool = True


class GoalController:
    """
    Gerencia o objetivo atual de um NPC.
    Um NPC pode ter um objetivo ativo, uma pilha de objetivos e objetivos cancelados.
    Integra com ECognitiveNPCState: o goal_str mapeia a contextos de estado.
    """

    def __init__(self) -> None:
        self._current_goal: Optional[GoalSpec] = None
        self._goal_stack: List[GoalSpec] = []
        self._cancelled: Set[str] = set()
        self._step_since_goal: int = 0

    def set_goal(
        self,
        goal_str: str,
        priority: float = 1.0,
        max_steps: Optional[int] = None,
    ) -> GoalSpec:
        spec = GoalSpec(goal_str=goal_str, priority=priority, max_steps=max_steps)
        self._current_goal = spec
        self._step_since_goal = 0
        logger.info(f"[GOAL] Novo objetivo: '{goal_str}' (priority={priority})")
        return spec

    def update_goal(self, goal_str: str) -> None:
        if self._current_goal:
            old = self._current_goal.goal_str
            self._current_goal.goal_str = goal_str
            logger.info(f"[GOAL] Objetivo atualizado: '{old}' → '{goal_str}'")

    def cancel_goal(self) -> Optional[str]:
        if self._current_goal:
            cancelled = self._current_goal.goal_str
            self._cancelled.add(cancelled)
            self._current_goal = None
            logger.info(f"[GOAL] Objetivo cancelado: '{cancelled}'")
            return cancelled
        return None

    def current_goal_str(self) -> str:
        if self._current_goal and self._current_goal.active:
            return self._current_goal.goal_str
        return "explore_environment"

    def tick(self) -> Optional[str]:
        """
        Incrementa step counter. Se o objetivo atingiu max_steps, expira-o
        e retorna o goal_str expirado; caso contrário retorna None.
        """
        self._step_since_goal += 1
        if (
            self._current_goal
            and self._current_goal.max_steps
            and self._step_since_goal >= self._current_goal.max_steps
        ):
            expired = self._current_goal.goal_str
            self._current_goal = None
            logger.info(
                f"[GOAL] Objetivo expirou após {self._step_since_goal} steps: '{expired}'"
            )
            return expired
        return None

    @property
    def has_goal(self) -> bool:
        return self._current_goal is not None and self._current_goal.active

    @property
    def current_goal(self) -> Optional[GoalSpec]:
        return self._current_goal

    @property
    def steps_in_goal(self) -> int:
        return self._step_since_goal


# ──────────────────────────────────────────────────────────────────────────────
# ExplorationController (original — mantido intacto)
# ──────────────────────────────────────────────────────────────────────────────

class ExplorationController:
    """
    Controla o modo de exploração do NPC durante treinamento.
    Compatível com OnlineImitationLearner do projeto.
    """

    MODES = ("deterministic", "exploratory", "epsilon_greedy", "auto")

    def __init__(self, mode: str = "auto", epsilon: float = 0.1) -> None:
        assert mode in self.MODES, f"Modo inválido: {mode}. Válidos: {self.MODES}"
        self._mode = mode
        self._epsilon = epsilon
        logger.info(f"[EXPLORATION] mode={mode} epsilon={epsilon}")

    def set_mode(self, mode: str, epsilon: Optional[float] = None) -> None:
        assert mode in self.MODES, f"Modo inválido: {mode}"
        self._mode = mode
        if epsilon is not None:
            self._epsilon = float(epsilon)
        logger.info(f"[EXPLORATION] mode alterado para {mode}")

    def is_deterministic(self) -> bool:
        return self._mode == "deterministic"

    def is_exploratory(self) -> bool:
        return self._mode == "exploratory"

    def should_explore(self) -> bool:
        """Retorna True se o NPC deve explorar em vez de explotar."""
        if self._mode == "deterministic":
            return False
        if self._mode == "exploratory":
            return True
        if self._mode == "epsilon_greedy":
            import random
            return random.random() < self._epsilon
        # auto: baseado em taxa de sucesso (padrão: exploratory)
        return True

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def mode(self) -> str:
        return self._mode


# ──────────────────────────────────────────────────────────────────────────────
# PolicyController (adaptado — usa BehaviorConfig do projeto)
# ──────────────────────────────────────────────────────────────────────────────

class PolicyController:
    """
    Controla quais planners e skills estão ativos para um NPC.

    Adaptação: aceita BehaviorConfig do projeto (config.py) em vez do
    config genérico original. Usa hasattr para compatibilidade retroativa.
    """

    def __init__(self, config) -> None:
        """
        config: BehaviorConfig ou qualquer objeto com os atributos:
          use_actor_critic, use_tree_planner, use_skill_discovery, use_llm
        """
        self.config = config
        self._planners: Dict[str, bool] = {
            "actor_critic":   getattr(config, "use_actor_critic", True),
            "tree_planner":   getattr(config, "use_tree_planner", False),
            "mpc":            True,
            "skill_discovery": getattr(config, "use_skill_discovery", False),
            "hierarchical":   getattr(config, "use_llm", True),
        }
        self._disabled_skills: Set[str] = set()
        self._forced_planner: Optional[str] = None

        logger.info(f"[POLICY] Planners ativos: {[k for k, v in self._planners.items() if v]}")

    def enable_planner(self, name: str) -> None:
        self._planners[name] = True
        logger.info(f"[POLICY] Planner '{name}' ativado")

    def disable_planner(self, name: str) -> None:
        self._planners[name] = False
        logger.info(f"[POLICY] Planner '{name}' desativado")

    def select_planner(self, name: Optional[str]) -> None:
        self._forced_planner = name
        logger.info(f"[POLICY] Planner forçado: {name}")

    def disable_skill(self, skill_name: str) -> None:
        self._disabled_skills.add(skill_name)
        logger.info(f"[POLICY] Skill '{skill_name}' desativada")

    def enable_skill(self, skill_name: str) -> None:
        self._disabled_skills.discard(skill_name)
        logger.info(f"[POLICY] Skill '{skill_name}' ativada")

    def is_planner_active(self, name: str) -> bool:
        if self._forced_planner and self._forced_planner != name:
            return False
        return self._planners.get(name, False)

    def is_skill_active(self, skill_name: str) -> bool:
        return skill_name not in self._disabled_skills

    def status(self) -> dict:
        return {
            "planners": dict(self._planners),
            "forced_planner": self._forced_planner,
            "disabled_skills": list(self._disabled_skills),
        }


# ──────────────────────────────────────────────────────────────────────────────
# NPCBehaviorController — composição dos três controladores por sessão
# ──────────────────────────────────────────────────────────────────────────────

class NPCBehaviorController:
    """
    Composição de GoalController + ExplorationController + PolicyController
    para um único NPC/sessão. Instanciado por ClientSession.
    """

    def __init__(self, session_id: str, behavior_config) -> None:
        self.session_id = session_id
        self.goal = GoalController()
        self.exploration = ExplorationController(
            mode=getattr(behavior_config, "default_exploration_mode", "auto"),
            epsilon=getattr(behavior_config, "epsilon", 0.1),
        )
        self.policy = PolicyController(behavior_config)

        # Define objetivo inicial baseado no estado idle
        self.goal.set_goal(
            "explore_environment",
            priority=0.5,
            max_steps=getattr(behavior_config, "goal_max_steps", 1000),
        )

    def tick(self, npc_state: int, confidence: float) -> None:
        """
        Tick chamado a cada request processado.
        Atualiza goal timer e ajusta exploration baseado em confiança.
        """
        expired = self.goal.tick()
        if expired:
            # Redefine objetivo baseado no estado atual
            goal_map = {
                0: "explore_environment",   # Idle
                1: "casual_movement",       # CasualMovement
                4: "engage_combat",         # Combat
                5: "stealth_approach",      # Stealth
                6: "flee_threat",           # Flee
            }
            new_goal = goal_map.get(npc_state, "explore_environment")
            self.goal.set_goal(new_goal, priority=0.8,
                               max_steps=self.goal.steps_in_goal or 1000)

        # Ajusta epsilon baseado em confiança do modelo
        if confidence > 0.8:
            self.exploration.set_mode("epsilon_greedy", epsilon=0.05)
        elif confidence < 0.3:
            self.exploration.set_mode("exploratory")
        else:
            self.exploration.set_mode("auto")

    def status(self) -> dict:
        return {
            "session_id": self.session_id,
            "goal": self.goal.current_goal_str(),
            "exploration_mode": self.exploration.mode,
            "policy": self.policy.status(),
        }
