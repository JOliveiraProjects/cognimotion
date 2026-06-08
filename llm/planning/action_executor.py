"""
planning/action_executor.py
============================
ActionExecutor — decodifica índice de ação discreta em (direction, speed).

Adaptado de realtime_brain.zip/action/executor.py:
  - Remove dependência de torch import no nível de módulo desnecessário
  - Adiciona action_map extensível por ECognitiveMotionStyle
  - Mantém decode / decode_batch / movement_action_indices
  - Suporte a mapeamento customizado para 9 estilos de movimento
"""
from __future__ import annotations

import math
import logging
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# Mapeamento padrão: ECognitiveMotionStyle → (direction, speed)
# Indexado por int value do enum (0–8)
# Semântica CANÔNICA — espelhada em learning/inverse_dynamics.py e no
# UCognitiveNPCBoneDriver (UE5). As três fontes DEVEM concordar, senão a ação
# que a política escolhe significa coisas diferentes em cada lado.
# 0=idle 1=forward 2=backward 3=left 4=right 5=run 6=jump 7=crouch 8=stop
DEFAULT_MOTION_ACTION_MAP: Dict[int, Tuple[List[float], float]] = {
    0: ([ 0.0,  0.0, 0.0], 0.00),   # idle
    1: ([ 1.0,  0.0, 0.0], 0.50),   # forward (walk)
    2: ([-1.0,  0.0, 0.0], 0.50),   # backward
    3: ([ 0.0, -1.0, 0.0], 0.50),   # left
    4: ([ 0.0,  1.0, 0.0], 0.50),   # right
    5: ([ 1.0,  0.0, 0.0], 1.00),   # run (forward fast)
    6: ([ 1.0,  0.0, 1.0], 0.60),   # jump (forward + up)
    7: ([ 1.0,  0.0, 0.0], 0.30),   # crouch (slow forward)
    8: ([ 0.0,  0.0, 0.0], 0.00),   # stop
}


class ActionExecutor:
    """
    Converte índice de ação discreta (ECognitiveMotionStyle int) em:
      (action_idx: int, move_direction: List[float], speed: float)

    Compatible com:
      - Policy.get_action() → action_idx
      - binary_protocol.build_motion_action()
      - UE5 UCognitiveMotionLearnerComponent
    """

    def __init__(
        self,
        action_dim: int = 9,
        discrete:   bool = True,
        custom_map: Optional[Dict[int, Tuple[List[float], float]]] = None,
    ) -> None:
        self.action_dim = action_dim
        self.discrete   = discrete

        if custom_map is not None:
            self.action_map = custom_map
        else:
            self.action_map = self._build_default_map(action_dim)

        logger.info(
            f"ActionExecutor | action_dim={action_dim} | discrete={discrete} "
            f"| movement_actions={len(self.movement_action_indices)}"
        )

    def _build_default_map(
        self, action_dim: int
    ) -> Dict[int, Tuple[List[float], float]]:
        """
        Usa o mapeamento padrão ECognitiveMotionStyle → (direction, speed).
        Para action_dim > 9, preenche com ações adicionais em anel.
        """
        result: Dict[int, Tuple[List[float], float]] = {}

        for i in range(action_dim):
            if i in DEFAULT_MOTION_ACTION_MAP:
                result[i] = DEFAULT_MOTION_ACTION_MAP[i]
            else:
                angle = math.pi * 2.0 * i / max(action_dim - 1, 1)
                result[i] = ([math.cos(angle), math.sin(angle), 0.0], 0.45)

        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Decode
    # ──────────────────────────────────────────────────────────────────────────

    def decode(
        self, action: int | torch.Tensor
    ) -> Tuple[int, List[float], float]:
        """
        Decodifica um índice de ação ou tensor.

        Args:
            action: int ou Tensor (1,) ou (action_dim,) one-hot ou logits

        Returns:
            (action_idx, move_direction, speed)
        """
        if isinstance(action, torch.Tensor):
            if action.dim() == 0:
                action_idx = int(action.item())
            elif self.discrete and action.dim() >= 1:
                probs      = torch.softmax(action.float(), dim=-1)
                action_idx = int(torch.argmax(probs).item())
            else:
                action_idx = int(action.item())
        else:
            action_idx = int(action)

        action_idx = max(0, min(action_idx, self.action_dim - 1))
        move_dir, speed = self.action_map[action_idx]
        return action_idx, list(move_dir), float(speed)

    def decode_batch(
        self, logits: torch.Tensor
    ) -> Tuple[List[int], List[List[float]], List[float]]:
        """
        Decodifica batch de logits/one-hots.
        logits: (B, action_dim)
        Returns: (idxs, dirs, speeds)
        """
        indices, dirs, speeds = [], [], []
        if self.discrete:
            acts = torch.argmax(logits, dim=-1)  # (B,)
            for i in range(acts.size(0)):
                idx, d, s = self.decode(int(acts[i].item()))
                indices.append(idx); dirs.append(d); speeds.append(s)
        else:
            for i in range(logits.size(0)):
                idx, d, s = self.decode(logits[i])
                indices.append(idx); dirs.append(d); speeds.append(s)
        return indices, dirs, speeds

    # ──────────────────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def movement_action_indices(self) -> List[int]:
        """Índices de ações que produzem movimento (speed > 0)."""
        return [i for i, (_, s) in self.action_map.items() if s > 0.0]

    @property
    def idle_action_index(self) -> int:
        """Índice da ação de idle/parada."""
        for i, (_, s) in self.action_map.items():
            if s == 0.0:
                return i
        return 0

    def action_name(self, idx: int) -> str:
        """Nome legível do estilo de movimento."""
        names = [
            "neutral", "aggressive", "relaxed", "injured",
            "fatigued", "stealth", "military", "civilian", "criminal",
        ]
        if 0 <= idx < len(names):
            return names[idx]
        return f"action_{idx}"
