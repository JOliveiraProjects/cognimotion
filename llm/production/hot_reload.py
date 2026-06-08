"""
production/hot_reload.py
=========================
AtomicPolicyLoader — hot-reload atômico de políticas sem parar o servidor.

Adaptado de realtime_brain.zip/realtime/hot_reload.py:
  - Remove core.logger → logging padrão
  - Remove log_policy_event → logging simples
  - Mantém: FallbackPolicy, PromotionResult, AtomicPolicyLoader
  - Usado pelo DreamerTrainer para promover checkpoints em produção
"""
from __future__ import annotations

import copy
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class PromotionResult(str, Enum):
    SUCCESS           = "success"
    VALIDATION_FAILED = "validation_failed"
    LOAD_FAILED       = "load_failed"
    DRIFT_DETECTED    = "drift_detected"
    FALLBACK_ACTIVE   = "fallback_active"
    ALREADY_CURRENT   = "already_current"


@dataclass
class PolicyRecord:
    version:          int
    checkpoint_path:  str
    loaded_at:        float = field(default_factory=time.time)
    metrics:          Dict[str, float] = field(default_factory=dict)
    is_stable:        bool = False
    promotion_result: str = ""


class FallbackPolicy:
    """Política de fallback que retorna ação idle."""

    def __call__(self, *args, **kwargs) -> Tuple[int, List[float], float]:
        return 0, [0.0, 0.0, 0.0], 0.0

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return torch.zeros(1, 9)

    def parameters(self):
        return iter([])

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> None:
        # FallbackPolicy não tem parâmetros treináveis.
        # Ignorar state_dict é o comportamento correto — não há pesos a restaurar.
        return


class AtomicPolicyLoader:
    """
    Carrega checkpoints de forma atômica (sem parar o servidor).

    Usa modelo sombra (shadow) para carregar enquanto o modelo vivo (live)
    continua servindo. Troca atômica via lock quando pronto.
    """

    def __init__(
        self,
        live_model,
        shadow_model,
        model_lock:       threading.RLock,
        device:           Optional[str] = None,
        fallback_policy:  Optional[FallbackPolicy] = None,
        max_history:      int = 5,
        hidden_dim:       int = 512,
        stochastic_dim:   int = 1024,
    ) -> None:
        self.live_model    = live_model
        self.shadow_model  = shadow_model
        self.model_lock    = model_lock
        self.device        = device
        self.fallback      = fallback_policy or FallbackPolicy()
        self.max_history   = max_history
        self.hidden_dim    = hidden_dim
        self.stochastic_dim = stochastic_dim

        self._current_version: int        = 0
        self._history: List[PolicyRecord] = []
        self._use_fallback: bool          = False
        self._promotion_lock = threading.Lock()

        logger.info(f"AtomicPolicyLoader | device={device} | max_history={max_history}")

    # ──────────────────────────────────────────────────────────────────────────

    def try_promote(
        self,
        checkpoint_path: str,
        version:         int,
        metrics:         Optional[Dict[str, float]] = None,
        experiment_id:   str = "",
    ) -> PromotionResult:
        with self._promotion_lock:
            return self._promote_internal(checkpoint_path, version, metrics or {}, experiment_id)

    def _promote_internal(
        self,
        checkpoint_path: str,
        version:         int,
        metrics:         Dict[str, float],
        experiment_id:   str,
    ) -> PromotionResult:
        if version == self._current_version:
            return PromotionResult.ALREADY_CURRENT

        if self._use_fallback:
            logger.warning(f"HotReload | fallback ativo, não promovendo v{version}")
            return PromotionResult.FALLBACK_ACTIVE

        # Carrega no shadow
        try:
            payload = torch.load(
                checkpoint_path,
                map_location=self.device or "cpu",
                weights_only=False,
            )
            state = payload.get("state_dict", payload)
        except Exception as exc:
            logger.error(f"HotReload | falha ao carregar {checkpoint_path}: {exc}")
            return PromotionResult.LOAD_FAILED

        try:
            self.shadow_model.load_state_dict(state, strict=False)
        except Exception as exc:
            logger.error(f"HotReload | load_state_dict falhou: {exc}")
            return PromotionResult.VALIDATION_FAILED

        # Troca atômica
        with self.model_lock:
            live_state   = copy.deepcopy(self.shadow_model.state_dict())
            self.live_model.load_state_dict(live_state, strict=False)
            self._current_version = version

        record = PolicyRecord(
            version=version,
            checkpoint_path=checkpoint_path,
            metrics=metrics,
            is_stable=True,
            promotion_result=PromotionResult.SUCCESS,
        )
        self._history.append(record)
        if len(self._history) > self.max_history:
            self._history.pop(0)

        logger.info(
            f"HotReload | promovido v{version} | "
            f"metrics={metrics}"
        )
        return PromotionResult.SUCCESS

    def activate_fallback(self) -> None:
        self._use_fallback = True
        logger.warning("HotReload | fallback ativado")

    def deactivate_fallback(self) -> None:
        self._use_fallback = False
        logger.info("HotReload | fallback desativado")

    def rollback_to_previous(self) -> bool:
        if len(self._history) < 2:
            logger.warning("HotReload | sem versão anterior para rollback")
            return False
        prev = self._history[-2]
        result = self.try_promote(prev.checkpoint_path, prev.version)
        return result == PromotionResult.SUCCESS

    @property
    def current_version(self) -> int:
        return self._current_version

    @property
    def is_using_fallback(self) -> bool:
        return self._use_fallback

    def should_check_drift(self, kl: float, rw_drop: float) -> bool:
        return kl > 0.5 or rw_drop > 0.2
