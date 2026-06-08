"""
policy_registry.py
==================
Gerencia checkpoints de política (model state_dicts).
Adaptado de itens.zip — remove dependência de core.logger.
Compatível com multiprocessing: usa threading.RLock (seguro dentro de um processo).
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


class PolicyRegistry:
    """
    Salva/carrega checkpoints de política com validação por reward médio.

    Thread-safe. Pode ser usado pelo TrainingProcess e pelo MainProcess
    para sincronizar pesos entre workers.
    """

    def __init__(
        self,
        save_dir: str = "runs/policies",
        min_reward_threshold: float = -5.0,
        validation_window: int = 50,
        keep_last_n: int = 10,
        publish_min_interval: int = 50,
        category: str = "Default",
        skeleton_signature: str = "",
    ) -> None:
        self._save_dir = Path(save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._threshold = min_reward_threshold
        self._window_sz = max(validation_window, 2)
        self._keep_last = keep_last_n
        self._publish_min_interval = max(publish_min_interval, 1)
        self._lock = threading.RLock()
        self._version = 0
        self._publish_calls = 0
        self._last_saved_call = 0
        self._rewards: deque[float] = deque(maxlen=self._window_sz)

        # Treino acumulativo por categoria: cada categoria tem UM arquivo único
        # ("policy_<categoria>.pt") que é retomado e sobrescrito, ficando mais
        # completo a cada sessão — em vez de criar v1, v2, v3 soltos.
        self._category = self._sanitize(category)
        self._skeleton_sig = skeleton_signature
        self._accumulating_path = self._save_dir / f"policy_{self._category}.pt"

        self._step_accum = 0.0
        self._step_accum_count = 0

        logger.info(
            f"PolicyRegistry | dir={save_dir} | categoria={self._category} "
            f"| arquivo acumulativo={self._accumulating_path.name} "
            f"| threshold={min_reward_threshold}"
        )

    @staticmethod
    def _sanitize(name: str) -> str:
        """Torna a categoria segura como nome de arquivo (Urbano|MMA → Urbano_MMA)."""
        keep = [c if (c.isalnum() or c in "-_") else "_" for c in (name or "Default")]
        return "".join(keep) or "Default"

    # ──────────────────────────────────────────────────────────────────────────
    # Registro de rewards
    # ──────────────────────────────────────────────────────────────────────────

    def record_reward(self, episode_reward: float) -> None:
        """Registra o reward total de um episódio."""
        with self._lock:
            self._rewards.append(float(episode_reward))

    def record_step_reward(self, step_reward: float, n_steps: int = 1) -> None:
        """
        Acumula rewards de step e registra um episódio sintético quando
        n_steps steps forem acumulados.
        """
        with self._lock:
            self._step_accum += step_reward
            self._step_accum_count += 1
            if self._step_accum_count >= max(n_steps, 1):
                self._rewards.append(self._step_accum)
                self._step_accum = 0.0
                self._step_accum_count = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Consultas de estado
    # ──────────────────────────────────────────────────────────────────────────

    def can_publish(self) -> bool:
        with self._lock:
            n = len(self._rewards)
            if n < max(self._window_sz // 2, 1):
                return False
            return self.mean_reward > self._threshold

    @property
    def has_reward_data(self) -> bool:
        with self._lock:
            return len(self._rewards) > 0

    @property
    def mean_reward(self) -> float:
        with self._lock:
            if not self._rewards:
                return float("-inf")
            return sum(self._rewards) / len(self._rewards)

    @property
    def reward_count(self) -> int:
        with self._lock:
            return len(self._rewards)

    @property
    def current_version(self) -> int:
        with self._lock:
            return self._version

    # ──────────────────────────────────────────────────────────────────────────
    # Publicação / carga de checkpoints
    # ──────────────────────────────────────────────────────────────────────────

    def autosave(self, model_state_dict: Dict[str, Any]) -> bool:
        """
        Salva o estado ATUAL no arquivo acumulativo da categoria, SEM as
        condições de publicação (threshold/intervalo). Garante que parar o
        servidor não perca o progresso de treino. Chamado periodicamente.
        """
        payload = {
            "version": self._version,
            "timestamp": time.time(),
            "mean_reward": self.mean_reward if self.reward_count else 0.0,
            "state_dict": model_state_dict,
            "metadata": {"autosave": True},
            "skeleton_signature": self._skeleton_sig,
            "category": self._category,
        }
        try:
            torch.save(payload, self._accumulating_path)
            logger.info(
                f"[AUTOSAVE] treino '{self._category}' salvo → "
                f"{self._accumulating_path.name} (v{self._version})"
            )
            return True
        except Exception as exc:
            logger.warning(f"PolicyRegistry: autosave falhou: {exc}")
            return False

    def publish(
        self,
        model_state_dict: Dict[str, Any],
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Salva um checkpoint se o reward médio for suficiente.
        Retorna o número da versão salva, ou -1 se ignorado.
        """
        with self._lock:
            self._publish_calls += 1
            n_episodes = len(self._rewards)

        if n_episodes == 0:
            logger.debug("PolicyRegistry: skip publish — sem episódios registrados")
            return -1

        with self._lock:
            calls_since_last = self._publish_calls - self._last_saved_call
        if calls_since_last < self._publish_min_interval:
            return -1

        mean_r = self.mean_reward
        if mean_r <= self._threshold:
            logger.warning(
                f"PolicyRegistry: publish rejeitado | mean_reward={mean_r:.3f} "
                f"< threshold={self._threshold:.3f}"
            )
            return -1

        with self._lock:
            self._version += 1
            version = self._version
            self._last_saved_call = self._publish_calls

        path = self._save_dir / f"policy_v{version:06d}.pt"
        payload = {
            "version": version,
            "timestamp": time.time(),
            "mean_reward": mean_r,
            "state_dict": model_state_dict,
            "metadata": extra_metadata or {},
            # Assinatura do skeleton com que foi treinado — usada para impedir
            # carregar o modelo num skeleton incompatível (Python ou .pt).
            "skeleton_signature": self._skeleton_sig,
            "category": self._category,
        }
        torch.save(payload, path)
        # Arquivo ACUMULATIVO da categoria: sempre aponta para o treino mais
        # completo desta categoria. É o que será retomado na próxima sessão.
        try:
            torch.save(payload, self._accumulating_path)
        except Exception as exc:
            logger.warning(f"PolicyRegistry: falha ao gravar arquivo acumulativo: {exc}")
        logger.info(
            f"[CHECKPOINT SALVO] v{version} ({self._category}) → {path} "
            f"(mean_reward={mean_r:.3f}, episodes={n_episodes})"
        )
        self._prune_old_checkpoints()
        return version

    def load_latest(self, model: Any) -> int:
        """
        Carrega o checkpoint mais recente em `model`.
        Retorna a versão carregada (0 se nenhum checkpoint existir).
        """
        checkpoints = sorted(self._save_dir.glob("policy_v*.pt"))
        if not checkpoints:
            logger.info("PolicyRegistry: nenhum checkpoint encontrado, iniciando do zero.")
            return 0

        latest = checkpoints[-1]
        try:
            payload = torch.load(latest, map_location="cpu", weights_only=False)
            model.load_state_dict(payload["state_dict"])
            version = int(payload["version"])
        except Exception as exc:
            logger.error(f"PolicyRegistry: falha ao carregar {latest}: {exc}")
            return 0

        with self._lock:
            self._version = version

        logger.info(
            f"[CHECKPOINT CARREGADO] v{version} de {latest} "
            f"(mean_reward={payload.get('mean_reward', 'n/a')})"
        )
        return version

    def get_latest_path(self) -> Optional[Path]:
        """Retorna o caminho do checkpoint mais recente sem carregá-lo."""
        checkpoints = sorted(self._save_dir.glob("policy_v*.pt"))
        return checkpoints[-1] if checkpoints else None

    def resume_category(self, model: Any, current_profile=None) -> int:
        """
        Retoma o treino ACUMULATIVO desta categoria: carrega o arquivo único
        'policy_<categoria>.pt' em `model`, continuando de onde parou — em vez
        de recomeçar do zero a cada sessão.

        Se `current_profile` (SkeletonProfile) for fornecido, VERIFICA se o
        skeleton do checkpoint é compatível. Se não for, NÃO carrega e retorna
        -1, deixando uma mensagem de erro clara no log (e em last_error).
        """
        self.last_error: str = ""
        if not self._accumulating_path.exists():
            logger.info(
                f"PolicyRegistry: sem treino anterior para '{self._category}'. "
                "Começando um treino novo (será salvo de forma acumulativa)."
            )
            return 0

        try:
            payload = torch.load(self._accumulating_path, map_location="cpu",
                                 weights_only=False)
        except Exception as exc:
            logger.error(f"PolicyRegistry: falha ao ler {self._accumulating_path}: {exc}")
            return 0

        # ── Verificação de compatibilidade de skeleton ──────────────────────
        trained_sig = payload.get("skeleton_signature", "")
        if current_profile is not None:
            from world_model.skeleton_profile import check_compatibility
            err = check_compatibility(trained_sig, current_profile)
            if err:
                self.last_error = err
                logger.error("\n" + "=" * 60 + "\n" + err + "\n" + "=" * 60)
                return -1

        try:
            state = payload["state_dict"]
            # O checkpoint é salvo no formato aninhado de submódulos
            # ({"rssm":..., "decoder":..., "actor":..., "critic":...}).
            # Se o `model` tiver load_checkpoint_state (DreamerTrainer), usamos
            # ele para restaurar submódulo por submódulo. Senão, tentamos o
            # load_state_dict direto (compatibilidade com formato achatado).
            if hasattr(model, "load_checkpoint_state"):
                model.load_checkpoint_state(state)
            else:
                model.load_state_dict(state)
            version = int(payload.get("version", 0))
        except Exception as exc:
            logger.error(
                f"PolicyRegistry: checkpoint de '{self._category}' não pôde ser "
                f"carregado ({exc}). Provavelmente um arquivo antigo/incompatível. "
                f"Renomeando e começando um treino novo (será salvo de forma acumulativa)."
            )
            # Move o arquivo problemático para .bak para não travar o treino nem
            # perder o arquivo. O autosave criará um novo no formato correto.
            try:
                backup = self._accumulating_path.with_suffix(".pt.bak")
                if backup.exists():
                    backup.unlink()
                self._accumulating_path.rename(backup)
            except Exception:
                pass
            return 0

        with self._lock:
            self._version = version
        logger.info(
            f"[TREINO RETOMADO] categoria '{self._category}' | v{version} | "
            f"continuando de mean_reward={payload.get('mean_reward', 'n/a')}"
        )
        return version

    def _prune_old_checkpoints(self) -> None:
        if self._keep_last <= 0:
            return
        checkpoints = sorted(self._save_dir.glob("policy_v*.pt"))
        excess = len(checkpoints) - self._keep_last
        for old in checkpoints[:excess]:
            try:
                old.unlink()
                logger.debug(f"PolicyRegistry: removido {old}")
            except OSError as exc:
                logger.warning(f"PolicyRegistry: não foi possível remover {old}: {exc}")
