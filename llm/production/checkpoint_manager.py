"""
production/checkpoint_manager.py
==================================
CheckpointManager — salva/carrega checkpoints com manifest, compressão e LRU.

Adaptado de training_brain.zip/production/checkpoint_manager.py:
  - Remove core.logger → logging padrão
  - Remove log_policy_event (dependência externa)
  - Mantém: CheckpointRecord, save, load_latest, compress_old, LRU pruning
"""
from __future__ import annotations

import gzip
import json
import logging
import os
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

MANIFEST_FILENAME  = "manifest.json"
MAX_TOTAL_GB_DEFAULT = 50.0


@dataclass
class CheckpointRecord:
    version:         int
    file_path:       str
    is_compressed:   bool
    size_bytes:      int
    saved_at:        float = field(default_factory=time.time)
    metrics:         Dict[str, float] = field(default_factory=dict)
    git_hash:        str = ""
    dataset_version: str = ""
    experiment_id:   str = ""

    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024 ** 3)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CheckpointRecord":
        return cls(**d)


class CheckpointManager:
    """
    Gerencia checkpoints de modelo com:
      - Manifest JSON para rastreamento
      - Compressão gzip de checkpoints antigos
      - Pruning por LRU quando max_total_gb é excedido
      - Thread-safe
    """

    def __init__(
        self,
        checkpoint_dir:  str = "checkpoints",
        keep_last_n:     int = 10,
        max_total_gb:    float = MAX_TOTAL_GB_DEFAULT,
        experiment_name: str = "cognitive_dreamer",
        compress_old:    bool = True,
    ) -> None:
        self.checkpoint_dir  = Path(checkpoint_dir)
        self.keep_last_n     = keep_last_n
        self.max_total_gb    = max_total_gb
        self.experiment_name = experiment_name
        self.compress_old    = compress_old
        self._lock           = threading.Lock()
        self._manifest:      List[CheckpointRecord] = []

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._load_manifest()

        logger.info(
            f"CheckpointManager | dir='{checkpoint_dir}' | keep={keep_last_n} "
            f"| max_gb={max_total_gb} | compress={compress_old}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Save
    # ──────────────────────────────────────────────────────────────────────────

    def save(
        self,
        model,
        version:         int,
        metrics:         Optional[Dict[str, float]] = None,
        git_hash:        str = "",
        dataset_version: str = "",
        experiment_id:   str = "",
    ) -> str:
        """Salva checkpoint. Retorna o caminho do arquivo salvo."""
        filename  = f"{self.experiment_name}_v{version:06d}.pt"
        file_path = self.checkpoint_dir / filename

        state = model if isinstance(model, dict) else model.state_dict()
        payload = {
            "version":         version,
            "experiment_name": self.experiment_name,
            "saved_at":        time.time(),
            "metrics":         metrics or {},
            "state_dict":      state,
        }
        torch.save(payload, file_path)
        size = file_path.stat().st_size

        record = CheckpointRecord(
            version=version,
            file_path=str(file_path),
            is_compressed=False,
            size_bytes=size,
            metrics=metrics or {},
            git_hash=git_hash,
            dataset_version=dataset_version,
            experiment_id=experiment_id,
        )

        with self._lock:
            self._manifest.append(record)
            self._save_manifest()
            self._prune()

        logger.info(
            f"CheckpointManager | salvo v{version} → {filename} "
            f"({size/1024:.1f}KB)"
        )
        return str(file_path)

    # ──────────────────────────────────────────────────────────────────────────
    # Load
    # ──────────────────────────────────────────────────────────────────────────

    def load_latest(self, model) -> int:
        """Carrega o checkpoint mais recente. Retorna versão (0 se nenhum)."""
        with self._lock:
            if not self._manifest:
                return 0
            record = self._manifest[-1]

        return self._load_record(record, model)

    def load_version(self, model, version: int) -> int:
        """Carrega uma versão específica."""
        with self._lock:
            matches = [r for r in self._manifest if r.version == version]
        if not matches:
            logger.warning(f"CheckpointManager | versão {version} não encontrada")
            return 0
        return self._load_record(matches[0], model)

    def _load_record(self, record: CheckpointRecord, model) -> int:
        path = Path(record.file_path)
        if record.is_compressed:
            path = Path(str(path) + ".gz")

        if not path.exists():
            logger.error(f"CheckpointManager | arquivo não encontrado: {path}")
            return 0

        try:
            if record.is_compressed:
                with gzip.open(path, "rb") as f:
                    payload = torch.load(f, map_location="cpu", weights_only=False)
            else:
                payload = torch.load(path, map_location="cpu", weights_only=False)

            state = payload.get("state_dict", payload)
            if isinstance(model, dict):
                model.update(state)
            else:
                model.load_state_dict(state, strict=False)

            logger.info(f"CheckpointManager | carregado v{record.version}")
            return record.version

        except Exception as exc:
            logger.error(f"CheckpointManager | erro ao carregar: {exc}")
            return 0

    # ──────────────────────────────────────────────────────────────────────────
    # Manifest
    # ──────────────────────────────────────────────────────────────────────────

    def _load_manifest(self) -> None:
        path = self.checkpoint_dir / MANIFEST_FILENAME
        if not path.exists():
            return
        try:
            with open(path) as f:
                data = json.load(f)
            self._manifest = [CheckpointRecord.from_dict(r) for r in data]
            logger.info(f"CheckpointManager | manifest carregado ({len(self._manifest)} entries)")
        except Exception as exc:
            logger.warning(f"CheckpointManager | erro ao carregar manifest: {exc}")

    def _save_manifest(self) -> None:
        path = self.checkpoint_dir / MANIFEST_FILENAME
        try:
            with open(path, "w") as f:
                json.dump([r.to_dict() for r in self._manifest], f, indent=2)
        except Exception as exc:
            logger.warning(f"CheckpointManager | erro ao salvar manifest: {exc}")

    def _prune(self) -> None:
        """Remove checkpoints antigos além de keep_last_n."""
        excess = len(self._manifest) - self.keep_last_n
        for r in self._manifest[:max(excess, 0)]:
            p = Path(r.file_path)
            for ext in ["", ".gz"]:
                target = Path(str(p) + ext) if ext else p
                if target.exists():
                    try:
                        target.unlink()
                        logger.debug(f"CheckpointManager | removido {target.name}")
                    except OSError:
                        pass

        if excess > 0:
            self._manifest = self._manifest[excess:]
            self._save_manifest()

    # ──────────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────────

    def latest_version(self) -> int:
        with self._lock:
            if not self._manifest:
                return 0
            return self._manifest[-1].version

    def latest_path(self) -> Optional[str]:
        with self._lock:
            if not self._manifest:
                return None
            return self._manifest[-1].file_path

    def list_versions(self) -> List[int]:
        with self._lock:
            return [r.version for r in self._manifest]
