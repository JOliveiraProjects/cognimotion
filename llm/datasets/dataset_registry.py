"""
datasets/dataset_registry.py
==============================
DatasetRegistry — carrega configuração YAML, gera sequências de treinamento
e injeta no SequenceBuffer do DreamerV3.

Uso:
    registry = DatasetRegistry.from_yaml("config/dataset_config.yaml")
    registry.load_into_buffer(sequence_buffer, obs_dim=256)
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from datasets.schemas import (
    InteractionAction, InteractionSequence, ObjectType,
    INTERACTION_ACTION_DIM,
)
from datasets.scenario_generator import ScenarioGenerator
from datasets.skeleton_targets import apply_bone_remap, register_custom

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """
    Configuração completa do dataset de interações.
    Pode ser carregada de YAML ou construída programaticamente.
    """
    # Escala de geração (1.0 = padrão, 0.1 = rápido/debug, 2.0 = dataset grande)
    scale:              float = 1.0
    seed:               int   = 42
    obs_dim:            int   = 256
    action_dim:         int   = INTERACTION_ACTION_DIM  # 24

    # Subsets habilitados
    enable_weapons:     bool  = True
    enable_ball:        bool  = True
    enable_threats:     bool  = True
    enable_vehicles:    bool  = True
    enable_mounts:      bool  = True
    enable_traffic:     bool  = True

    # Remapeamento de bones para skeleton customizado
    bone_remap:         Dict[str, str] = field(default_factory=dict)

    # Weights por categoria (para amostragem balanceada)
    category_weights: Dict[str, float] = field(default_factory=lambda: {
        "weapon":   1.0,
        "ball":     1.0,
        "threat":   1.5,   # ameaça é o cenário mais crítico — oversampled
        "vehicle":  1.0,
        "mount":    0.8,
        "traffic":  0.8,
    })

    @classmethod
    def from_yaml(cls, path: str) -> "DatasetConfig":
        import yaml
        with open(path) as f:
            raw = yaml.safe_load(f)
        cfg = cls()
        for k, v in (raw or {}).items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    def to_yaml(self, path: str) -> None:
        import yaml
        from dataclasses import asdict
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=True)


class DatasetRegistry:
    """
    Gerencia geração e carregamento de datasets de interação.

    Thread-safe: pode chamar load_into_buffer em background.
    """

    def __init__(self, config: Optional[DatasetConfig] = None) -> None:
        self.config    = config or DatasetConfig()
        self._gen      = ScenarioGenerator(seed=self.config.seed)
        self._lock     = threading.Lock()
        self._sequences: List[InteractionSequence] = []
        self._loaded   = False

    @classmethod
    def from_yaml(cls, path: str) -> "DatasetRegistry":
        cfg = DatasetConfig.from_yaml(path)
        return cls(config=cfg)

    @classmethod
    def default(cls) -> "DatasetRegistry":
        return cls(DatasetConfig())

    # ──────────────────────────────────────────────────────────────────────────

    def generate(self) -> List[InteractionSequence]:
        cfg = self.config

        if cfg.bone_remap:
            apply_bone_remap(cfg.bone_remap)

        seqs: List[InteractionSequence] = []

        def weighted_n(category: str, base: int) -> int:
            w = cfg.category_weights.get(category, 1.0)
            return max(1, int(base * cfg.scale * w))

        if cfg.enable_weapons:
            seqs += self._gen.gen_weapon_pickup(weighted_n("weapon", 50))
            seqs += self._gen.gen_weapon_aim(weighted_n("weapon", 30))

        if cfg.enable_ball:
            seqs += self._gen.gen_ball_kick(weighted_n("ball", 60))
            seqs += self._gen.gen_ball_grab(weighted_n("ball", 40))
            seqs += self._gen.gen_ball_push(weighted_n("ball", 30))

        if cfg.enable_threats:
            seqs += self._gen.gen_threat_detection(weighted_n("threat", 80))

        if cfg.enable_vehicles:
            seqs += self._gen.gen_vehicle_enter_exit(ObjectType.CAR,        weighted_n("vehicle", 50))
            seqs += self._gen.gen_vehicle_enter_exit(ObjectType.MOTORCYCLE, weighted_n("vehicle", 40))
            seqs += self._gen.gen_vehicle_enter_exit(ObjectType.BICYCLE,    weighted_n("vehicle", 30))

        if cfg.enable_traffic:
            seqs += self._gen.gen_traffic_light_response(weighted_n("traffic", 60))

        if cfg.enable_mounts:
            seqs += self._gen.gen_mount_dismount(ObjectType.HORSE,   weighted_n("mount", 40))
            seqs += self._gen.gen_mount_dismount(ObjectType.BICYCLE, weighted_n("mount", 30))

        with self._lock:
            self._sequences = seqs
            self._loaded    = True

        logger.info(
            f"DatasetRegistry | {len(seqs)} sequências geradas "
            f"| scale={cfg.scale} | seed={cfg.seed}"
        )
        return seqs

    def load_into_buffer(self, sequence_buffer, obs_dim: Optional[int] = None) -> int:
        """
        Gera sequências e injeta no SequenceBuffer do DreamerV3.
        Retorna número de sequências carregadas.

        sequence_buffer: runtime.SequenceBuffer
        obs_dim: se None, usa config.obs_dim
        """
        odim = obs_dim or self.config.obs_dim
        seqs = self.generate()
        loaded = 0

        for seq in seqs:
            try:
                obs_arr, act_arr, rew_arr, done_arr = seq.to_numpy_arrays(obs_dim=odim, action_dim=self.config.action_dim)
                if len(obs_arr) < 2:
                    continue
                group_key = f"dataset_{seq.object_type.name.lower()}"
                sequence_buffer.add_sequence(
                    obs_seq=obs_arr,
                    action_seq=act_arr,
                    reward_seq=rew_arr,
                    done_seq=done_arr,
                    group_key=group_key,
                )
                loaded += 1
            except Exception as exc:
                logger.warning(f"DatasetRegistry | seq {seq.scenario_id} falhou: {exc}")

        logger.info(
            f"DatasetRegistry | {loaded}/{len(seqs)} sequências carregadas no SequenceBuffer"
        )
        return loaded

    def load_into_buffer_async(self, sequence_buffer, obs_dim: Optional[int] = None) -> threading.Thread:
        """Carrega em thread de background. Não bloqueia o servidor."""
        t = threading.Thread(
            target=self.load_into_buffer,
            args=(sequence_buffer, obs_dim),
            daemon=True,
            name="DatasetLoader",
        )
        t.start()
        return t

    # ──────────────────────────────────────────────────────────────────────────

    def stats(self) -> Dict:
        with self._lock:
            seqs = self._sequences
        if not seqs:
            return {"loaded": False}

        by_type: Dict[str, int] = {}
        total_steps = 0
        success_rate = 0.0
        for s in seqs:
            key = s.object_type.name
            by_type[key] = by_type.get(key, 0) + 1
            total_steps += len(s.steps)
            success_rate += float(s.success)

        return {
            "loaded":        True,
            "total_seqs":    len(seqs),
            "total_steps":   total_steps,
            "success_rate":  round(success_rate / max(len(seqs), 1), 3),
            "by_object_type": by_type,
        }

    @property
    def sequences(self) -> List[InteractionSequence]:
        with self._lock:
            return list(self._sequences)

    @property
    def is_loaded(self) -> bool:
        with self._lock:
            return self._loaded
