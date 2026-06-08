from __future__ import annotations
import asyncio
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Awaitable
from memory.motion_replay_buffer import MotionReplayBuffer, ReplayEntry
from memory.motion_memory_bank import MotionMemoryBank
import logging

logger = logging.getLogger(__name__)


@dataclass
class DreamConfig:
    dream_interval_s: float = 30.0
    dream_batch_size: int = 128
    imagination_steps: int = 16
    diversity_injection_rate: float = 0.1
    entropy_scale: float = 0.05
    max_dream_duration_s: float = 10.0
    min_buffer_size: int = 256


@dataclass
class DreamResult:
    synthetic_embeddings: List[np.ndarray] = field(default_factory=list)
    styles: List[int] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    dream_duration_s: float = 0.0
    imagination_steps_done: int = 0


class MotionDreamScheduler:
    def __init__(
        self,
        replay_buffer: MotionReplayBuffer,
        memory_bank: MotionMemoryBank,
        config: DreamConfig = None,
        on_dream_complete: Optional[Callable[[DreamResult], Awaitable[None]]] = None,
    ):
        self.replay_buffer = replay_buffer
        self.memory_bank   = memory_bank
        self.config        = config or DreamConfig()
        self.on_dream_complete = on_dream_complete
        self._running      = False
        self._dream_count  = 0
        self._total_synthetic = 0

    async def run(self) -> None:
        self._running = True
        logger.info("MotionDreamScheduler started")
        while self._running:
            await asyncio.sleep(self.config.dream_interval_s)
            if self.replay_buffer.size() >= self.config.min_buffer_size:
                result = await self._dream()
                if self.on_dream_complete and result.synthetic_embeddings:
                    await self.on_dream_complete(result)

    async def stop(self) -> None:
        self._running = False

    async def _dream(self) -> DreamResult:
        t0 = time.perf_counter()
        result = DreamResult()

        batch = self.replay_buffer.sample(self.config.dream_batch_size)
        if batch is None:
            return result

        loop = asyncio.get_running_loop()
        synthetic = await loop.run_in_executor(None, self._imagine, batch)

        result.synthetic_embeddings    = [s["embedding"] for s in synthetic]
        result.styles                  = [s["style"] for s in synthetic]
        result.confidences             = [s["confidence"] for s in synthetic]
        result.dream_duration_s        = time.perf_counter() - t0
        result.imagination_steps_done  = len(synthetic)

        for s in synthetic:
            self.replay_buffer.add(
                embedding=s["embedding"],
                style=s["style"],
                confidence=s["confidence"],
            )

        self._dream_count    += 1
        self._total_synthetic += len(synthetic)
        logger.info(
            f"Dream {self._dream_count} | generated={len(synthetic)} | "
            f"duration={result.dream_duration_s*1000:.1f}ms"
        )
        return result

    def _imagine(self, batch: List[ReplayEntry]) -> List[dict]:
        synthetic = []
        embeddings = np.stack([e.embedding for e in batch], axis=0)
        dim = embeddings.shape[1]
        dream_start = time.perf_counter()

        for step in range(self.config.imagination_steps):
            idx_a = np.random.randint(0, len(batch))
            idx_b = np.random.randint(0, len(batch))
            alpha = np.random.beta(0.4, 0.4)
            blended = alpha * embeddings[idx_a] + (1.0 - alpha) * embeddings[idx_b]

            if np.random.random() < self.config.diversity_injection_rate:
                noise = np.random.randn(dim).astype(np.float32)
                noise /= max(np.linalg.norm(noise), 1e-8)
                blended += noise * self.config.entropy_scale

            norm = np.linalg.norm(blended)
            if norm > 1e-8:
                blended /= norm

            style_a = batch[idx_a].style
            style_b = batch[idx_b].style
            style = style_a if alpha > 0.5 else style_b

            conf = float(alpha * batch[idx_a].confidence + (1.0 - alpha) * batch[idx_b].confidence)
            conf = float(np.clip(conf * (1.0 - self.config.entropy_scale), 0.1, 1.0))

            synthetic.append({
                "embedding": blended.astype(np.float32),
                "style": style,
                "confidence": conf,
            })

            if (time.perf_counter() - dream_start) > self.config.max_dream_duration_s:
                break

        return synthetic

    def get_stats(self) -> dict:
        return {
            "dream_count": self._dream_count,
            "total_synthetic": self._total_synthetic,
            "config": {
                "interval_s": self.config.dream_interval_s,
                "batch_size": self.config.dream_batch_size,
                "imagination_steps": self.config.imagination_steps,
            },
        }
