from __future__ import annotations
import logging
import sys
import time
from typing import Optional


_FMT = "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)-32s | %(message)s"
_DATE = "%H:%M:%S"


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(_FMT, _DATE))
        logger.addHandler(h)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


class MotionLogger:
    def __init__(self, name: str, level: str = "INFO"):
        self._log = get_logger(name, level)
        self._timers: dict[str, float] = {}

    def timer_start(self, key: str) -> None:
        self._timers[key] = time.perf_counter()

    def timer_end(self, key: str) -> float:
        elapsed = (time.perf_counter() - self._timers.pop(key, time.perf_counter())) * 1000.0
        return elapsed

    def debug(self, msg: str, **kw) -> None:
        self._log.debug(msg, **({"extra": kw} if kw else {}))

    def info(self, msg: str, **kw) -> None:
        self._log.info(msg, **({"extra": kw} if kw else {}))

    def warning(self, msg: str, **kw) -> None:
        self._log.warning(msg, **({"extra": kw} if kw else {}))

    def error(self, msg: str, **kw) -> None:
        self._log.error(msg, **({"extra": kw} if kw else {}))

    def inference(self, session_id: str, latency_ms: float, confidence: float,
                  style: int, seq_id: int) -> None:
        self._log.info(
            f"[{session_id}] Inference | seq={seq_id} | "
            f"latency={latency_ms:.1f}ms | confidence={confidence:.3f} | style={style}"
        )
