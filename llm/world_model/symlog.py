"""
world_model/symlog.py
=====================
Utilitários de estabilização DreamerV3:
  - symlog / symexp: transformações simétricas log para reward/value
  - unimix_categorical: mistura prior com uniforme (evita colapso)
  - EMANormalizer: normalização de targets com média móvel exponencial
  - two_hot_encode: codificação two-hot para distribuição de valores (DreamerV3)
"""
from __future__ import annotations

import math
import threading
from typing import Tuple

import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# symlog / symexp
# ──────────────────────────────────────────────────────────────────────────────

def symlog(x: torch.Tensor) -> torch.Tensor:
    """
    Transformação simétrica logarítmica.
    symlog(x) = sign(x) * log(1 + |x|)

    Comprime grandes valores preservando o sinal. Usada para reward/value
    antes de entrar no RSSM e no crítico.
    """
    return x.sign() * (x.abs() + 1.0).log()


def symexp(x: torch.Tensor) -> torch.Tensor:
    """
    Inversa de symlog.
    symexp(x) = sign(x) * (exp(|x|) - 1)
    """
    return x.sign() * (x.abs().exp() - 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# unimix_categorical
# ──────────────────────────────────────────────────────────────────────────────

def unimix_categorical(
    logits: torch.Tensor,
    mix: float = 0.01,
) -> torch.Tensor:
    """
    Mistura a distribuição categórica aprendida com uma uniforme.
    Previne colapso estocástico do prior no RSSM.

    p_final = (1 - mix) * softmax(logits) + mix * uniform

    Args:
        logits: (..., num_classes) — logits crus do prior ou posterior
        mix:    fração de mistura uniforme (padrão DreamerV3 = 0.01)

    Returns:
        Distribuição misturada como tensor de probabilidades (mesmo shape dos logits).
    """
    probs   = F.softmax(logits, dim=-1)
    uniform = torch.ones_like(probs) / probs.size(-1)
    return (1.0 - mix) * probs + mix * uniform


def unimix_logits(
    logits: torch.Tensor,
    mix: float = 0.01,
) -> torch.Tensor:
    """
    Retorna logits equivalentes após aplicar unimix.
    Útil quando a função downstream espera logits, não probs.
    """
    probs = unimix_categorical(logits, mix)
    return (probs + 1e-8).log()


# ──────────────────────────────────────────────────────────────────────────────
# EMANormalizer
# ──────────────────────────────────────────────────────────────────────────────

class EMANormalizer:
    """
    Normalização de targets com média e variância móvel exponencial (EMA).
    Usada para normalizar returns/advantages antes de calcular a perda do crítico.

    Thread-safe via threading.Lock.
    """

    def __init__(
        self,
        decay: float = 0.99,
        epsilon: float = 1e-6,
        clip: float = 10.0,
        warmup_steps: int = 100,
    ) -> None:
        self.decay   = decay
        self.epsilon = epsilon
        self.clip    = clip
        self.warmup  = warmup_steps

        self._mean: float    = 0.0
        self._var: float     = 1.0
        self._count: int     = 0
        self._lock           = threading.Lock()

    def update(self, x: torch.Tensor) -> None:
        """Atualiza estatísticas EMA com tensor de valores."""
        with self._lock:
            vals = x.detach().float().flatten().tolist()
            for v in vals:
                self._count += 1
                self._mean = self.decay * self._mean + (1.0 - self.decay) * v
                self._var  = self.decay * self._var  + (1.0 - self.decay) * (v - self._mean) ** 2

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normaliza x com as estatísticas EMA acumuladas."""
        with self._lock:
            if self._count < self.warmup:
                return x
            std = math.sqrt(max(self._var, 0.0)) + self.epsilon
            normed = (x - self._mean) / std

        return normed.clamp(-self.clip, self.clip)

    def update_and_normalize(self, x: torch.Tensor) -> torch.Tensor:
        self.update(x)
        return self.normalize(x)

    @property
    def mean(self) -> float:
        with self._lock:
            return self._mean

    @property
    def std(self) -> float:
        with self._lock:
            return math.sqrt(max(self._var, 0.0)) + self.epsilon


# ──────────────────────────────────────────────────────────────────────────────
# two_hot_encode / two_hot_decode  (DreamerV3 value distribution)
# ──────────────────────────────────────────────────────────────────────────────

def two_hot_encode(
    x: torch.Tensor,
    bins: torch.Tensor,
) -> torch.Tensor:
    """
    Codificação two-hot para distribuição discreta de valores.
    Interpola entre os dois bins mais próximos.

    Args:
        x:    tensor de valores escalares (B,)
        bins: tensor dos centros de bins (K,)

    Returns:
        Tensor two-hot (B, K) com valores em [0, 1] somando 1.
    """
    B   = x.shape[0]
    K   = bins.shape[0]
    x_c = x.unsqueeze(1).expand(B, K)
    b_c = bins.unsqueeze(0).expand(B, K)

    below = (b_c <= x_c).sum(-1).clamp(0, K - 2)
    above = below + 1

    b_low  = bins[below]
    b_high = bins[above]
    alpha  = ((x - b_low) / (b_high - b_low + 1e-8)).clamp(0.0, 1.0)

    th = torch.zeros(B, K, device=x.device)
    th.scatter_(1, below.unsqueeze(1),  (1.0 - alpha).unsqueeze(1))
    th.scatter_(1, above.unsqueeze(1),  alpha.unsqueeze(1))
    return th


def two_hot_decode(probs: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Expectativa da distribuição two-hot: E[x] = sum(probs * bins)."""
    return (probs * bins.unsqueeze(0)).sum(-1)
