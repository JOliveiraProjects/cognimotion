"""
continuous_trainer.py
=====================
Thread de treinamento contínuo em background.
Adaptado de itens.zip — remove dependência de core.logger,
adapta interface para OnlineImitationLearner do projeto.
"""
from __future__ import annotations

import logging
import threading
from learning.learning_monitor import LearningMonitor
import time
from typing import Optional

logger = logging.getLogger(__name__)


class ContinuousTrainer(threading.Thread):
    """
    Thread daemon que executa train_step() do OnlineImitationLearner
    continuamente, governada por evento ou poll periódico.

    Interface esperada de `learner`:
      - learner.train_step(batch_size: int) -> Optional[dict]
      - learner.replay_buffer.size() -> int
      - learner.replay_buffer.get_stats() -> dict
    """

    def __init__(
        self,
        learner,          # OnlineImitationLearner
        batch_size: int = 64,
        min_buffer: int = 512,
        train_interval_s: float = 0.5,
    ) -> None:
        super().__init__(daemon=True, name="CognitiveTrainer")
        self.learner = learner
        self.batch_size = batch_size
        self.min_buffer = min_buffer
        self.train_interval_s = train_interval_s

        self._train_event = threading.Event()
        self.model_lock = threading.RLock()

        self._latest_loss: Optional[dict] = None
        self.monitor = LearningMonitor()
        self._loss_lock = threading.Lock()

        self._running = True
        self._train_count = 0

    # ──────────────────────────────────────────────────────────────────────────

    def request_train(self) -> None:
        """Sinaliza ao trainer para executar um step imediatamente."""
        self._train_event.set()

    def get_latest_loss(self) -> Optional[dict]:
        """Retorna o último dict de métricas de loss (thread-safe)."""
        with self._loss_lock:
            return self._latest_loss

    def stop(self) -> None:
        """Para o trainer graciosamente."""
        self._running = False
        self._train_event.set()
        self.join(timeout=10.0)
        logger.info(f"ContinuousTrainer encerrado | {self._train_count} treinos executados")

    # ──────────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        logger.info("ContinuousTrainer iniciado.")
        while self._running:
            # Aguarda sinal explícito ou timeout para poll periódico
            self._train_event.wait(timeout=self.train_interval_s)
            self._train_event.clear()

            if not self._running:
                break

            # Verifica buffer mínimo
            try:
                buf_size = self.learner.replay_buffer.size()
            except Exception:
                buf_size = 0

            if buf_size < self.min_buffer:
                continue

            try:
                with self.model_lock:
                    loss_dict = self.learner.train_step(self.batch_size)

                if loss_dict is not None:
                    with self._loss_lock:
                        self._latest_loss = loss_dict
                    self._train_count += 1

                    if self._train_count % 100 == 0:
                        wm   = loss_dict.get("wm/loss", loss_dict.get("loss_total", 0))
                        rec  = loss_dict.get("wm/rec",  0)
                        kl   = loss_dict.get("wm/kl",   0)
                        rew  = loss_dict.get("wm/reward", 0)
                        actr = loss_dict.get("ac/actor_loss", 0)
                        crit = loss_dict.get("ac/critic_loss", 0)
                        entr = loss_dict.get("ac/entropy", 0)
                        ret_ = loss_dict.get("ac/returns_mean", 0)
                        logger.info(
                            f"[TREINO] step={self._train_count} "
                            f"| buf={buf_size} "
                            f"| WM loss={wm:.4f} "
                            f"(rec={rec:.4f} kl={kl:.4f} rew={rew:.4f}) "
                            f"| Policy: actor={actr:.4f} critic={crit:.4f} "
                            f"entropia={entr:.4f} retorno_medio={ret_:.4f}"
                        )

                    # Painel de VEREDITO de aprendizado (tendências) a cada 100.
                    if self.monitor is not None:
                        self.monitor.record_metrics(loss_dict)
                        if self._train_count % 100 == 0:
                            self.monitor.log_panel(self._train_count)
                    if self._train_count % 500 == 0:
                        logger.info(
                            f"[TREINO][CHECKPOINT] step={self._train_count} — "
                            f"avaliando política para salvar checkpoint..."
                        )
            except Exception as exc:
                logger.error(f"ContinuousTrainer.train_step() erro: {exc}", exc_info=True)
                time.sleep(0.1)
