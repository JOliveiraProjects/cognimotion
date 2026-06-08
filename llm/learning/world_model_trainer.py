"""
learning/world_model_trainer.py
=================================
WorldModelTrainer — thread daemon que treina RSSM + ActorCritic continuamente.

Padrão idêntico ao ContinuousTrainer existente, mas especializado para DreamerV3:
  - Amostra do SequenceBuffer em intervalos regulares
  - Delega para DreamerTrainer.train_world_model() e update_actor_critic()
  - Publica checkpoints via PolicyRegistry

Pode rodar em paralelo com o ContinuousTrainer (VAE) sem conflito,
pois opera sobre parâmetros distintos (RSSM vs PoseEncoder/MotionLatentSpace).
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from learning.learning_monitor import LearningMonitor

logger = logging.getLogger(__name__)


class WorldModelTrainerThread:
    """
    Thread daemon para treinamento contínuo do World Model (RSSM) + ActorCritic.

    Uso:
        trainer_thread = WorldModelTrainerThread(
            dreamer_trainer=DreamerTrainer(...),
            sequence_buffer=SequenceBuffer(...),
            config=config,
        )
        trainer_thread.start()
        # ... servidor rodando ...
        trainer_thread.stop()
    """

    def __init__(
        self,
        dreamer_trainer,          # world_model.DreamerTrainer
        sequence_buffer,          # runtime.SequenceBuffer
        config,                   # MotionIntelligenceConfig
        train_interval_s: float = 5.0,
        ac_ratio:         int   = 4,    # treina AC a cada N steps de WM
        pipeline_stats=None,            # runtime.PipelineStats (opcional)
    ) -> None:
        self.trainer          = dreamer_trainer
        self.seq_buffer       = sequence_buffer
        self.config           = config
        self.train_interval_s = train_interval_s
        self.ac_ratio         = ac_ratio
        self.pipeline_stats   = pipeline_stats

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._wm_steps  = 0
        self._ac_steps  = 0

        # Telemetria de aprendizado legível (veredito SIM/NÃO + tendências).
        self.monitor = LearningMonitor()

        logger.info(
            f"WorldModelTrainerThread | interval={train_interval_s}s "
            f"| ac_ratio=1/{ac_ratio}"
        )

    # ──────────────────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="WorldModelTrainer",
            daemon=True,
        )
        self._thread.start()
        logger.info("WorldModelTrainerThread | iniciada")

    def stop(self, timeout: float = 10.0) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        logger.info("WorldModelTrainerThread | encerrada")

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ──────────────────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        # RETOMA o treino acumulativo da categoria (se houver checkpoint salvo).
        # Verifica também a compatibilidade do skeleton. Sem isso, cada sessão
        # recomeçava do zero. Agora o treino fica mais completo a cada sessão.
        reg = getattr(self.trainer, "policy_registry", None)
        if reg is not None and hasattr(reg, "resume_category"):
            try:
                profile = getattr(self.trainer, "skeleton_profile", None)
                v = reg.resume_category(self.trainer, current_profile=profile)
                if v == -1:
                    logger.error(
                        "[TREINO] Não retomado por incompatibilidade de skeleton. "
                        "Veja a mensagem acima. Treinando do zero NÃO é seguro aqui "
                        "— corrija o skeleton ou escolha outra categoria."
                    )
            except Exception as exc:
                logger.warning(f"[TREINO] resume falhou (seguindo do zero): {exc}")

        logger.info(
            "\n" + "=" * 60 + "\n"
            "  TREINO DO WORLD MODEL (DreamerV3) INICIADO\n"
            f"  batch_size={self.config.world_model.batch_size} "
            f"seq_len={self.config.world_model.seq_len} "
            f"intervalo={self.train_interval_s}s\n"
            "  Aguardando dados do Unreal (modo Observing)...\n"
            + "=" * 60
        )
        last_train = time.time()

        while not self._stop_event.is_set():
            now = time.time()
            if (now - last_train) < self.train_interval_s:
                time.sleep(0.1)
                continue

            last_train = now

            # Verifica se há dados suficientes
            if not self.seq_buffer.ready_sequence(
                self.config.world_model.batch_size
            ):
                # Diagnóstico claro: o treino NÃO está rodando ainda, e por quê.
                self._waiting_logs = getattr(self, "_waiting_logs", 0) + 1
                if self._waiting_logs % 6 == 0:  # ~a cada 6s
                    have = self.seq_buffer.total_transitions()
                    need = self.config.world_model.batch_size
                    if self.pipeline_stats is not None:
                        self.pipeline_stats.set(
                            training_active=False,
                            buffer_transitions=have,
                            waiting_reason=f"sem dados ({have}/{need} transições) — mova o líder",
                        )
                    logger.warning(
                        f"[TREINO AGUARDANDO] sem dados suficientes para treinar: "
                        f"tenho {have} transições, preciso de ~{need}. "
                        f"O agente está só COLETANDO — mova mais o líder em "
                        f"Observing para acumular experiência."
                    )
                time.sleep(1.0)
                continue

            # Treina World Model
            try:
                wm_metrics = self.trainer.train_world_model()
                if wm_metrics:
                    self._wm_steps += 1
                    # Alimenta o monitor com as métricas REAIS do batch.
                    self.monitor.record_metrics(wm_metrics)
                    # Reporta ao pipeline transparente (painel ao vivo).
                    if self.pipeline_stats is not None:
                        # Considera o modelo pronto p/ gerar poses após treino
                        # suficiente E com pose loss saudável (< 5.0 na escala
                        # normalizada). Antes disso, o NPC copia o líder.
                        pose_loss = float(wm_metrics.get("wm/pose", 999))
                        ready = self._wm_steps >= 200 and pose_loss < 5.0
                        self.pipeline_stats.set(
                            training_active=True,
                            wm_steps=self._wm_steps,
                            last_wm_loss=float(wm_metrics.get("wm/loss", 0)),
                            last_pose_loss=pose_loss,
                            pose_ready=ready,
                            waiting_reason="",
                        )
                    # Loga a loss do world model a cada batch (o que faltava).
                    logger.info(
                        f"[WM] step={self._wm_steps} "
                        f"| loss={wm_metrics.get('wm/loss', 0):.4f} "
                        f"(rec={wm_metrics.get('wm/rec', 0):.4f} "
                        f"kl={wm_metrics.get('wm/kl', 0):.4f} "
                        f"pose={wm_metrics.get('wm/pose', 0):.4f})"
                    )
            except Exception as exc:
                logger.error(f"WorldModelTrainer | erro WM step: {exc}", exc_info=True)
                time.sleep(1.0)
                continue

            # Treina Actor-Critic (a cada ac_ratio steps de WM)
            if self._wm_steps % self.ac_ratio == 0:
                try:
                    ac_metrics = self.trainer.update_actor_critic()
                    if ac_metrics:
                        self._ac_steps += 1
                        # Junta métricas de AC ao monitor (entropia, retorno).
                        self.monitor.record_metrics(ac_metrics)
                        logger.info(
                            f"[AC] step={self._ac_steps} "
                            f"| actor={ac_metrics.get('ac/actor_loss', 0):.4f} "
                            f"critic={ac_metrics.get('ac/critic_loss', 0):.4f} "
                            f"entropia={ac_metrics.get('ac/entropy', 0):.4f} "
                            f"retorno={ac_metrics.get('ac/returns_mean', 0):.4f}"
                        )
                except Exception as exc:
                    logger.error(f"WorldModelTrainer | erro AC step: {exc}", exc_info=True)

            # Painel de VEREDITO de aprendizado a cada 20 steps de WM.
            if self._wm_steps % 20 == 0:
                self.monitor.set_context(
                    training=self.monitor.current_training or "—",
                    mode="Treinando (WM+AC)",
                )
                self.monitor.log_panel(self._wm_steps)

            # Publica checkpoint
            try:
                self.trainer.maybe_publish(
                    min_interval_steps=self.config.dreamer.publish_interval_steps
                )
            except Exception as exc:
                logger.debug(f"WorldModelTrainer | publish falhou: {exc}")

            # AUTOSAVE periódico: salva o treino acumulativo a cada ~30s,
            # independente das condições de publish. Garante que parar o servidor
            # nunca perca o progresso — o resume na próxima sessão funciona mesmo
            # que o treino ainda não tenha "publicado" um checkpoint formal.
            now2 = time.time()
            if now2 - getattr(self, "_last_autosave", 0.0) >= 30.0:
                self._last_autosave = now2
                reg = getattr(self.trainer, "policy_registry", None)
                if reg is not None and hasattr(reg, "autosave"):
                    try:
                        state = self.trainer.build_checkpoint_state()
                        reg.autosave(state)
                    except Exception as exc:
                        logger.debug(f"WorldModelTrainer | autosave falhou: {exc}")

            # PAUSA ENTRE STEPS: o treino segura o model_lock durante todo o batch
            # (~2s). Sem ceder, a inferência do NPC fica bloqueada esperando o
            # lock → picos de latência de 500ms. Esta pausa curta libera o lock e
            # a GPU entre os steps, dando à inferência a chance de responder rápido.
            # Configurável: dreamer.step_pause_s (padrão 0.05s).
            time.sleep(float(getattr(self.config.dreamer, "step_pause_s", 0.05)))

        logger.info(
            f"WorldModelTrainerThread | encerrada | "
            f"wm_steps={self._wm_steps} | ac_steps={self._ac_steps}"
        )

    # ──────────────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        return {
            "wm_steps":   self._wm_steps,
            "ac_steps":   self._ac_steps,
            "is_alive":   self.is_alive(),
            "buffer_eps": self.seq_buffer.episode_count(),
            "buffer_transitions": self.seq_buffer.total_transitions(),
        }
