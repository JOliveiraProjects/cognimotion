"""
main.py — CognitiveMotionIntelligence DreamerV3 Server
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import multiprocessing as mp
import os
import signal
import sys

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)


def _setup_logging(level: str = "INFO") -> None:
    fmt = "%(asctime)s | %(processName)-20s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    for noisy in ("transformers", "torch", "faiss", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CognitiveMotionIntelligence DreamerV3 server")
    p.add_argument("--host",           default="0.0.0.0")
    p.add_argument("--port",    type=int, default=9000)
    p.add_argument("--device",         default="cpu",
                   choices=["cpu", "cuda", "mps"])
    p.add_argument("--workers", type=int, default=2,
                   help="Número de InferenceWorker processes")
    p.add_argument("--max-clients", type=int, default=64)
    p.add_argument("--llm",            default="gpt2",
                   help="Modelo LLM para hints de estilo")
    p.add_argument("--llm-device",     default="cpu")
    p.add_argument("--no-llm",  action="store_true",
                   help="Desabilita processo LLM")
    p.add_argument("--no-dreamer", action="store_true",
                   help="Desabilita processo Dreamer dedicado")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--log-level",      default="INFO")
    p.add_argument("--sim", action="store_true",
                   help="Treina localmente no MotionEnv 2D (sem UE5)")
    p.add_argument("--sim-episodes", type=int, default=200)
    return p.parse_args()


def _build_config(args: argparse.Namespace):
    from config import (
        MotionIntelligenceConfig, ServerConfig, LLMConfig,
        MultiProcessConfig, PopulationConfig, DreamerConfig, ProductionConfig,
    )
    return MotionIntelligenceConfig(
        device=args.device,
        server=ServerConfig(host=args.host, port=args.port, max_clients=args.max_clients),
        llm=LLMConfig(model_name=args.llm, device=args.llm_device, cache_dir="models/llm"),
        multiprocess=MultiProcessConfig(
            n_inference_workers=args.workers,
            enable_llm_process=not args.no_llm,
            shared_checkpoint_dir=args.checkpoint_dir,
        ),
        population=PopulationConfig(n_members=min(args.workers, 4), enable_pbt=True),
        dreamer=DreamerConfig(
            enable_dreamer_process=not args.no_dreamer,
            dreamer_device=args.device,
        ),
        production=ProductionConfig(checkpoint_dir=args.checkpoint_dir),
        checkpoint_dir=args.checkpoint_dir,
    )


def run_sim(args: argparse.Namespace) -> None:
    from envs.env_runner import EnvRunner
    log = logging.getLogger("sim")
    log.info(f"Modo simulação | episodes={args.sim_episodes} | device={args.device}")
    runner = EnvRunner(device=args.device, seed=42)
    runner.run(max_episodes=args.sim_episodes)


def main() -> None:
    args = _parse_args()
    _setup_logging(args.log_level)
    log = logging.getLogger("main")

    for d in [args.checkpoint_dir, "models/llm", "checkpoints/dreamer"]:
        os.makedirs(d, exist_ok=True)

    if args.sim:
        run_sim(args)
        return

    log.info("=" * 60)
    log.info("CognitiveMotionIntelligence DreamerV3 — iniciando")
    log.info(f"  {args.host}:{args.port} | device={args.device}")
    log.info(f"  workers={args.workers} | dreamer={'off' if args.no_dreamer else 'on'}")
    log.info("=" * 60)

    from runtime.motion_inference_service import MotionInferenceService
    config  = _build_config(args)
    service = MotionInferenceService(config)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    stop_event = asyncio.Event()

    def _on_signal(sig, _frame):
        log.info(f"Sinal {sig} — encerrando")
        loop.call_soon_threadsafe(stop_event.set)

    signal.signal(signal.SIGINT,  _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    async def _run_server():
        start_task = asyncio.create_task(service.start())

        try:
            await stop_event.wait()
        finally:
            await service.stop()

            if not start_task.done():
                start_task.cancel()
                try:
                    await start_task
                except asyncio.CancelledError:
                    pass

    try:
        loop.run_until_complete(_run_server())
    except (KeyboardInterrupt, SystemExit):
        pass
    except Exception as exc:
        log.error(f"Erro fatal: {exc}", exc_info=True)
        sys.exit(1)
    finally:
        loop.close()
        log.info("Servidor encerrado.")


if __name__ == "__main__":
    main()
