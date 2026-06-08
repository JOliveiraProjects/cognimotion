"""
llm_interface.py
================
Interface LLM para geração de intenção comportamental de NPCs.

Arquitetura:
  - Roda em processo dedicado (multiprocessing.Process)
  - Usa GPT-2 (ou variante) via HuggingFace transformers
  - Recebe: estado do NPC (blackboard dict) via Queue
  - Retorna: int de ECognitiveMotionStyle + tokens de contexto

Nota: o modelo é baixado automaticamente na primeira execução via
      transformers.AutoModelForCausalLM.from_pretrained(model_name).
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Mapeamento de tokens gerados pelo LLM → ECognitiveMotionStyle (int)
# Esses tokens são palavras-chave que o LLM aprende a emitir.
_STYLE_TOKEN_MAP: Dict[str, int] = {
    "neutral":    0,
    "aggressive": 1,
    "relaxed":    2,
    "injured":    3,
    "fatigued":   4,
    "stealth":    5,
    "military":   6,
    "civilian":   7,
    "criminal":   8,
}

# Prompt template que contextualiza o NPC para o LLM.
_PROMPT_TEMPLATE = (
    "NPC status: health={health:.0f} stamina={stamina:.0f} "
    "alertness={alertness:.0f} fear={fear:.0f} aggression={aggression:.0f} "
    "state={state}. Movement style:"
)


@dataclass
class LLMRequest:
    session_id: str
    health: float
    stamina: float
    alertness: float
    fear_level: float
    aggression_level: float
    current_state: int      # ECognitiveNPCState int value


@dataclass
class LLMResponse:
    session_id:     str
    motion_style:   int        # ECognitiveMotionStyle int value
    selected_style: int        # alias for motion_style — used by service llm_response_loop
    generated_text: str
    confidence:     float
    latency_ms:     float

    def __post_init__(self) -> None:
        if self.selected_style == 0 and self.motion_style != 0:
            self.selected_style = self.motion_style
        elif self.motion_style == 0 and self.selected_style != 0:
            self.motion_style = self.selected_style


class LLMInterface:
    """
    Wrapper em torno de um modelo GPT-2 (HuggingFace).

    Uso standalone:
        interface = LLMInterface("gpt2", device="cpu")
        response = interface.infer(request)

    Em produção é instanciado dentro de LLMWorkerProcess e chamado
    via multiprocessing.Queue.
    """

    _NPC_STATE_NAMES = [
        "idle", "casual", "alert", "investigate", "combat",
        "stealth", "flee", "hide", "cover", "surrender",
        "healing", "incapacitated", "dead", "react", "contextual",
        "mounted", "driving",
    ]

    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "cpu",
        cache_dir: str = "models/llm",
        max_new_tokens: int = 32,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.92,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

        self._model = None
        self._tokenizer = None
        self._cache_dir = cache_dir
        self._loaded = False
        self._stub_mode = False  # True quando modelo não disponível — retorna fallback neutro

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Baixa/carrega o modelo. Bloqueia até concluir.
        
        Tenta primeiro modo offline (cache local). Se o cache não existir,
        tenta download com timeout configurável. Se falhar, usa modo stub
        para não travar o servidor inteiro.
        """
        if self._loaded:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers não instalado. Execute: pip install transformers"
            ) from exc

        os.makedirs(self._cache_dir, exist_ok=True)
        logger.info(f"LLMInterface | carregando modelo '{self.model_name}' em {self.device}...")
        t0 = time.perf_counter()

        # Tenta primeiro offline (cache local) para não depender de rede no startup.
        # Se o modelo não estiver em cache, tenta download. Se falhar por timeout
        # ou firewall, entra em modo stub — o servidor continua funcionando sem LLM.
        load_kwargs = {"cache_dir": self._cache_dir}
        offline_ok = self._try_load_offline(AutoTokenizer, AutoModelForCausalLM, load_kwargs)

        if not offline_ok:
            logger.warning(
                f"LLMInterface | modelo '{self.model_name}' não encontrado em cache. "
                f"Tentando download..."
            )
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, **load_kwargs)
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, **load_kwargs)
            except Exception as exc:
                logger.error(
                    f"LLMInterface | falha ao carregar '{self.model_name}': {exc}\n"
                    f"  → Modo stub ativo. Execute com rede disponível para baixar o modelo,\n"
                    f"    ou pré-baixe com: python -c \"from transformers import "
                    f"AutoTokenizer; AutoTokenizer.from_pretrained('{self.model_name}', "
                    f"cache_dir='{self._cache_dir}')\""
                )
                self._stub_mode = True
                self._loaded = True
                return

        if self._tokenizer is not None and self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        if self._model is not None:
            self._model.eval()
            self._model.to(self.device)

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(f"LLMInterface | modelo pronto em {elapsed:.0f}ms")
        self._loaded = True

    def _try_load_offline(self, TokenizerClass, ModelClass, kwargs: dict) -> bool:
        """Tenta carregar exclusivamente do cache local. Retorna True se bem-sucedido."""
        try:
            self._tokenizer = TokenizerClass.from_pretrained(
                self.model_name, local_files_only=True, **kwargs)
            self._model = ModelClass.from_pretrained(
                self.model_name, local_files_only=True, **kwargs)
            logger.info("LLMInterface | modelo carregado do cache local (offline).")
            return True
        except Exception:
            self._tokenizer = None
            self._model = None
            return False

    def infer(self, request: LLMRequest) -> LLMResponse:
        """
        Dado um LLMRequest (estado do NPC), retorna um LLMResponse com
        o motion_style sugerido pelo LLM.
        """
        if not self._loaded:
            self.load()

        # Modo stub: modelo não disponível (sem rede, sem cache).
        # Retorna resposta neutra para não travar o pipeline de inferência.
        if self._stub_mode:
            return LLMResponse(
                session_id=request.session_id,
                motion_style=0,
                selected_style=0,
                generated_text="[LLM indisponível — modo stub]",
                confidence=0.0,
                latency_ms=0.0,
            )

        t0 = time.perf_counter()

        state_name = self._NPC_STATE_NAMES[
            min(request.current_state, len(self._NPC_STATE_NAMES) - 1)
        ]
        prompt = _PROMPT_TEMPLATE.format(
            health=request.health,
            stamina=request.stamina,
            alertness=request.alertness,
            fear=request.fear_level,
            aggression=request.aggression_level,
            state=state_name,
        )

        generated = self._generate(prompt)
        style_int, confidence = self._parse_style(generated)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return LLMResponse(
            session_id=request.session_id,
            motion_style=style_int,
            selected_style=style_int,
            generated_text=generated,
            confidence=confidence,
            latency_ms=latency_ms,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _generate(self, prompt: str) -> str:
        import torch

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        # Decodifica apenas os tokens novos (não o prompt)
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).lower().strip()

    @staticmethod
    def _parse_style(text: str):
        """
        Extrai o estilo de movimento dominante do texto gerado.
        Retorna (style_int, confidence).
        """
        best_style = 0          # Neutral
        best_count = 0
        total_hits = 0

        for token, style_id in _STYLE_TOKEN_MAP.items():
            count = text.count(token)
            if count > best_count:
                best_count = count
                best_style = style_id
            total_hits += count

        confidence = min(1.0, best_count / max(total_hits, 1))
        if total_hits == 0:
            # Nenhum token reconhecido — fallback heurístico simples
            confidence = 0.3
            best_style = 0

        return best_style, confidence


# ──────────────────────────────────────────────────────────────────────────────
# Processo dedicado ao LLM
# ──────────────────────────────────────────────────────────────────────────────

def llm_worker_process(
    request_queue,   # multiprocessing.Queue[LLMRequest | None]
    response_queue,  # multiprocessing.Queue[LLMResponse]
    model_name: str = "gpt2",
    device: str = "cpu",
    cache_dir: str = "models/llm",
) -> None:
    """
    Entry-point do processo LLM.
    Loop: lê LLMRequest → infere → escreve LLMResponse.
    Termina quando recebe None na fila.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | LLM | %(levelname)-8s | %(message)s",
    )
    log = logging.getLogger("llm_worker")
    log.info(f"LLM Worker iniciado | modelo={model_name} | device={device}")

    interface = LLMInterface(
        model_name=model_name,
        device=device,
        cache_dir=cache_dir,
    )
    # load() captura erros de rede/timeout internamente (modo stub).
    # O processo nunca crasha por falta de conexão com HuggingFace.
    interface.load()

    while True:
        # Timeout de 1s para encerrar graciosamente sem depender de None na fila.
        try:
            req = request_queue.get(timeout=1.0)
        except Exception:
            continue
        if req is None:
            log.info("LLM Worker encerrando.")
            break
        try:
            resp = interface.infer(req)
            response_queue.put(resp)
        except Exception as exc:
            log.error(f"Erro na inferência LLM: {exc}")
            response_queue.put(LLMResponse(
                session_id=req.session_id,
                motion_style=0,
                selected_style=0,
                generated_text="",
                confidence=0.0,
                latency_ms=0.0,
            ))
