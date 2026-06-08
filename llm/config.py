from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Existing configs (kept verbatim — DO NOT modify names/fields)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EncoderConfig:
    pose_dim: int = 256
    trajectory_dim: int = 128
    embedding_dim: int = 256
    n_heads: int = 8
    n_layers: int = 4
    dropout: float = 0.1
    max_seq_len: int = 64
    bone_count: int = 9
    bone_dim: int = 10
    trajectory_samples: int = 6
    sample_dim: int = 10


@dataclass
class MemoryConfig:
    bank_capacity: int = 50000
    replay_capacity: int = 100000
    faiss_index_type: str = "Flat"
    similarity_threshold: float = 0.85
    diversity_weight: float = 0.3
    recency_weight: float = 0.4
    confidence_weight: float = 0.3
    priority_alpha: float = 0.6
    priority_beta: float = 0.4


@dataclass
class LearningConfig:
    learning_rate: float = 3e-4
    batch_size: int = 64
    gradient_clip: float = 1.0
    warmup_steps: int = 1000
    style_loss_weight: float = 0.2
    trajectory_loss_weight: float = 0.4
    imitation_loss_weight: float = 0.4
    entropy_weight: float = 0.01
    min_buffer_for_train: int = 512


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 9000
    max_clients: int = 32          # aumentado para multiplayer
    recv_buffer_size: int = 131072
    send_buffer_size: int = 131072
    heartbeat_interval: float = 1.0
    max_latency_ms: float = 80.0
    worker_threads: int = 4


# ──────────────────────────────────────────────────────────────────────────────
# New configs — adicionados para expansão multi-processo / LLM / multiplayer
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LLMConfig:
    """Configuração do processo LLM (GPT-2 ou melhor)."""
    model_name: str = "gpt2"                 # "gpt2", "gpt2-medium", "distilgpt2"
    device: str = "cpu"
    max_new_tokens: int = 32
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.92
    cache_dir: str = "models/llm"
    # Intervalo mínimo entre inferências LLM por NPC (segundos)
    inference_interval_s: float = 2.0
    # Tamanho do pool de contexto de personalidade (tokens)
    context_window: int = 128
    # Habilita geração de texto para logs / depuração
    verbose_generation: bool = False


@dataclass
class MultiProcessConfig:
    """Configuração dos processos paralelos."""
    # Número de processos de inferência paralelos (um pool de workers)
    n_inference_workers: int = 4
    # Tamanho das filas IPC entre main e workers
    ipc_queue_maxsize: int = 256
    # Processo de treinamento contínuo dedicado
    enable_training_process: bool = True
    # Processo de dreaming dedicado
    enable_dream_process: bool = True
    # Processo LLM dedicado
    enable_llm_process: bool = True
    # Intervalo de sync de pesos entre worker processes (segundos)
    weight_sync_interval_s: float = 30.0
    # Diretório de checkpoints compartilhados entre processos
    shared_checkpoint_dir: str = "runs/shared"
    # Timeout de shutdown gracioso (segundos)
    shutdown_timeout_s: float = 10.0


@dataclass
class PopulationConfig:
    """Configuração de Population-Based Training (PBT)."""
    n_members: int = 4
    eval_interval_steps: int = 10_000
    exploit_ratio: float = 0.25
    perturb_factor: float = 0.2
    enable_pbt: bool = True


@dataclass
class EpisodicMemoryConfig:
    """Configuração de memória episódica por NPC."""
    max_episodes: int = 200
    max_keyframes_per_episode: int = 64
    similarity_threshold: float = 0.75
    vector_store_dim: int = 256    # = embedding_dim


@dataclass
class SemanticMemoryConfig:
    """Configuração de memória semântica compartilhada."""
    max_facts: int = 10_000
    decay_rate: float = 0.001
    vector_store_dim: int = 256


@dataclass
class UnifiedBufferConfig:
    """Configuração do UnifiedReplayBuffer com PER."""
    capacity: int = 200_000
    alpha: float = 0.6
    beta_start: float = 0.4
    beta_end: float = 1.0
    total_steps: int = 1_000_000
    use_per: bool = True
    action_dim: int = 9            # = len(ECognitiveMotionStyle)
    demo_priority_boost: float = 2.0
    # Seq len para sample_sequence (usado pelo trainer recorrente)
    seq_len: int = 16


@dataclass
class ActorCriticConfig:
    """Configuração do ActorCritic (GAE + imitation loss)."""
    action_dim: int = 9            # = len(ECognitiveMotionStyle)
    skill_embed_dim: int = 64
    discount: float = 0.99
    lambda_gae: float = 0.95
    entropy_weight: float = 0.01
    grad_clip_norm: float = 1.0
    imagination_horizon: int = 16
    # Dimensão de embedding quando rssm=None (usa OnlineImitationLearner latent)
    embedding_dim: int = 256
    # Alpha de mistura BC loss em update_with_demo
    imitation_alpha: float = 0.5
    learning_rate: float = 1e-4
    enabled: bool = True


@dataclass
class BehaviorConfig:
    """Configuração dos controladores de comportamento por NPC."""
    # ExplorationController default mode: auto | deterministic | exploratory | epsilon_greedy
    default_exploration_mode: str = "auto"
    epsilon: float = 0.1
    # GoalController: máximo de steps por objetivo (None = sem limite)
    goal_max_steps: int = 1000
    # Usa ActorCritic para selecionar ação em vez de argmax do embedding
    use_actor_critic: bool = True
    # Ativa tree_planner (reservado para expansão futura)
    use_tree_planner: bool = False
    # Ativa skill_discovery
    use_skill_discovery: bool = False
    # Ativa LLM como planner hierárquico
    use_llm: bool = True



@dataclass
class WorldModelConfig:
    """Configuração do RSSM / World Model (DreamerV3)."""
    rssm_hidden_dim:   int   = 512
    num_bones:         int   = 89   # bones reais enviados pelo UE5 (PoseDecoder)
    rssm_num_categories: int = 32
    rssm_category_dim: int   = 32
    rssm_free_nats:    float = 3.0
    rssm_kl_balance:   float = 0.8
    rssm_unimix:       float = 0.01
    use_block_gru:     bool  = True
    n_blocks:          int   = 8
    # Training
    learning_rate:     float = 1e-4
    batch_size:        int   = 32
    seq_len:           int   = 16
    recon_weight:      float = 1.0
    kl_weight:         float = 1.0
    reward_weight:     float = 1.0
    done_weight:       float = 0.5
    grad_clip_norm:    float = 100.0
    overshooting_steps: int  = 2
    sequence_buffer_capacity: int = 200_000
    train_interval_s:  float = 1.5   # treina WM mais rápido (era 5s)
    warmup_wm_steps:   int   = 100
    imagination_batch_size: int = 64
    # Autonomous inference
    autonomous_mode:   bool  = True


@dataclass
class DreamerConfig:
    """Configuração do processo Dreamer dedicado."""
    enable_dreamer_process: bool  = True
    dreamer_device:         str   = "cpu"
    publish_interval_steps: int   = 50   # 1º checkpoint em ~75s (era 500=42min)
    ipc_queue_maxsize:      int   = 512
    step_pause_s:           float = 0.05  # pausa entre steps de treino — cede o
                                          # model_lock p/ a inferência (evita picos
                                          # de latência). Suba se a latência do NPC
                                          # ainda tiver picos; abaixe p/ treinar mais rápido.


@dataclass
class NPCSessionConfig:
    """Configuração do NPCSessionManager."""
    max_sessions:   int   = 256
    timeout_s:      float = 120.0


@dataclass
class ProductionConfig:
    """Configuração de produção (checkpoints, hot-reload)."""
    checkpoint_dir:  str  = "checkpoints/dreamer"
    keep_last_n:     int  = 10
    compress_old:    bool = True
    max_total_gb:    float = 50.0


@dataclass
class DatasetConfig:
    """Configuração do dataset de interações (weapon, ball, threat, vehicle, etc.)."""
    config_path:     str   = "config/dataset_config.yaml"
    scale:           float = 1.0
    seed:            int   = 42
    obs_dim:         int   = 256
    action_dim:      int   = 9   # deve coincidir com ActorCriticConfig.action_dim
    enable_weapons:  bool  = True
    enable_ball:     bool  = True
    enable_threats:  bool  = True
    enable_vehicles: bool  = True
    enable_mounts:   bool  = True
    enable_traffic:  bool  = True
    load_on_start:   bool  = False  # DESATIVADO: as 24 ações sintéticas colapsam
    # para idle no espaço de 9 ações e poluem a política com viés de "parado".
    # Para imitação pura do líder, o NPC aprende só das demonstrações reais.
    # Reative (True) apenas se quiser bootstrap de dinâmica genérica do world model.


# ──────────────────────────────────────────────────────────────────────────────
# Root config — agora inclui todos os sub-configs
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MotionIntelligenceConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    multiprocess: MultiProcessConfig = field(default_factory=MultiProcessConfig)
    population: PopulationConfig = field(default_factory=PopulationConfig)
    episodic_memory: EpisodicMemoryConfig = field(default_factory=EpisodicMemoryConfig)
    semantic_memory: SemanticMemoryConfig = field(default_factory=SemanticMemoryConfig)
    unified_buffer: UnifiedBufferConfig = field(default_factory=UnifiedBufferConfig)
    actor_critic: ActorCriticConfig = field(default_factory=ActorCriticConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    dreamer: DreamerConfig = field(default_factory=DreamerConfig)
    npc_session: NPCSessionConfig = field(default_factory=NPCSessionConfig)
    production: ProductionConfig = field(default_factory=ProductionConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    device: str = "cpu"
    checkpoint_dir: str = "checkpoints/motion"
    log_level: str = "INFO"
    enable_faiss_gpu: bool = False


DEFAULT_CONFIG = MotionIntelligenceConfig()
