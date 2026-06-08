# Guia Completo para Executar o Projeto
## 1. Pré-requisitos
### 1.1 Hardware Recomendado
- Mínimo: CPU com 8GB RAM (modo CPU)
- Recomendado: GPU NVIDIA com 8GB+ VRAM (para treinamento mais rápido e LLM)

### 1.2 Software Necessário
- Python 3.10 ou 3.11
- CUDA 11.8+ (se usar GPU)
- Git

## 2. Instalação Passo a Passo
### 2.1 Clone o Repositório

```bash
git clone https://github.com/seu-usuario/cognitive_agent.git
cd cognitive_agent
```
### 2.2 Crie um Ambiente Virtual (Recomendado)

```bash
# Linux/Mac
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 2.3 Instale as Dependências
```bash
pip install --upgrade pip
python.exe -m pip install --upgrade pip 
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # para GPU
# ou para CPU:
# pip install torch torchvision

pip install -r requirements.txt
```
### 2.4 Verifique a Instalação
```bash
python -c "import torch; print(torch.cuda.is_available())"  # Deve retornar True se GPU disponível
```

## 4. Configuração do Ambiente
### 4.1 Ajuste o Arquivo core/config.py (Opcional)
Antes da primeira execução, você pode ajustar alguns parâmetros:
```bash
# Principais parâmetros para ajustar
num_envs = 4                    # Número de ambientes paralelos (aumenta para mais amostras)
warmup_steps = 5000             # Passos iniciais sem treinamento (reduza para testes rápidos)
use_llm = False                 # Desative LLM se não quiser usar GPT-2
use_actor_critic = True         # Mantenha True para melhor eficiência
render_mode = "rgb_array"       # Use "human" para ver o agente (mais lento)
```

## 5. Execução
### 5.1 Teste Rápido (Sem LLM, Sem Render)
```bash
python main.py --env MiniGrid-Empty-5x5-v0
```
Isso executa o agente no ambiente MiniGrid vazio, sem janela gráfica. O agente começará a explorar e treinar.

### 5.2 Com Visualização
```bash
python main.py --env MiniGrid-Empty-5x5-v0 --render
```
Isso abrirá uma janela mostrando o agente interagindo com o ambiente.

### 5.3 Ambiente Mais Complexo
```bash
python main.py --env MiniGrid-DoorKey-8x8-v0 --render
```
### 5.4 Com LLM Ativado (Requer Download do GPT-2)
```bash
# Primeira execução baixará o modelo GPT-2 (cerca de 500MB)
python main.py --env MiniGrid-Empty-5x5-v0 --render
```
### 5.5 Retomar de Checkpoint
```bash
python main.py --env MiniGrid-Empty-5x5-v0 --load checkpoint_step_10000.pt --render
```

## 6. Monitoramento e Logs
### 6.1 Logs em Tempo Real
O sistema exibe logs no console:
```text
Step 100 | Loss: 2.3456 | Rec: 1.2345 | KL: 0.1234
Step 200 | AC | Actor: -0.45 | Critic: 0.89 | Entropy: 1.23
Step 300 | Avg reward: 2.34 | Nodes: 45 | Skills: 3 | Goal: encontrar_saida
Env 0 | Episode reward: 5.67 | Intrinsic contrib: 0.23
```

### 6.2 Arquivos de Log
Os logs são salvos em logs/cognitive_agent.log:
```bash
tail -f logs/cognitive_agent.log
```

### 6.3 Checkpoints
A cada 10.000 passos, o sistema salva um checkpoint:
```text
checkpoint_step_10000.pt
checkpoint_step_20000.pt
```


## 7. Solução de Problemas Comuns
### 7.1 Erro: "No module named 'minigrid'"

```bash
pip install minigrid
```

### 7.2 Erro: "CUDA out of memory"
Reduza o número de ambientes paralelos:

```python
# Em config.py
num_envs = 1  # em vez de 4
```

### 7.3 Erro: "Failed to load FAISS"

```bash
pip install faiss-cpu  # ou faiss-gpu se tiver GPU
```

### 7.4 O Agente Não Aprende (Recompensa Zero)
- Verifique se warmup_steps não é muito alto
- Reduza num_envs para 1 e observe o comportamento
- Ative --render para ver o que o agente está fazendo

### 7.5 LLM Muito Lento
Desative o LLM em config.py:

```python
use_llm = False
```

## 8. Teste dos Módulos Individualmente
### 8.1 Testar Encoder/Decoder

```python
python -c "
from perception.encoder import CNNEncoder
from core.config import Config
config = Config()
encoder = CNNEncoder(config.obs_shape, config.latent_dim)
print('Encoder OK')
"
```

### 8.2 Testar RSSM

```python
python -c "
from world_model.rssm import RSSM
from core.config import Config
config = Config()
rssm = RSSM(config.latent_dim, config.action_dim, config.hidden_dim, config.num_categories, config.category_dim)
print('RSSM OK')
"
```

### 8.3 Testar Ambiente

```python
python -c "
from environment.env_interface import EnvInterface
env = EnvInterface('MiniGrid-Empty-5x5-v0', num_envs=1)
obs = env.reset()
print('Ambiente OK, obs shape:', obs[0].shape)
"
```

## 9. Exemplo de Execução Completa
```bash
# 1. Ativar ambiente virtual
source venv/bin/activate

# 2. Verificar GPU
python -c "import torch; print('GPU:', torch.cuda.is_available())"

# 3. Executar com configuração otimizada
python main.py --env MiniGrid-Empty-5x5-v0 --render

# 4. Após treinar, carregar checkpoint
python main.py --env MiniGrid-DoorKey-8x8-v0 --load checkpoint_step_10000.pt --render
```

## 10. Dicas para Melhor Performance
- Use GPU: Instale a versão CUDA do PyTorch e FAISS-GPU
- Aumente num_envs: 8-16 ambientes paralelos aceleram muito
- Reduza warmup_steps: 1000 para testes rápidos
- Desative LLM: Use use_llm = False se não precisar
- Use torch.compile: Adicione no código após criar os modelos:

```python
rssm = torch.compile(rssm)
actor_critic = torch.compile(actor_critic)
```

## 11. Limpeza
Para remover logs antigos:

```bash
rm -rf logs/
```

Para recomeçar do zero:

```bash
rm -f checkpoint_*.pt
```

## 12. Referências Rápidas
|Comando	                    | Descrição                     |
|python main.py --help	        | Mostra todos os argumentos    |
|--env MiniGrid-Empty-5x5-v0	| Ambiente mais simples         |
|--env MiniGrid-DoorKey-8x8-v0	| Ambiente com portas e chaves  |
|--render	                    | Ativa visualização            |
|--load checkpoint.pt	        | Carrega checkpoint            |


# 3. Implementação do Lado Unreal Engine (C++ / Blueprint)

### 3.1 Estrutura do Plugin Unreal
Crie um plugin chamado CognitiveAgent com os seguintes componentes:

```text
CognitiveAgent/
├── Source/
│   ├── CognitiveAgent/
│   │   ├── Public/
│   │   │   ├── CognitiveAgentModule.h
│   │   │   └── CognitiveAgentClient.h
│   │   └── Private/
│   │       ├── CognitiveAgentModule.cpp
│   │       └── CognitiveAgentClient.cpp
│   └── CognitiveAgentEditor/
│       └── ...
└── Content/
    └── Blueprints/
        └── BP_CognitiveAgent.uasset
```

### 3.3 Blueprint Setup no Unreal
- Criar um Blueprint baseado em ACognitiveAgentClient (ex.: BP_CognitiveAgent)
- Adicionar ao Level
- Configurar no Event Graph:

```text
Event BeginPlay
    └─> Connect("127.0.0.1", 8888)

Event Tick (ou Timer)
    └─> Capture Scene Camera
    └─> SendFrameAndGetAction(CameraTexture, Reward, bDone)
        └─> OnActionReceived -> Mover Jogador
```

### 7. Comandos para Testar

```bash
# 1. Iniciar o agente Python (aguardando conexão)
python main.py --unreal --render

python -m unittest discover -s tests -p "test_*.py" -v

# 2. Executar o projeto Unreal
# O Unreal se conecta automaticamente ao agente

# 3. Ver logs
tail -f logs/cognitive_agent.log
```