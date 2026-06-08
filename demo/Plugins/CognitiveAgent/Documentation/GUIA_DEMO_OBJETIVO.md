# Guia Objetivo — Configurar a Demo

Três coisas: (1) NPC recebe dano, (2) trocar ObservationState em jogo,
(3) importar e ler o .pt no editor. Direto ao ponto.

═══════════════════════════════════════════════════════════════════════
## 1. NPC RECEBE DANO
═══════════════════════════════════════════════════════════════════════

No NPC (BP_CognitiveNPC), adicione o componente **Cognitive Health** e set
Max Health = 100.

### 1.1 Player bate no NPC (tecla/clique)
No Blueprint do Player, no evento de ataque:
```
InputAction Attack (ou LeftMouseButton)
  → Sphere Trace By Channel (Start: câmera, End: câmera + Forward*200)
  → Break Hit Result → Hit Actor
  → Cast to BP_CognitiveNPC
      → Get Component by Class (Cognitive Health)
      → Apply Damage (Amount = 25)
```

### 1.2 Ligar a reação de dano e morte (no NPC)
No Event Graph do BP_CognitiveNPC, no BeginPlay:
```
Get Component (Cognitive Health)
  → Bind Event to On Health Changed → [evento custom DanoRecebido]
  → Bind Event to On Death          → [evento custom Morrer]
```
Evento **DanoRecebido (NewHealth, Delta)**:
```
Branch (Delta < 0)
  True → Play Anim Montage (flinch / tomar dano)
       → Update barra de vida: Get Health Fraction → Set Percent
```
Evento **Morrer**:
```
→ Set Locomotion State (Dead)   [no Cognitive NPC Bone Driver]
→ Play Anim Montage (morte)  — ou deixa o modelo neural gerar se treinou Dead
→ Disable Input / Set Can Be Damaged = false
```

### 1.3 Objeto atinge o NPC
No objeto arremessável:
```
Event Hit (Other Actor)
  → Cast to BP_CognitiveNPC
  → Get Velocity → Size → Branch (> 400)
      True → Apply Damage (Amount = 20)
```

═══════════════════════════════════════════════════════════════════════
## 2. TROCAR ObservationState EM JOGO
═══════════════════════════════════════════════════════════════════════

O Bone Driver tem `Set Observation State`. Estados:
- **Observing Leader** (0) — treina observando o líder.
- **Inferring from Python** (1) — Python controla (teste durante treino).
- **Imported Model** (2) — roda o .pt nativo, sem rede.

### Trocar por tecla (no Player ou num Blueprint de controle):
```
Tecla 1 → Get NPC → Get Component (Cognitive NPC Bone Driver)
        → Set Observation State (Observing Leader)
Tecla 2 → ... Set Observation State (Inferring from Python)
Tecla 3 → ... Set Observation State (Imported Model)
```

### Importante por estado:
- **Observing**: precisa do servidor Python rodando + Leader Observer no NPC.
- **Inferring**: precisa do Python rodando.
- **Imported**: precisa do componente **Cognitive Native Inference** com o .pt
  carregado (ver seção 3). NÃO precisa de Python.

═══════════════════════════════════════════════════════════════════════
## 3. IMPORTAR E LER O .pt NO EDITOR
═══════════════════════════════════════════════════════════════════════

Adicione o componente **Cognitive Native Inference** ao NPC. Há dois jeitos:

### Jeito A — pasta padrão (mais simples)
1. Copie `CognitiveModel.pt` para:
   `<Projeto>/Plugins/CognitiveAgent/Content/Models/CognitiveModel.pt`
2. Deixe **Model Path** vazio no componente. Ao dar Play, ele procura
   automaticamente nessa pasta.
3. No log você verá: `[NativeInfer] modelo carregado: ... | device=CPU`

### Jeito B — importar de qualquer caminho (em jogo/editor)
Use o método novo `Load Model From File`:
```
(ex.: num botão de UI ou no BeginPlay)
Get Component (Cognitive Native Inference)
  → Load Model From File (FilePath = "C:/MeusModelos/CognitiveModel.pt")
```
Retorna true se carregou. Loga:
`[NativeInfer] LoadModelFromFile(...) → carregado`

### Conferir se carregou
- `Is Model Loaded()` → bool (use num Branch para confirmar).
- Ao entrar em **Imported**, o log mostra:
  `[NativeInfer] ação=3 | bones=89 | 1.7ms`
- Se aparecer `[NativeInfer] FALTA o componente...` → adicione o Cognitive
  Native Inference no NPC.
- Se aparecer `[NativeInfer] modelo .pt NÃO carregado` → confira o caminho.

> Pré-requisito: a LibTorch precisa estar instalada (WITH_LIBTORCH=1). Sem ela,
> o componente compila mas não infere. Confira o log de build: deve dizer
> WITH_LIBTORCH=1, não 0.

═══════════════════════════════════════════════════════════════════════
## 4. POR QUE O `[NativeInfer]` NÃO APARECIA
═══════════════════════════════════════════════════════════════════════

Dois motivos corrigidos nesta versão:
1. O componente nativo só era chamado em **Inferring**; agora roda em
   **Imported** (o estado correto para o .pt sem rede).
2. Faltava diagnóstico. Agora, em Imported, o log diz exatamente o que falta
   (componente ausente ou modelo não carregado).

E lembre: `[NativeInfer]` só aparece com a LibTorch instalada E o NPC em
**Imported** com o modelo carregado. Em Observing/Inferring ele não roda
(usa o Python).
