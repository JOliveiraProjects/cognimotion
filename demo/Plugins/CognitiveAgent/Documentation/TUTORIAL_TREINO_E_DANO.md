# Tutorial Completo — Treino, Telemetria de Aprendizado e Dano/Reação

Este guia te leva do zero a uma demo funcional: configurar o plugin, treinar
Idle/Morte/etc., **saber com certeza se está aprendendo**, e fazer o Player
causar dano no NPC com reação.

---

# PARTE 1 — Como saber se está aprendendo (telemetria)

Antes de treinar, entenda os sinais. Agora você tem leitura clara nos dois lados.

## No Python (terminal do servidor de treino)

A cada 100 passos aparece um painel de **VEREDITO**:

```
╔════════════════════════════════════════════════════════════════╗
  APRENDIZADO: SIM ✓
  treino = Luta|MMA   |   modo = Observing   |   passo 1500
  frames recebidos = 1820   sequências = 12
  ------------------------------------------------------------
  WM loss       +2.6492  ↓ caindo     (bom)  modela o mundo
  Pose loss     +0.5300  ↓ caindo     (bom)  gera animação
  Entropia      +0.4100  ↓ caindo     (bom)  política decide
  Retorno       +1.4600  ↑ subindo    (bom)  recompensa
  Confiança     +0.5940  ↑ subindo    (bom)
╚════════════════════════════════════════════════════════════════╝
```

Como ler:
- **APRENDIZADO: SIM** = pelo menos 2 dos 3 sinais-chave estão certos.
- **WM loss ↓** — o world model está modelando melhor o movimento.
- **Pose loss ↓** — a animação gerada está ficando parecida com o líder. **É o
  sinal mais importante para a animação.**
- **Entropia ↓** — a política está decidindo (menos aleatória).
- **Retorno ↑** — a recompensa acumulada cresce.
- **frames recebidos** — confirma que o Python está recebendo dados do Unreal.

Se aparecer **"COLETANDO DADOS…"**, ainda faltam frames — continue observando.
Se **"AINDA NÃO"** após muitos passos, o líder está parado demais (varie os
movimentos) ou o buffer está pequeno.

Linhas adicionais úteis no Python:
- `[DBG][INFER] RECEBIDO seq=… | ENVIADO ação=… conf=… bones=… estado=…`
  mostra cada inferência. Ative com `set_debug(True)` ou `CMI_DEBUG=1`.

## No Unreal (Output Log)

- `[ENVIO→Python] treino=Luta|MMA | estado_loco=0 | bones=89 | enviados=300`
  — o que o plugin **envia** (a cada ~1s).
- `[RECEBIDO←Python] ação=5 | conf=0.61 | bones=89 | estado_físico=0 | latência=8.2ms`
  — o que o plugin **recebe**.

Ative os logs: no **Debug Dashboard** (menu Cognitive Motion → Debug Dashboard),
botão "Enable Debug Logs". O dashboard também mostra confiança, latência e a
última ação ao vivo, por NPC.

**Resumo prático:** se no Python o painel diz SIM e a Pose loss cai, e no Unreal
você vê ENVIO e RECEBIDO fluindo, está aprendendo de verdade.

---

# PARTE 2 — Configurar o plugin para treinar

## 2.1 Setup do NPC (uma vez)

No ator do NPC (um Character com SkeletalMesh), adicione os componentes:

1. **Cognitive Leader Observer** — observa o personagem-líder (o Player).
2. **Cognitive NPC Bone Driver** — envia/aplica poses; controla o modo.
3. **Cognitive Motion Learner** — recebe respostas do Python.
4. **Cognitive Health** — vida e dano (Parte 4).
5. (Opcional) **Cognitive World Perception** — percepção do mundo.
6. (Opcional) **Cognitive Native Inference** — para rodar modelo .pt sem rede.

No **Bone Driver** (Details, categoria "Cognitive | Setup"):
- **Observation State** = `Observing Leader` (para treinar).
- **Training Context → Category** = o tipo de treino (Urbano/Luta/Esporte/
  Zumbi/Corrida).
- **Training Context → Subtype** = o nome livre ("MMA", "pedestre", "futebol").
- **Locomotion State** = como sinalizar o frame atual (ver 2.2).

No **Leader Observer**: deixe **Target Leader** vazio para usar o Player
automaticamente, ou aponte para o personagem que o NPC deve imitar.

## 2.2 O conceito do treino único com estados

Você treina **uma categoria por vez** (ex.: Luta|MMA), e dentro dela o NPC
aprende idle, walk, run, ações e morte — tudo junto. O que diferencia é o
**Locomotion State** que você sinaliza durante a captura:

- Parado → `Idle`
- Andando → `Walk`
- Correndo → `Run`
- Executando a ação da categoria (soco, chute…) → `Action`
- Sem vida → `Dead`

O Python recebe esse rótulo junto com cada frame e aprende a associar.

---

# PARTE 3 — Roteiro de treino para a demo

## 3.1 Treino de IDLE (parado)

1. Bone Driver: Category=`Urbano`, Subtype=`pedestre`, Observation State=
   `Observing Leader`.
2. No Blueprint do NPC, sinalize idle:
   ```
   (BeginPlay ou quando o líder está parado)
   → Bone Driver → Set Locomotion State (Idle)
   ```
3. Deixe o líder **parado** variando pequenas poses (respiração, olhar) por
   3–5 min.
4. Confira no Python: painel com `treino = Urbano|pedestre`, Pose loss caindo.

## 3.2 Treino de WALK/RUN

1. Mantenha a mesma Category/Subtype.
2. Sinalize conforme a velocidade do líder:
   ```
   Event Tick → Get Velocity → Size
     > 300 → Set Locomotion State (Run)
     > 10  → Set Locomotion State (Walk)
     else  → Set Locomotion State (Idle)
   ```
3. Movimente o líder andando e correndo por 5–10 min.

## 3.3 Treino de MORTE (Dead)

A animação de morte é aprendida como qualquer outra: sinalize `Dead` enquanto
o líder executa a pose de morte.

1. Coloque o líder na animação/pose de morte (deitado, caído).
2. Sinalize:
   ```
   (quando capturando a morte)
   → Bone Driver → Set Locomotion State (Dead)
   ```
3. Capture várias variações de queda/morte por 2–3 min.

Em runtime (Parte 4), quando a vida do NPC zera, o sistema seleciona o estado
Dead e o NPC reproduz a animação de morte aprendida.

## 3.4 Treino de uma categoria de ação (ex.: Luta|MMA)

1. Category=`Luta`, Subtype=`MMA`.
2. Sinalize `Action` durante os golpes, `Idle`/`Walk` entre eles.
3. Capture socos, chutes, esquivas por 10–15 min.

## 3.5 Exportar o treino para a demo (rodar sem Python)

Quando o painel disser SIM por um tempo e a Pose loss estiver baixa:

```bash
cd llm
python export_torchscript.py --checkpoint checkpoints/policy_vXXXXXX.pt \
                             --output CognitiveModel.pt
```

Copie `CognitiveModel.pt` para `Plugins/CognitiveAgent/Content/Models/`.
No NPC, mude Observation State para `Imported Model (.pt)` e adicione o
componente **Cognitive Native Inference**. Agora roda nativo, sem rede.

---

# PARTE 4 — Dano e reação (Player ↔ NPC)

Aqui o Player bate/atira/joga objeto no NPC, ele toma dano e reage (morre,
foge, ou se desvia).

## 4.1 Setup de vida

No NPC, com o componente **Cognitive Health**:
- **Max Health** = 100 (ou o que quiser).
- Ligue os eventos no Event Graph do NPC:
  ```
  Cognitive Health → On Death        → tocar animação de morte / desabilitar IA
  Cognitive Health → On Health Changed (NewHealth, Delta)
        → atualizar barra de vida (Get Health Fraction)
        → se Delta < 0: tocar reação de dano (flinch)
  ```

## 4.2 Player bate no NPC (melee)

No Blueprint do **Player**, no momento do golpe (notify da animação de soco):

```
(AnimNotify "Hit" no montage de ataque)
→ Sphere Trace / Get Overlapping Actors à frente do Player
→ Para cada ator atingido:
     Cast to BP_CognitiveNPC
     → (NPC) Cognitive Health → Apply Damage (Amount = 25)
```

O `Apply Damage` reduz a vida, dispara `On Health Changed` (flinch) e, se chegar
a zero, `On Death` (animação de morte aprendida).

## 4.3 Player atira no NPC

```
(ao disparar)
→ Line Trace By Channel (da câmera/arma para frente)
→ Break Hit Result → Hit Actor
→ Cast to BP_CognitiveNPC
   → Cognitive Health → Apply Damage (Amount = 40)
```

## 4.4 Player joga um objeto / objeto atinge o NPC

No objeto arremessável (um Actor com física + Cognitive Entity Tag opcional):

```
(no objeto) Event Hit (OtherActor)
→ Cast OtherActor to BP_CognitiveNPC
→ Branch: velocidade do objeto > limiar?
   True → Cognitive Health → Apply Damage (massa × velocidade ÷ fator)
```

## 4.5 NPC se desvia de um objeto (decisão de reação)

Use a percepção: marque o objeto perigoso com **Cognitive Entity Tag**
(Category=`Hazard`, ou um projétil com ThreatWeight alto). No NPC com
**Cognitive World Perception**:

```
Event Tick (ou timer)
→ (NPC) World Perception → Get Nearest Threat
→ Branch: Threat.Actor válido E Threat.Distance < 300?
   True → Switch on Threat.SuggestedReaction
            Flee → mover na direção oposta a Threat
            Hide → mover para o Cover mais próximo
            (Dodge) → tocar montage de esquiva + deslocamento lateral
```

Para um projétil vindo na direção do NPC, o `Get Nearest Threat` já entrega a
direção relativa (`RelativeDirection`) — use-a para escolher esquivar para o
lado oposto.

## 4.6 NPC bate em um objeto e sofre dano (ex.: cair, colidir)

No NPC:
```
Event Hit (do Character) → NormalImpulse Size > limiar?
   True → Cognitive Health → Apply Damage (impulso ÷ fator)
```

Ou, ao cair de muito alto, use o estado físico Falling→Landing para aplicar
dano de queda proporcional ao tempo de queda.

## 4.7 A morte fecha o ciclo

Quando `Apply Damage` zera a vida:
1. `On Death` dispara no NPC.
2. O sistema marca o estado físico como **Dead**.
3. Se você treinou o estado Dead (3.3), o NPC reproduz a animação de morte
   aprendida; senão, toque um montage de morte tradicional no `On Death`.

---

# PARTE 5 — Métodos e variáveis (referência rápida)

## Cognitive Health
- `Apply Damage(Amount)` — causa dano.
- `Heal(Amount)` — cura.
- `Set Health(NewHealth)` / `Kill()` / `Revive(WithHealth)`.
- `Get Health Fraction()` → 0..1 (barra de vida).
- `Is Dead()` → bool.
- Eventos: `On Death`, `On Health Changed(NewHealth, Delta)`.
- Propriedades: `Max Health`, `Start Health`, `Current Health`.

## Cognitive NPC Bone Driver
- `Set Observation State(State)` — Observing / Inferring / Imported.
- `Set Training Context(Category, Subtype)` — define o treino.
- `Set Locomotion State(State)` — Idle/Walk/Run/Action/Dead.
- `Has Valid Response()`, `Get Last Confidence()`.

## Cognitive World Perception
- `Get Nearest Threat()` → entidade hostil mais próxima (+ reação sugerida).
- `Get Nearest Pickup()` → item pegável.
- `Get Nearest Traffic State()` → estado do semáforo.
- `Attach Object To Hand(Object, "hand_r")` / `Drop Held Object()`.

## Cognitive Entity Tag (em atores do mundo)
- `Category`, `Disposition`, `Faction`, `bCanPickUp`, `VehicleType`,
  `TrafficState`, `ThreatWeight`.
- `Set Traffic State(State)`, `Set Disposition(D)`.

## Cognitive Motion Learner
- `Get Physical State()`, `Is Dead()`.
- Evento: `On Physical State Changed(NewState)`.

---

# PARTE 6 — Checklist da demo

- [ ] NPC com os 4 componentes essenciais (Observer, BoneDriver, Learner, Health).
- [ ] Python rodando; painel mostra `treino = …` e frames subindo.
- [ ] Painel diz APRENDIZADO: SIM e Pose loss caindo.
- [ ] Treinos capturados: Idle, Walk/Run, Dead, e a categoria de ação.
- [ ] Modelo exportado (.pt) em Content/Models e testado com Imported.
- [ ] Player causa dano (melee/tiro/objeto) e o NPC reage.
- [ ] Morte dispara animação de morte aprendida.
