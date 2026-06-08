# Guia de Integração — Comportamento Cognitivo do NPC

Este guia conecta TUDO que já está implementado e testado no servidor Python
ao seu projeto Unreal. Os valores aqui foram extraídos diretamente do código
(não são suposições). Siga na ordem.

> **Regra de ouro da versão:** o formato do wire de percepção mudou (cada
> entidade agora tem 44 bytes, com o campo `Role`). O C++ e o Python precisam
> estar na MESMA versão. Comite os dois juntos na sua branch.

---

## 1. O que já funciona (testado no servidor)

- Percepção de entidades (categoria, disposição, papel social, ameaça, direção).
- Emoções derivadas: calm, happy, alert, fear, anger, panic.
- Relações: friend, ally, neutral, enemy, hostage, captor.
- Geometria social: front, left, right, back, out_of_view (do vetor de direção).
- 5 perfis com reações distintas: urbano, militar, esportivo, piloto, lutador.
- Cenário sequestrador + refém (decide atacar / aguardar / proteger).
- A percepção alimenta tanto a DECISÃO quanto o TREINO.

## 2. O que depende de você no Unreal (não é código Python)

1. Compilar o C++ no VS2026.
2. Taggear os atores da cena (componente Cognitive Entity Tag).
3. Enviar o blackboard a cada frame (campos na seção 4).
4. Mapear cada `physical_state` a uma animação (seção 5).
5. Implementar veículos com física, se quiser pilotagem (seção 7).

---

## 3. Tags dos atores (componente Cognitive Entity Tag)

Cada ator que o NPC deve perceber precisa do componente **Cognitive Entity
Tag**. Campos relevantes:

| Campo | Valores | Para quê |
|---|---|---|
| Category | Character, Weapon, Pickup, Vehicle, TrafficLight, Cover, Hazard, Objective, Ignore | O que é a entidade |
| Disposition | Neutral, Friend, Enemy, Ally | Relação (só para Character) |
| Role | None, Hostage, Captor, Civilian, Wounded, Leader | Papel social (cenário refém) |
| ThreatWeight | 0.0 a 1.0 | Quão perigoso |
| VehicleType | None, Car, Motorcycle, Bicycle, Tank, Boat, Aircraft | Tipo de veículo |

**Cenário sequestrador + refém:** marque o sequestrador com
`Disposition=Enemy, Role=Captor, ThreatWeight≈0.8` e o refém com
`Role=Hostage`. O NPC só neutraliza o sequestrador com ângulo limpo.

---

## 4. Blackboard — campos que o NPC envia ao Python a cada frame

A decisão lê estes campos (todos opcionais; ausentes usam o default):

| Campo | Tipo | Default | Efeito |
|---|---|---|---|
| `npc_id` | int | 0 | Identifica o NPC |
| `health` | float | 100 | <= 0 → estado "dead" |
| `fear_level` | float | 0.0 | Medo; alto → foge/esconde |
| `aggression_level` | float | 0.0 | Agressão; alto → ataca |
| `threat_level` | float | 0.0 | Ameaça via blackboard (fallback) |
| `alertness` | float | 0.0 | Usado pelo LLM de estilo |
| `stamina` | float | 100 | Usado pelo LLM de estilo |
| `profile` | string | "" | Perfil: urbano/militar/esportivo/piloto/lutador |

> `profile` é o que faz o MESMO inimigo gerar reações diferentes. Defina por
> NPC (um militar agressivo ataca; um urbano medroso foge).

---

## 5. physical_state → animação (mapeie no AnimGraph/Blueprint)

O Python responde com um `PhysicalState` (enum `ECognitivePhysicalState`). O
Unreal recebe o número e você decide qual animação/montagem tocar:

| Estado | Valor | Animação sugerida (você define) |
|---|---|---|
| Alive | 0 | locomoção normal (a política controla) |
| Dead | 1 | morte / ragdoll |
| Falling | 2 | queda |
| Swimming | 3 | nado |
| Landing | 4 | pouso |
| Attack | 5 | montagem de ataque (soco/chute/espada/tiro) |
| Flee | 6 | corrida de fuga |
| Hide | 7 | agachar / cobertura |
| PickUp | 8 | pegar objeto |
| Enter | 9 | entrar no veículo |

> O Python NÃO escolhe QUAL animação de ataque (soco vs espada vs tiro). Ele
> diz "Attack"; o AnimGraph escolhe conforme a arma equipada / contexto.

---

## 6. Perfis e como cada um reage (referência rápida)

Reação ao MESMO inimigo (fear=0.5, aggression=0.5), conforme testado:

| Perfil | Reação | Caráter |
|---|---|---|
| urbano | esconde | civil, cauteloso, foge cedo |
| militar | ataca | treinado, agressivo, tático |
| esportivo | aproxima (confronta sem arma) | atlético, corajoso |
| piloto | foge | evita combate a pé, busca veículo |
| lutador | ataca | corpo a corpo, muito agressivo |

Para ajustar: mude `fear_level`/`aggression_level` no blackboard, ou edite os
pesos do perfil em `planning/behavior_catalog.py` (PROFILES).

---

## 7. Veículos (pilotagem) — o que falta

O catálogo decide `enter` (entrar no veículo). A partir daí, **dirigir** cada
veículo é implementação de UE5 que não existe no Python:

- Carro, ônibus, caminhão: PhysX Vehicle / Chaos Vehicle.
- Moto, bicicleta: veículo de 2 rodas (Chaos).
- Avião: física de voo customizada.
- Tanque: tração de esteira.
- Skate, patins: movimento customizado.

O perfil "piloto" prioriza `enter`; o resto é trabalho de Blueprint/C++ de
veículo no seu projeto.

---

## 8. Checklist de validação (o que olhar no log do Python)

Com o líder se movendo de forma variada e atores taggeados:

1. `VOCABULÁRIO ensinado pelo líder` — o teach chegou.
2. `PERCEPÇÃO: N entidade(s)` — a percepção chega, com categoria/role corretos.
3. `RECOMPENSA passo=... tarefa=...` — a percepção entra no treino.
4. `[estado] ...` nas decisões — o catálogo está decidindo (attack/flee/hide).
5. `WM loss` caindo ao longo do tempo — o modelo está aprendendo a imitar.

Se a decisão de combate não disparar: confira se os atores têm tag e se o
blackboard envia `fear_level`/`aggression_level`/`profile`.

---

## 9. Limites honestos (o que NÃO é mágica)

- A decisão de combate é por **regras configuráveis**, não emergente do RL. O
  cérebro neural aprende a IMITAR movimento; a decisão de atacar/fugir vem do
  catálogo determinístico. Fazer emergir do RL exigiria recriar o modelo de
  256 dimensões do zero.
- Animações e veículos são trabalho de editor Unreal.
- O C++ foi validado por inspeção (enums, formato binário byte a byte,
  balanceamento), mas não foi compilado fora do seu VS2026.
