# Inferência Nativa (LibTorch) — Sem rede, sem Python em runtime

Esta é a arquitetura AAA: o treino continua em Python (onde funciona), mas a
**inferência roda 100% nativa em C++ dentro do Unreal**, via LibTorch. Sem TCP,
sem Python rodando, sem latência de rede.

## Fluxo completo

```
[TREINO - Python]                    [RUNTIME - C++ no Unreal]
  Observa o líder                       Carrega CognitiveModel.pt (LibTorch)
  Treina RSSM + PoseDecoder + Actor     A cada frame:
  Salva checkpoint (.pt)                  RSSM → Actor → PoseDecoder
       │                                  → ação + 89 poses de bones
       │  export_torchscript.py           → aplica no esqueleto + move a cápsula
       ▼
  CognitiveModel.pt  ──────────────►  <Plugin>/Content/Models/CognitiveModel.pt
```

## Passo 1 — Treinar (como já faz)

Rode o servidor Python em modo Observing Leader até o NPC aprender. Acompanhe
`wm/pose` nos logs: quando cair perto de 0, o PoseDecoder aprendeu a animação.
Os checkpoints são salvos em `checkpoints/policy_vNNNNNN.pt`.

## Passo 2 — Exportar para TorchScript

```bash
cd llm
python export_torchscript.py --checkpoint checkpoints/policy_v000002.pt \
                             --output CognitiveModel.pt
```

Isso empacota RSSM + Actor + PoseDecoder num único `.pt` que roda sem Python.

## Passo 3 — Instalar a LibTorch no plugin (uma vez)

1. Baixe a LibTorch para Windows que case com a versão do treino
   (treino usa torch 2.12 + CUDA 13). Em https://pytorch.org/get-started/locally/
   escolha: LibTorch / C++ / Windows / Release. Use a versão **Release**.
2. Extraia para: `Plugins/CognitiveAgent/Source/ThirdParty/LibTorch/`
   Deve conter `include/` e `lib/`.
3. Copie o modelo: `CognitiveModel.pt` → `Plugins/CognitiveAgent/Content/Models/`
4. Regenere os arquivos do projeto (botão direito no .uproject → Generate
   Visual Studio project files) e recompile.

> Se a LibTorch NÃO estiver instalada, o plugin compila normalmente com
> `WITH_LIBTORCH=0` e continua usando o servidor Python via TCP como antes.
> A inferência nativa é um upgrade opcional, não quebra nada.

## Passo 4 — Usar no NPC

1. No ator do NPC, adicione o componente **Cognitive Native Inference**.
2. Confirme as dimensões (devem casar com o modelo exportado):
   HiddenDim=512, StochasticDim=1024, ActionDim=9, NumBones=89.
3. Mude o ObservationState para **Inferring**.

Pronto. No modo Inferring, o BoneDriver detecta o componente nativo, roda a
inferência localmente (RSSM→Actor→PoseDecoder), aplica as 89 poses geradas no
esqueleto e move a cápsula conforme a ação — tudo sem rede.

## Por que não compilar o Python direto no C++?

A latência do TCP em localhost é ~0.1ms — desprezível. O custo real é a
inferência da rede neural (PyTorch). Embedar o Python manteria esse custo e
ainda exigiria Python+PyTorch instalados, com risco de o GIL travar o game
thread. A LibTorch elimina a rede E o Python em runtime, rodando a mesma rede
neural de forma nativa e thread-safe — é a solução correta.

## Verificação de aprendizado

- `LastInferenceMs` no componente mostra o tempo de inferência (deve ser baixo).
- `LastActionIndex` mostra a ação escolhida a cada frame.
- Ative o debug (`Cognitive Motion → Debug Dashboard`) para ver `[NativeInfer]`
  nos logs com ação, número de bones e latência.
