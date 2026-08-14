# CMI Worker — inferência LibTorch isolada (resolve o crash 0xC0000374)

## O que é

A LibTorch (`torch_cpu.dll` + `libiomp5md.dll`) e o Unreal (Mimalloc) redirecionam
`malloc`/`free` de formas incompatíveis. Quando a LibTorch roda **dentro** do
`UnrealEditor.exe`, o primeiro `forward` corrompe a heap → `0xC0000374` → editor
fecha.

A solução: a LibTorch agora roda num **processo separado** (`cmi_worker.exe`).
O plugin spawna esse processo no BeginPlay e fala com ele por um *named pipe*
local. Como a torch nunca entra no processo do Unreal, **não há mais conflito de
heap**. O `.pt` não muda. A API do componente (`RunInference`) não muda — só a
implementação interna.

## Como compilar o worker (uma vez, e a cada vez que mudar `cmi_worker.cpp`)

Pré-requisitos: CMake (https://cmake.org/download/) e o Visual Studio 2022/2026
com toolchain C++ (o mesmo que compila o UE).

No PowerShell, dentro de `Source/Worker/`:

```powershell
cd Source\Worker
mkdir build
cd build
cmake -G "Visual Studio 17 2022" -A x64 `
      -DCMAKE_PREFIX_PATH="..\..\ThirdParty\LibTorch\LibTorch" ..
cmake --build . --config Release
```

> Se o seu gerador for o VS 2026, troque `"Visual Studio 17 2022"` pelo nome do
> gerador correspondente (rode `cmake --help` para ver a lista).

Isso gera `build\Release\cmi_worker.exe` **junto com as DLLs da LibTorch**
(copiadas automaticamente pelo CMake).

## Onde colocar o worker

Copie **todo o conteúdo** de `build\Release\` (o `cmi_worker.exe` e as DLLs da
LibTorch ao lado dele) para:

```
<Projeto>\Plugins\CognitiveAgent\Binaries\Win64\
```

O plugin procura `cmi_worker.exe` exatamente nessa pasta. As DLLs da torch
precisam estar **ao lado do `cmi_worker.exe`** (não ao lado do UnrealEditor.exe).

## Recompilar o plugin

O módulo do plugin **não linka mais a LibTorch** (essa é a correção). Apague
`Plugins\CognitiveAgent\Binaries\` e `Plugins\CognitiveAgent\Intermediate\` e
recompile o projeto normalmente pelo editor ou IDE.

> Importante: depois de recompilar, copie o `cmi_worker.exe` + DLLs para
> `Binaries\Win64\` de novo (o passo acima), pois apagar `Binaries\` remove o worker.

## Verificação

No Play, o log deve mostrar:

```
[NativeInfer][TRACE] modelo achado: ...CognitiveModel.pt
[WorkerClient] worker conectado e modelo carregado (pipe=cmi_infer_...)
[NativeInfer][TRACE] worker start (ok=1)
[NativeInfer][TRACE] BeginPlay FIM (bModelLoaded=1)
[NativeInfer] ação=N | bones=89 | X.XXms
```

Sem `0xC0000374`, sem o editor fechar. O NPC anima.

## Fallback

Se o worker não subir (ex.: `cmi_worker.exe` ausente), `LoadModel` retorna false
e o BoneDriver/Learner caem no fallback (TCP/Python ou comportamento offline),
sem crash.
