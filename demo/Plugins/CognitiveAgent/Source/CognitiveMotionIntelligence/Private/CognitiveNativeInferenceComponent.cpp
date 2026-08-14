#include "CognitiveNativeInferenceComponent.h"
#include "CognitiveDebugLog.h"
#include "Interfaces/IPluginManager.h"
#include "Misc/Paths.h"
#include "HAL/PlatformTime.h"
#include "HAL/PlatformMisc.h"
#include "Components/SkeletalMeshComponent.h"

// Inferência isolada: a LibTorch roda num PROCESSO SEPARADO (cmi_worker.exe),
// comunicando por named pipe. Nada de torch dentro do processo do Unreal —
// elimina o conflito de heap (0xC0000374) entre Mimalloc do UE e o alocador
// da LibTorch. O cliente do pipe encapsula spawn do worker + protocolo.
#include "CMIWorkerClient.h"


// ─────────────────────────────────────────────────────────────────────────────
// Estado interno: cliente do worker isolado + estado recorrente (h,z) mantido
// no UE como arrays simples (a fonte da verdade fica aqui; o worker é stateless
// por frame). Nada de tipos LibTorch.
// ─────────────────────────────────────────────────────────────────────────────
struct UCognitiveNativeInferenceComponent::FTorchState
{
    FCMIWorkerClient Worker;
    TArray<float> H;          // (hidden_dim)
    TArray<float> Z;          // (stochastic_dim)
    int32 LastAction = 0;
    bool bReady = false;
};

// ─────────────────────────────────────────────────────────────────────────────
UCognitiveNativeInferenceComponent::UCognitiveNativeInferenceComponent()
{
    PrimaryComponentTick.bCanEverTick = false;
    Torch = new FTorchState();
}

// Construtor de vtable helper exigido pelo UObject. Definido aqui (tipo
// completo). Não aloca — o objeto real é criado no construtor padrão.
UCognitiveNativeInferenceComponent::UCognitiveNativeInferenceComponent(FVTableHelper& Helper)
{
    Torch = nullptr;
}

// Destrutor definido AQUI (não no header) onde FTorchState é tipo completo —
// necessário para deletar com segurança (corrige erro C4150).
UCognitiveNativeInferenceComponent::~UCognitiveNativeInferenceComponent()
{
    if (Torch)
    {
        delete Torch;
        Torch = nullptr;
    }
}

void UCognitiveNativeInferenceComponent::BeginPlay()
{
    Super::BeginPlay();

    // Logs de rastreamento com flush imediato: se o editor fechar, a ÚLTIMA
    // linha escrita no log diz exatamente onde parou. UE_LOG vai sempre ao
    // arquivo (não depende de toggle de debug).
    UE_LOG(LogTemp, Warning, TEXT("[NativeInfer][TRACE] BeginPlay INICIO"));
    GLog->Flush();

    LoadModel();

    UE_LOG(LogTemp, Warning, TEXT("[NativeInfer][TRACE] BeginPlay FIM (bModelLoaded=%d)"),
           bModelLoaded ? 1 : 0);
    GLog->Flush();
}

void UCognitiveNativeInferenceComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    Super::EndPlay(EndPlayReason);
    // Encerra o worker (fecha o pipe → o processo sai sozinho).
    if (Torch)
    {
        Torch->Worker.Stop();
        Torch->bReady = false;
    }
    bModelLoaded = false;
}

// ─────────────────────────────────────────────────────────────────────────────
FString UCognitiveNativeInferenceComponent::ResolveModelPath() const
{
    if (!ModelPath.IsEmpty() && FPaths::FileExists(ModelPath))
        return ModelPath;

    // Procura em <Plugin>/Content/Models/CognitiveModel.pt
    TSharedPtr<IPlugin> Plugin = IPluginManager::Get().FindPlugin(TEXT("CognitiveAgent"));
    if (Plugin.IsValid())
    {
        const FString Candidate = FPaths::Combine(
            Plugin->GetContentDir(), TEXT("Models"), TEXT("CognitiveModel.pt"));
        if (FPaths::FileExists(Candidate))
            return Candidate;
    }
    return ModelPath;
}

// ─────────────────────────────────────────────────────────────────────────────
bool UCognitiveNativeInferenceComponent::LoadModel()
{
    if (!Torch) { bModelLoaded = false; return false; }

    const FString Path = ResolveModelPath();
    if (Path.IsEmpty() || !FPaths::FileExists(Path))
    {
        CMI_DBG("[NativeInfer] modelo não encontrado: %s", *Path);
        bModelLoaded = false;
        return false;
    }

    UE_LOG(LogTemp, Warning, TEXT("[NativeInfer][TRACE] modelo achado: %s"), *Path);
    GLog->Flush();

    // Spawna o worker isolado e conecta. A LibTorch carrega o .pt DENTRO do
    // worker (outro processo) — nada de torch aqui. Se o worker não subir ou o
    // modelo não carregar lá, caímos no fallback (TCP/Python ou Learner).
    const bool bOk = Torch->Worker.Start(Path, HiddenDim, StochasticDim, ActionDim, ObsDim);

    UE_LOG(LogTemp, Warning, TEXT("[NativeInfer][TRACE] worker start (ok=%d)"), bOk ? 1 : 0);
    GLog->Flush();

    if (!bOk)
    {
        CMI_DBG("[NativeInfer] worker de inferência não disponível — nativo desativado.");
        bModelLoaded = false;
        Torch->bReady = false;
        return false;
    }

    // Estado recorrente (h,z) inicia zerado, mantido aqui no UE.
    Torch->H.Init(0.0f, HiddenDim);
    Torch->Z.Init(0.0f, StochasticDim);
    Torch->LastAction = 0;
    Torch->bReady = true;
    bModelLoaded = true;

    CMI_DBG("[NativeInfer] modelo carregado no worker isolado: %s", *Path);
    return true;
}

bool UCognitiveNativeInferenceComponent::LoadModelFromFile(const FString& FilePath)
{
    if (FilePath.IsEmpty() || !FPaths::FileExists(FilePath))
    {
        CMI_DBG("[NativeInfer] LoadModelFromFile: arquivo não existe: %s", *FilePath);
        return false;
    }
    ModelPath = FilePath;
    const bool bOk = LoadModel();
    CMI_DBG("[NativeInfer] LoadModelFromFile(%s) → %s", *FilePath,
            bOk ? TEXT("carregado") : TEXT("falhou"));
    return bOk;
}

void UCognitiveNativeInferenceComponent::ResetState()
{
    if (Torch && Torch->bReady)
    {
        Torch->H.Init(0.0f, HiddenDim);
        Torch->Z.Init(0.0f, StochasticDim);
        Torch->LastAction = 0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
int32 UCognitiveNativeInferenceComponent::RunInference(
    const TArray<float>& ObsEnc, TArray<FTransform>& OutBones)
{
    OutBones.Reset();

    if (!bModelLoaded || !Torch || !Torch->bReady || !Torch->Worker.IsRunning())
        return -1;

    const double T0 = FPlatformTime::Seconds();

    // ── Monta a ação anterior como one-hot ───────────────────────────────────
    TArray<float> Action;
    Action.Init(0.0f, ActionDim);
    const int32 PrevA = FMath::Clamp<int32>(Torch->LastAction, 0, ActionDim - 1);
    Action[PrevA] = 1.0f;

    // ── Observação (ou zeros quando não há obs do servidor) ───────────────────
    const bool bUseObs = ObsEnc.Num() > 0;
    TArray<float> Obs;
    if (bUseObs && ObsEnc.Num() == ObsDim)
    {
        Obs = ObsEnc;
    }
    else
    {
        // Sem obs (ou tamanho divergente): zeros. Com use_obs=false o modelo
        // ignora o obs (RSSM recebe None), então o conteúdo não importa.
        Obs.Init(0.0f, ObsDim);
    }

    // ── Forward no worker isolado (LibTorch noutro processo) ──────────────────
    TArray<float> OutH, OutZ, OutPose;
    int32 ActionIndex = 0;
    const bool bOk = Torch->Worker.Forward(
        Torch->H, Torch->Z, Action, Obs, bUseObs && ObsEnc.Num() == ObsDim,
        OutH, OutZ, ActionIndex, OutPose);

    if (!bOk)
    {
        CMI_DBG("[NativeInfer] forward no worker falhou — desativando nativo (cai no fallback).");
        bModelLoaded = false;
        Torch->bReady = false;
        return -1;
    }

    // ── Atualiza estado recorrente (mantido no UE) ────────────────────────────
    Torch->H = MoveTemp(OutH);
    Torch->Z = MoveTemp(OutZ);
    Torch->LastAction = ActionIndex;

    // ── Métricas de debug: normas L2 de h e z (leitura do "pensamento") ───────
    auto L2 = [](const TArray<float>& V)
    {
        double S = 0.0; for (float x : V) S += (double)x * (double)x;
        return (float)FMath::Sqrt(S);
    };
    LatentHiddenNorm     = L2(Torch->H);
    LatentStochasticNorm = L2(Torch->Z);
    // O modelo exporta só o índice da ação (não logits) → confiança uniforme.
    LastActionConfidence = 1.0f / FMath::Max(1, ActionDim);

    // ── Decodifica poses → FTransform (loc3 + quat4 por bone) ─────────────────
    const int32 NB = FMath::Min<int32>(NumBones, OutPose.Num() / 7);
    OutBones.Reserve(NB);
    for (int32 i = 0; i < NB; ++i)
    {
        const int32 Base = i * 7;
        const FVector Loc(
            OutPose[Base + 0] * PoseTranslationScale,
            OutPose[Base + 1] * PoseTranslationScale,
            OutPose[Base + 2] * PoseTranslationScale);
        // quaternion exportado em ordem (x, y, z, w)
        FQuat Quat(OutPose[Base + 3], OutPose[Base + 4], OutPose[Base + 5], OutPose[Base + 6]);
        if (!Quat.IsNormalized()) Quat.Normalize();
        OutBones.Add(FTransform(Quat, Loc, FVector::OneVector));
    }

    LastActionIndex = ActionIndex;
    LastInferenceMs = (float)((FPlatformTime::Seconds() - T0) * 1000.0);

    CMI_DBG("[NativeInfer] ação=%d | bones=%d | %.2fms",
            LastActionIndex, OutBones.Num(), LastInferenceMs);
    return LastActionIndex;
}


// ─────────────────────────────────────────────────────────────────────────────
// RemapPosesToMesh — retargeting por NOME de bone. Reordena as poses geradas
// (na ordem do modelo) para a ordem de bones do mesh atual, permitindo usar o
// mesmo .pt em skeletons humanoides diferentes sem retreinar. Mapeia por nome
// direto; se BoneRemapTable estiver preenchida, usa-a para nomes divergentes.
bool UCognitiveNativeInferenceComponent::RemapPosesToMesh(
    const TArray<FTransform>& ModelPoses,
    USkeletalMeshComponent* Mesh,
    TArray<FTransform>& OutRemapped) const
{
    // Precisa saber os nomes que o modelo gera e ter um mesh válido.
    if (!Mesh || ModelBoneNames.Num() == 0 || ModelPoses.Num() == 0)
        return false;

    const int32 MeshBoneCount = Mesh->GetNumBones();
    if (MeshBoneCount <= 0)
        return false;

    // Inicializa com identidade — bones do mesh sem correspondência ficam neutros.
    OutRemapped.Reset();
    OutRemapped.SetNum(MeshBoneCount);
    for (FTransform& T : OutRemapped)
        T = FTransform::Identity;

    const int32 N = FMath::Min(ModelPoses.Num(), ModelBoneNames.Num());
    int32 Mapped = 0;
    for (int32 i = 0; i < N; ++i)
    {
        const FName ModelBone = ModelBoneNames[i];

        // Resolve o nome correspondente no mesh: tabela, senão nome direto.
        FName MeshBone = ModelBone;
        if (const FName* Remapped = BoneRemapTable.Find(ModelBone))
        {
            MeshBone = *Remapped;
        }

        const int32 MeshIndex = Mesh->GetBoneIndex(MeshBone);
        if (MeshIndex != INDEX_NONE && MeshIndex < MeshBoneCount)
        {
            OutRemapped[MeshIndex] = ModelPoses[i];
            ++Mapped;
        }
    }

    // Só considera sucesso se mapeou uma fração razoável (evita aplicar lixo).
    return Mapped > 0;
}
