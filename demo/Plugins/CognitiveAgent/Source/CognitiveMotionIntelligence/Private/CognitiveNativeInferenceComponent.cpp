#include "CognitiveNativeInferenceComponent.h"
#include "CognitiveDebugLog.h"
#include "Interfaces/IPluginManager.h"
#include "Misc/Paths.h"
#include "HAL/PlatformTime.h"

// LibTorch só é incluída se instalada (WITH_LIBTORCH=1 vindo do Build.cs).
// IMPORTANTE: o c10/util/Flags.h usa `#ifdef C10_USE_GFLAGS` para decidir se
// inclui <gflags/gflags.h>. Como o teste é por existência (não por valor),
// garantimos aqui que a macro NÃO exista antes de puxar o torch — senão o
// build falha com 'Cannot open include file: gflags/gflags.h'.
#ifdef C10_USE_GFLAGS
  #undef C10_USE_GFLAGS
#endif
#ifdef C10_USE_GLOG
  #undef C10_USE_GLOG
#endif
#if WITH_LIBTORCH
THIRD_PARTY_INCLUDES_START
#include <torch/script.h>
#include <torch/torch.h>
THIRD_PARTY_INCLUDES_END
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Estado opaco LibTorch (mantido fora do header)
// ─────────────────────────────────────────────────────────────────────────────
struct UCognitiveNativeInferenceComponent::FTorchState
{
#if WITH_LIBTORCH
    torch::jit::script::Module Module;
    torch::Device Device = torch::kCPU;
    torch::Tensor H;   // (1, hidden_dim)
    torch::Tensor Z;   // (1, stochastic_dim)
    int64 LastAction = 0;
    bool bReady = false;
#endif
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
    LoadModel();
}

void UCognitiveNativeInferenceComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    Super::EndPlay(EndPlayReason);
    // O estado é liberado no destrutor. Aqui apenas marcamos como não-pronto
    // para impedir uso após o fim do play.
#if WITH_LIBTORCH
    if (Torch) { Torch->bReady = false; }
#endif
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
#if WITH_LIBTORCH
    if (!Torch) { bModelLoaded = false; return false; }

    const FString Path = ResolveModelPath();
    if (Path.IsEmpty() || !FPaths::FileExists(Path))
    {
        CMI_DBG("[NativeInfer] modelo não encontrado: %s", *Path);
        bModelLoaded = false;
        return false;
    }

    try
    {
        Torch->Device = (bUseGPU && torch::cuda::is_available())
                        ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);

        Torch->Module = torch::jit::load(TCHAR_TO_UTF8(*Path), Torch->Device);
        Torch->Module.eval();

        // Inicializa estado recorrente zerado
        Torch->H = torch::zeros({1, HiddenDim},     Torch->Device);
        Torch->Z = torch::zeros({1, StochasticDim}, Torch->Device);
        Torch->LastAction = 0;
        Torch->bReady = true;

        bModelLoaded = true;
        CMI_DBG("[NativeInfer] modelo carregado: %s | device=%s",
                *Path, (Torch->Device.is_cuda() ? TEXT("CUDA") : TEXT("CPU")));
        return true;
    }
    catch (const c10::Error& e)
    {
        CMI_DBG("[NativeInfer] ERRO ao carregar modelo: %s", UTF8_TO_TCHAR(e.what()));
        bModelLoaded = false;
        return false;
    }
#else
    CMI_DBG("[NativeInfer] LibTorch não compilada (WITH_LIBTORCH=0).");
    bModelLoaded = false;
    return false;
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
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
#if WITH_LIBTORCH
    if (Torch && Torch->bReady)
    {
        Torch->H = torch::zeros({1, HiddenDim},     Torch->Device);
        Torch->Z = torch::zeros({1, StochasticDim}, Torch->Device);
        Torch->LastAction = 0;
    }
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
int32 UCognitiveNativeInferenceComponent::RunInference(
    const TArray<float>& ObsEnc, TArray<FTransform>& OutBones)
{
    OutBones.Reset();

#if WITH_LIBTORCH
    if (!bModelLoaded || !Torch || !Torch->bReady)
        return -1;

    const double T0 = FPlatformTime::Seconds();

    try
    {
        torch::NoGradGuard NoGrad;

        // ── Monta entradas ────────────────────────────────────────────────────
        // Ação anterior como one-hot
        torch::Tensor Action = torch::zeros({1, ActionDim}, Torch->Device);
        const int64 PrevA = FMath::Clamp<int64>(Torch->LastAction, 0, ActionDim - 1);
        Action[0][PrevA] = 1.0f;

        const bool bUseObs = ObsEnc.Num() > 0;
        torch::Tensor Obs;
        if (bUseObs)
        {
            Obs = torch::from_blob(
                const_cast<float*>(ObsEnc.GetData()),
                {1, ObsEnc.Num()}, torch::kFloat32).clone().to(Torch->Device);
        }
        else
        {
            Obs = torch::zeros({1, 256}, Torch->Device);
        }

        // ── Forward: (h, z, action, obs, use_obs) → (h', z', action_idx, pose) ──
        std::vector<torch::jit::IValue> Inputs;
        Inputs.push_back(Torch->H);
        Inputs.push_back(Torch->Z);
        Inputs.push_back(Action);
        Inputs.push_back(Obs);
        Inputs.push_back(bUseObs);

        auto Output = Torch->Module.forward(Inputs).toTuple();

        torch::Tensor HNew = Output->elements()[0].toTensor();
        torch::Tensor ZNew = Output->elements()[1].toTensor();
        torch::Tensor ActIdx = Output->elements()[2].toTensor();
        torch::Tensor Pose = Output->elements()[3].toTensor().to(torch::kCPU).contiguous();

        // Atualiza estado recorrente
        Torch->H = HNew;
        Torch->Z = ZNew;
        const int64 ActionIndex = ActIdx.item<int64>();
        Torch->LastAction = ActionIndex;

        // ── Decodifica poses → FTransform (loc3 + quat4 por bone) ──────────────
        const float* P = Pose.data_ptr<float>();
        const int32 NB = FMath::Min<int32>(NumBones, (int32)(Pose.numel() / 7));
        OutBones.Reserve(NB);
        for (int32 i = 0; i < NB; ++i)
        {
            const int32 Base = i * 7;
            const FVector Loc(P[Base + 0], P[Base + 1], P[Base + 2]);
            // quaternion exportado em ordem (x, y, z, w)
            FQuat Quat(P[Base + 3], P[Base + 4], P[Base + 5], P[Base + 6]);
            if (!Quat.IsNormalized()) Quat.Normalize();
            OutBones.Add(FTransform(Quat, Loc, FVector::OneVector));
        }

        LastActionIndex = (int32)ActionIndex;
        LastInferenceMs = (float)((FPlatformTime::Seconds() - T0) * 1000.0);

        CMI_DBG("[NativeInfer] ação=%d | bones=%d | %.2fms",
                LastActionIndex, OutBones.Num(), LastInferenceMs);
        return LastActionIndex;
    }
    catch (const c10::Error& e)
    {
        CMI_DBG("[NativeInfer] ERRO inferência: %s", UTF8_TO_TCHAR(e.what()));
        return -1;
    }
#else
    return -1;
#endif
}
