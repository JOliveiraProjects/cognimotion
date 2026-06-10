#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "CognitiveNativeInferenceComponent.generated.h"

/**
 * UCognitiveNativeInferenceComponent
 *
 * Inferência NATIVA do modelo cognitivo dentro do Unreal, via LibTorch —
 * SEM rede, SEM Python em runtime. Carrega o TorchScript exportado
 * (CognitiveModel.pt) e roda o pipeline completo a cada frame:
 *
 *   obs_enc + (h,z) anteriores  →  [RSSM → Actor → PoseDecoder]  →
 *   h',z' novos + ação discreta + 89 poses de bones (a animação gerada)
 *
 * O estado recorrente (h,z) é mantido entre frames neste componente.
 *
 * Coloque no MESMO ator do NPC, junto do BoneDriver. Quando presente e com o
 * modelo carregado, o BoneDriver usa este componente no modo Inferring em vez
 * do servidor Python — eliminando latência de rede.
 *
 * Treino continua em Python: exporte com export_torchscript.py e copie o .pt
 * para <Plugin>/Content/Models/CognitiveModel.pt (ou aponte ModelPath).
 */
UCLASS(ClassGroup=(Cognitive), meta=(BlueprintSpawnableComponent),
       DisplayName="Cognitive Native Inference")
class COGNITIVEMOTIONINTELLIGENCE_API UCognitiveNativeInferenceComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    UCognitiveNativeInferenceComponent();
    virtual ~UCognitiveNativeInferenceComponent() override;
    // Exigido pelo UObject quando há destrutor próprio + membro de tipo
    // incompleto: o helper de vtable é definido no .cpp (tipo completo).
    UCognitiveNativeInferenceComponent(FVTableHelper& Helper);

    // Caminho do modelo TorchScript. Vazio → procura em
    // <Plugin>/Content/Models/CognitiveModel.pt
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|NativeInference")
    FString ModelPath;

    // Usar GPU (CUDA) se disponível. Senão CPU.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|NativeInference")
    bool bUseGPU = false;

    // Dimensões — DEVEM casar com o modelo exportado.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|NativeInference")
    int32 HiddenDim = 512;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|NativeInference")
    int32 StochasticDim = 1024;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|NativeInference")
    int32 ActionDim = 9;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|NativeInference")
    int32 NumBones = 89;

    // ── Estado (read-only) ────────────────────────────────────────────────────
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category="Cognitive|NativeInference|Debug")
    bool bModelLoaded = false;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category="Cognitive|NativeInference|Debug")
    int32 LastActionIndex = 0;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category="Cognitive|NativeInference|Debug")
    float LastInferenceMs = 0.f;

    // Estado latente (read-only para debug): normas L2 de h (determinístico) e
    // z (estocástico). Dão uma leitura compacta e barata do "pensamento" do
    // modelo em tempo real, sem expor os tensores inteiros. Atualizados a cada
    // RunInference. Aparecem no painel Details e no Debug Dashboard.
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category="Cognitive|NativeInference|Debug")
    float LatentHiddenNorm = 0.f;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category="Cognitive|NativeInference|Debug")
    float LatentStochasticNorm = 0.f;

    // Confiança da última ação (softmax max), 0..1. Atualizada a cada inferência.
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category="Cognitive|NativeInference|Debug")
    float LastActionConfidence = 0.f;

    // ── API ───────────────────────────────────────────────────────────────────
    UFUNCTION(BlueprintCallable, Category="Cognitive|NativeInference")
    bool LoadModel();

    // Importa e carrega um .pt de um caminho arbitrário (ex.: escolhido no
    // editor). Define ModelPath e recarrega. Use isto para importar o modelo
    // treinado sem precisar copiá-lo para Content/Models manualmente.
    UFUNCTION(BlueprintCallable, Category="Cognitive|NativeInference")
    bool LoadModelFromFile(const FString& FilePath);

    UFUNCTION(BlueprintPure, Category="Cognitive|NativeInference")
    bool IsModelLoaded() const { return bModelLoaded; }

    /**
     * Roda um passo de inferência.
     * @param ObsEnc       embedding 256-d observado (pode ser vazio → usa prior)
     * @param OutBones     [out] 89 transforms gerados (a animação)
     * @return índice da ação discreta (0-8); -1 em falha
     */
    int32 RunInference(const TArray<float>& ObsEnc, TArray<FTransform>& OutBones);

    // Reseta o estado recorrente (h,z) — chame ao reativar o NPC.
    UFUNCTION(BlueprintCallable, Category="Cognitive|NativeInference")
    void ResetState();

    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

private:
    // Opaco — esconde tipos LibTorch do header (evita vazar torch/script.h).
    // Usamos ponteiro bruto (não TUniquePtr) porque o destrutor gerado pelo
    // UObject (.gen.cpp / FVTableHelper) instanciaria a destruição do
    // TUniquePtr num contexto onde FTorchState é incompleto (erro C4150).
    // A gerência é manual em construtor/destrutor, definidos no .cpp onde o
    // tipo é completo.
    struct FTorchState;
    FTorchState* Torch = nullptr;

    FString ResolveModelPath() const;
};
