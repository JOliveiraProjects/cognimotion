#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "CognitiveMotionTypes.h"
#include "CognitiveTrajectoryGenerator.h"
#include "CognitiveRuntimePoseMemory.h"
#include "CognitiveMotionLearnerComponent.generated.h"

class UCognitiveInferenceSubsystem;
class UCognitivePoseRecorderComponent;
class UCognitiveAnimInstance;
class UCognitiveTrajectoryGenerator;
class UCognitiveRuntimePoseMemory;
class UCognitiveNativeInferenceComponent;


DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(
    FOnCognitivePhysicalStateChanged, ECognitivePhysicalState, NewState);

UCLASS(ClassGroup=(Cognitive), meta=(BlueprintSpawnableComponent))
class COGNITIVEMOTIONINTELLIGENCE_API UCognitiveMotionLearnerComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    UCognitiveMotionLearnerComponent();

    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type Reason) override;
    virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

    UFUNCTION(BlueprintCallable, Category = "Cognitive|Learning")
    void RequestMotionInference(const FCognitiveBlackboard& Blackboard, const FCognitiveTrajectory& DesiredTrajectory);

    UFUNCTION(BlueprintPure, Category = "Cognitive|Learning")
    const FCognitiveMotionResponse& GetLatestResponse() const { return LatestResponse; }

    UFUNCTION(BlueprintPure, Category = "Cognitive|Learning")
    bool HasValidResponse() const { return LatestResponse.bValid; }

    UFUNCTION(BlueprintPure, Category = "Cognitive|Learning")
    bool IsInFallbackMode() const { return bFallbackActive; }

    UFUNCTION(BlueprintPure, Category = "Cognitive|Learning")
    float GetResponseConfidence() const { return LatestResponse.Embedding.Confidence; }

    UFUNCTION(BlueprintCallable, Category = "Cognitive|Learning")
    void SetMotionStyle(ECognitiveMotionStyle Style);

    // MaxInferenceLatencyMs: tolerância de latência do Python.
    // CPU: 200-500ms por request. 80ms causava fallback imediato (80*0.002=0.16s).
    // 2000ms = fallback só ativa se Python não responder em 4 segundos (2000*0.002).
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cognitive|Learning")
    float MaxInferenceLatencyMs = 2000.f;

    // RequestIntervalSeconds: frequência de requests ao Python.
    // 0.05s (20Hz) sobrecarregava CPU Python — fila acumulava → timeout → disconnect.
    // 0.5s (2Hz): Python em CPU consegue responder confortavelmente.
    // Para GPU aumentar para 0.1s (10Hz) ou 0.033s (30Hz).
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cognitive|Learning")
    float RequestIntervalSeconds = 0.5f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cognitive|Learning")
    float FallbackBlendSpeed = 4.f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cognitive|Learning")
    bool bAutoConnect = true;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cognitive|Learning")
    FString PythonHost = TEXT("127.0.0.1");

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cognitive|Learning")
    int32 PythonPort = 9000;

    UPROPERTY(BlueprintAssignable, Category = "Cognitive|Learning")
    FCognitiveMotionResponseDelegate OnResponseReceived;

    // Última ação selecionada pelo Python (ação discreta 0-8)
    // 0=idle,1=fwd,2=back,3=left,4=right,5=run,6=jump,7=crouch,8=stop
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Cognitive|Debug")
    int32 LastSelectedStyle = 0;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Cognitive|Debug")
    FCognitiveMotionQualityMetrics MotionQuality;

    // Estado físico atual decidido pelo Python (vida/morte/queda/natação).
    // A AnimBP lê isto para tocar a animação correta (morte, queda, etc).
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Cognitive|Debug")
    ECognitivePhysicalState PhysicalState = ECognitivePhysicalState::Alive;

    UFUNCTION(BlueprintPure, Category = "Cognitive|Learning")
    ECognitivePhysicalState GetPhysicalState() const { return PhysicalState; }

    UFUNCTION(BlueprintPure, Category = "Cognitive|Learning")
    bool IsDead() const { return PhysicalState == ECognitivePhysicalState::Dead; }

    // Vida externa (vinda do UCognitiveHealthComponent). Alimenta o blackboard
    // enviado à inferência para a decisão reativa (fugir/morrer).
    UFUNCTION(BlueprintCallable, Category = "Cognitive|Learning")
    void SetExternalHealth(float InHealth) { ExternalHealth = InHealth; bHasExternalHealth = true; }

    // Força o estado físico diretamente (usado pelo Health component para
    // sinalizar morte mesmo quando a inferência neural não está ativa).
    UFUNCTION(BlueprintCallable, Category = "Cognitive|Learning")
    void ForcePhysicalState(ECognitivePhysicalState NewState)
    {
        if (NewState != PhysicalState)
        {
            PhysicalState = NewState;
            OnPhysicalStateChanged.Broadcast(PhysicalState);
        }
    }

    float ExternalHealth = 100.f;
    bool  bHasExternalHealth = false;

    // Disparado quando o estado físico muda (vivo→morto, etc). Ligue na AnimBP/ator.
    UPROPERTY(BlueprintAssignable, Category = "Cognitive|Learning")
    FOnCognitivePhysicalStateChanged OnPhysicalStateChanged;

private:
    void PollInferenceResponses();
    void ActivateFallback(const FString& Reason);
    void DeactivateFallback();
    void UpdateFallbackBlend(float DeltaTime);
    void EmitDebugDraw() const;

    // Fallback OFFLINE: roda o modelo .pt nativo (LibTorch) quando o servidor
    // Python não está disponível, para o NPC continuar funcionando sozinho.
    // Retorna true se a inferência nativa rodou e aplicou bones neste frame.
    bool TickNativeFallback(float DeltaTime);
    UCognitiveNativeInferenceComponent* ResolveNativeInference();

    // Componente de inferência nativa (.pt). Resolvido sob demanda do mesmo ator.
    UPROPERTY()
    TObjectPtr<UCognitiveNativeInferenceComponent> NativeInference;

    UPROPERTY()
    TObjectPtr<UCognitiveInferenceSubsystem> InferenceSubsystem;

    TWeakObjectPtr<UCognitivePoseRecorderComponent> PoseRecorder;
    TWeakObjectPtr<UCognitiveAnimInstance>          AnimInstance;
    UPROPERTY()
    TObjectPtr<UCognitiveTrajectoryGenerator>       TrajectoryGenerator;
    UPROPERTY()
    TObjectPtr<UCognitiveRuntimePoseMemory>         PoseMemory;

    FCognitiveMotionResponse LatestResponse;
    FCognitiveMotionIdentity CurrentIdentity;

    float RequestAccumulator = 0.f;
    float FallbackBlendAlpha = 0.f;
    bool  bFallbackActive    = false;
    int64 NextSequenceId     = 1;

    double LastResponseTime  = 0.0;
    double LastRequestTime   = 0.0;
};
