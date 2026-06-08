#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "CognitiveBehaviorTypes.h"
#include "CognitiveBoneTypes.h"
#include "CognitiveMotionTypes.h"
#include "CognitiveNPCBoneDriver.generated.h"

class UCognitiveInferenceSubsystem;
class UCognitiveAnimInstance;

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(
    FOnBoneResponseReceivedDelegate,
    FCognitiveBoneResponse, Response
);

/**
 * UCognitiveNPCBoneDriver
 *
 * Componente que envia todos os bones do NPC ao Python e recebe
 * bone transforms de volta para aplicar diretamente no skeleton.
 *
 * - Estado Observing  : NPC envia bones, Python aprende com o líder
 * - Estado Inferring: Python envia bone transforms → AnimInstance → AnimNode
 */
UCLASS(ClassGroup=(Cognitive), meta=(BlueprintSpawnableComponent),
       DisplayName="Cognitive NPC Bone Driver")
class COGNITIVEMOTIONINTELLIGENCE_API UCognitiveNPCBoneDriver : public UActorComponent
{
    GENERATED_BODY()

public:
    UCognitiveNPCBoneDriver();

    // ── Configuração ──────────────────────────────────────────────────────────
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|BoneDriver")
    FString PythonHost = TEXT("127.0.0.1");

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|BoneDriver")
    int32 PythonPort = 9000;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|BoneDriver")
    bool bAutoConnect = true;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|BoneDriver")
    ECognitiveObservationState ObservationState = ECognitiveObservationState::Observing;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|BoneDriver")
    FCognitiveBehaviorContext BehaviorContext;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|BoneDriver",
              meta=(ClampMin="0.0", ClampMax="120.0"))
    float SendRateHz = 30.f;

    // Alpha de blend para root motion (0=sem blend, 1=imediato)
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|BoneDriver",
              meta=(ClampMin="0.0", ClampMax="1.0"))
    float BlendAlpha = 0.5f;

    // ── Replicação de movimento físico ────────────────────────────────────────
    // Quando true: copia a velocidade do líder para o CharacterMovement do NPC.
    // O NPC se move fisicamente no mundo acompanhando o líder.
    // Necessário porque pose (bones) e movimento (cápsula) são sistemas separados.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|BoneDriver")
    bool bReplicateLeaderMovement = true;

    // Limiar mínimo de velocidade do líder para iniciar o movimento do NPC (cm/s)
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|BoneDriver",
              meta=(ClampMin="0.0"))
    float MovementThreshold = 5.f;

    // Máxima latência tolerada antes de desconsiderar a resposta (ms)
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|BoneDriver")
    float MaxLatencyMs = 500.f;

    // ── Eventos ───────────────────────────────────────────────────────────────
    UPROPERTY(BlueprintAssignable, Category="Cognitive|BoneDriver")
    FOnBoneResponseReceivedDelegate OnBoneResponseReceived;

    // ── Estado (read-only) ────────────────────────────────────────────────────
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category="Cognitive|BoneDriver|Debug")
    int32 BonesApplied = 0;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category="Cognitive|BoneDriver|Debug")
    float LastLatencyMs = 0.f;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category="Cognitive|BoneDriver|Debug")
    int32 TotalRequestsSent = 0;

    // ── Blueprint API ─────────────────────────────────────────────────────────
    UFUNCTION(BlueprintCallable, Category="Cognitive|BoneDriver")
    void SetObservationState(ECognitiveObservationState NewState);

    // Define a categoria e o subtipo do treino (ex.: Luta + "MMA").
    UFUNCTION(BlueprintCallable, Category = "Cognitive|Behavior")
    void SetTrainingContext(ECognitiveTrainingCategory Category, const FString& Subtype);

    // Atualiza o estado de locomoção sinalizado ao treino (idle/walk/run/dead).
    UFUNCTION(BlueprintCallable, Category = "Cognitive|Behavior")
    void SetLocomotionState(ECognitiveLocomotionState State);

    UFUNCTION(BlueprintPure, Category="Cognitive|BoneDriver")
    bool HasValidResponse() const { return LatestResponse.bValid; }

    UFUNCTION(BlueprintPure, Category="Cognitive|BoneDriver")
    float GetLastConfidence() const { return LatestResponse.Confidence; }

    UFUNCTION(BlueprintPure, Category="Cognitive|BoneDriver")
    FString GetDiagnostics() const;

    // ── UActorComponent ───────────────────────────────────────────────────────
    virtual void BeginPlay() override;
    virtual void TickComponent(float DeltaTime, ELevelTick TickType,
                                FActorComponentTickFunction* ThisTickFunction) override;
    virtual void EndPlay(const EEndPlayReason::Type Reason) override;

private:
    FCognitiveBoneFrame  BuildNPCFrame() const;
    void                 ApplyBoneTransforms(const FCognitiveBoneResponse& Response);
    FString              GetObservationStateString() const;

    TWeakObjectPtr<UCognitiveInferenceSubsystem> InferenceSubsystem;
    TWeakObjectPtr<UCognitiveAnimInstance>       CachedAnimInstance;
    TWeakObjectPtr<USkeletalMeshComponent>       CachedMesh;
    TWeakObjectPtr<class UCognitiveLeaderObserverComponent> CachedLeaderObserver;
    TArray<FName>                                CachedBoneNames;
    FCognitiveBoneResponse                       LatestResponse;
    bool bJumpTriggered = false;  // impede chamadas múltiplas de Jump() por tick

    float   SendAccumulator  = 0.f;
    float   StatusPanelAccumulator = 0.f;  // acumula tempo p/ o painel de status periódico
    float   SendInterval     = 1.f / 30.f;
    double  LastRequestTime  = 0.0;
    int64   NextSequenceId   = 0;
};
