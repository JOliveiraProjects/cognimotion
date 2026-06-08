#pragma once

#include "CoreMinimal.h"
#include "Animation/AnimInstance.h"
#include "CognitiveMotionTypes.h"
#include "GameFramework/Character.h"
#include "GameFramework/CharacterMovementComponent.h"
#include "CognitiveAnimInstance.generated.h"

UCLASS()
class COGNITIVEMOTIONINTELLIGENCE_API UCognitiveAnimInstance : public UAnimInstance
{
    GENERATED_BODY()

public:
    virtual void NativeInitializeAnimation() override;
    virtual void NativeUpdateAnimation(float DeltaSeconds) override;
    virtual void NativeThreadSafeUpdateAnimation(float DeltaSeconds) override;

    /**
     * Helper: returns this AnimInstance cast to UCognitiveAnimInstance.
     * Use this in Blueprints instead of "Cast To CognitiveAnimInstance" on a Pawn/Character,
     * which will always fail because AnimInstance does not inherit from Pawn.
     *
     * Correct Blueprint pattern:
     *   GetMesh -> GetAnimInstance -> Cast To CognitiveAnimInstance
     *   OR simply call GetCognitiveAnimInstance() on any actor that has a
     *   UCognitiveMotionLearnerComponent.
     */
    UFUNCTION(BlueprintPure, Category = "Cognitive|Anim",
              meta = (DefaultToSelf = "Target", HidePin = "Target"))
    static UCognitiveAnimInstance* GetCognitiveAnimInstance(const ACharacter* Character);

    UFUNCTION(BlueprintPure, Category = "Cognitive|Anim", meta = (BlueprintThreadSafe))
    FCognitiveBlackboard GetBlackboard() const
    {
        FRWScopeLock L(BlackboardLock, SLT_ReadOnly);
        return ThreadSafeBlackboard;
    }

    UFUNCTION(BlueprintPure, Category = "Cognitive|Anim", meta = (BlueprintThreadSafe))
    FCognitiveTrajectory GetFutureTrajectory() const
    {
        FRWScopeLock L(TrajectoryLock, SLT_ReadOnly);
        return ThreadSafeTrajectory;
    }

    UFUNCTION(BlueprintPure, Category = "Cognitive|Anim", meta = (BlueprintThreadSafe))
    FCognitiveMotionEmbedding GetCurrentEmbedding() const
    {
        FRWScopeLock L(EmbeddingLock, SLT_ReadOnly);
        return ThreadSafeEmbedding;
    }

    UFUNCTION(BlueprintPure, Category = "Cognitive|Anim", meta = (BlueprintThreadSafe))
    ECognitiveNPCState GetNPCState() const
    {
        FRWScopeLock L(BlackboardLock, SLT_ReadOnly);
        return ThreadSafeBlackboard.CurrentState;
    }

    UFUNCTION(BlueprintPure, Category = "Cognitive|Anim", meta = (BlueprintThreadSafe))
    ECognitiveMovementMode GetMovementMode() const
    {
        FRWScopeLock L(BlackboardLock, SLT_ReadOnly);
        return ThreadSafeBlackboard.MovementMode;
    }

    UFUNCTION(BlueprintPure, Category = "Cognitive|Anim", meta = (BlueprintThreadSafe))
    ECognitiveMotionStyle GetMotionStyle() const
    {
        FRWScopeLock L(BlackboardLock, SLT_ReadOnly);
        return ThreadSafeBlackboard.MotionStyle;
    }

    UFUNCTION(BlueprintPure, Category = "Cognitive|Anim", meta = (BlueprintThreadSafe))
    float GetSpeed() const
    {
        FRWScopeLock L(VelocityLock, SLT_ReadOnly);
        return ThreadSafeSpeed;
    }

    UFUNCTION(BlueprintPure, Category = "Cognitive|Anim", meta = (BlueprintThreadSafe))
    float GetEmbeddingConfidence() const
    {
        FRWScopeLock L(EmbeddingLock, SLT_ReadOnly);
        return ThreadSafeEmbedding.Confidence;
    }

    UFUNCTION(BlueprintPure, Category = "Cognitive|Anim", meta = (BlueprintThreadSafe))
    bool HasValidEmbedding() const
    {
        FRWScopeLock L(EmbeddingLock, SLT_ReadOnly);
        return ThreadSafeEmbedding.IsValid();
    }

    UFUNCTION(BlueprintPure, Category = "Cognitive|Anim", meta = (BlueprintThreadSafe))
    FVector GetRootVelocity() const
    {
        FRWScopeLock L(VelocityLock, SLT_ReadOnly);
        return ThreadSafeRootVelocity;
    }

    UFUNCTION(BlueprintPure, Category = "Cognitive|Anim", meta = (BlueprintThreadSafe))
    float GetBlendWeight() const
    {
        FRWScopeLock L(VelocityLock, SLT_ReadOnly);
        return ThreadSafeBlendWeight;
    }

    void SetBlackboard(const FCognitiveBlackboard& InBB);
    void SetTrajectory(const FCognitiveTrajectory& InTraj);
    void SetEmbedding(const FCognitiveMotionEmbedding& InEmbedding);
    // ── Bone Transforms do Python ─────────────────────────────────────────────
    // Python devolve os bone transforms de TODOS os bones do skeleton.
    // O AnimNode aplica esses transforms diretamente no skeleton do NPC.
    void SetBoneTransforms(const TArray<FTransform>& InTransforms)
    {
        FRWScopeLock Lock(BoneTransformsLock, SLT_Write);
        PythonBoneTransforms = InTransforms;
        bHasPythonBoneTransforms = InTransforms.Num() > 0;
    }

    TArray<FTransform> GetBoneTransforms() const
    {
        FRWScopeLock Lock(BoneTransformsLock, SLT_ReadOnly);
        return PythonBoneTransforms;
    }

    void ClearBoneTransforms()
    {
        FRWScopeLock Lock(BoneTransformsLock, SLT_Write);
        PythonBoneTransforms.Reset();
        bHasPythonBoneTransforms = false;
    }

    bool HasValidBoneTransforms() const
    {
        FRWScopeLock Lock(BoneTransformsLock, SLT_ReadOnly);
        return bHasPythonBoneTransforms;
    }

    // Modo de aprendizagem: true = NPC observa líder, false = Python dirige o NPC
    UPROPERTY(BlueprintReadWrite, Category = "Cognitive|Behavior")
    bool bLearningMode = true;

    // ── ThreadSafeBlendWeight protegido por VelocityLock ─────────────────────
    void SetBlendWeight(float InWeight);

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Cognitive|Debug")
    FCognitiveMotionQualityMetrics QualityMetrics;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Cognitive|Debug")
    float InferenceLatencyMs = 0.f;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Cognitive|Debug")
    bool bInferenceFallbackActive = false;

private:
    void CacheOwnerData();
    void UpdateQualityMetrics(float DeltaSeconds);
    void SampleFootPositions(FVector& OutLeftFoot, FVector& OutRightFoot) const;

    FCognitiveBlackboard       ThreadSafeBlackboard;
    FCognitiveTrajectory       ThreadSafeTrajectory;
    FCognitiveMotionEmbedding  ThreadSafeEmbedding;

    // mutable: permite locking em métodos const (getters thread-safe)
    mutable FRWLock BlackboardLock;
    mutable FRWLock TrajectoryLock;
    mutable FRWLock EmbeddingLock;
    mutable FRWLock VelocityLock;
    mutable FRWLock BoneTransformsLock;  // protege PythonBoneTransforms

    TArray<FTransform> PythonBoneTransforms;
    bool               bHasPythonBoneTransforms = false;

    float   ThreadSafeSpeed = 0.f;
    float   ThreadSafeBlendWeight = 1.f;
    FVector ThreadSafeRootVelocity = FVector::ZeroVector;

    FVector PrevLeftFootPos  = FVector::ZeroVector;
    FVector PrevRightFootPos = FVector::ZeroVector;
    float   AccumulatedFootSliding = 0.f;
    int32   FootSlidingFrames = 0;

    TWeakObjectPtr<ACharacter> OwnerCharacter;
    TWeakObjectPtr<UCharacterMovementComponent> OwnerMovement;
};
