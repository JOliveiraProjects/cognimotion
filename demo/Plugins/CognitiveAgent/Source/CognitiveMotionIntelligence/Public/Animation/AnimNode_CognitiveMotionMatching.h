#pragma once
#include "CoreMinimal.h"
#include "Animation/AnimNodeBase.h"
#include "BonePose.h"
#include "CognitiveMotionTypes.h"
#include "CognitiveBehaviorTypes.h"
#include "AnimNode_CognitiveMotionMatching.generated.h"

// AnimGraph: [CognitiveMotionMatching] → [Output Pose]  (nó único, sem entradas)
// Aplica os bone transforms recebidos do Python diretamente no skeleton do NPC.
// Converte corretamente Component Space (GetComponentSpaceTransforms) → Local Space (Output.Pose).
USTRUCT(BlueprintInternalUseOnly)
struct COGNITIVEMOTIONINTELLIGENCE_API FAnimNode_CognitiveMotionMatching : public FAnimNode_Base
{
    GENERATED_BODY()

    // Pose de entrada (locomoção da AnimBP). Quando NÃO há bones do Python
    // (ex.: modo Inferring, onde o Python só decide a ação), esta pose passa
    // direto — assim a locomoção (walk/run) anima as pernas e o NPC não desliza.
    UPROPERTY(EditAnywhere, Category = "CognitiveMotion")
    FPoseLink SourcePose;

    UPROPERTY(EditAnywhere, Category = "CognitiveMotion")
    float BoneBlendSpeed = 12.f;

    UPROPERTY(EditAnywhere, Category = "CognitiveMotion")
    bool bHoldLastPoseOnFallback = true;

    UPROPERTY(VisibleAnywhere, Category = "CognitiveMotion|Debug")
    FString DiagnosticReason;

    UPROPERTY(VisibleAnywhere, Category = "CognitiveMotion|Debug")
    float CurrentConfidence = 0.f;

    UPROPERTY(VisibleAnywhere, Category = "CognitiveMotion|Debug")
    int32 BoneCount = 0;

    virtual void Initialize_AnyThread(const FAnimationInitializeContext& Context) override;
    virtual void CacheBones_AnyThread(const FAnimationCacheBonesContext& Context) override;
    virtual void Update_AnyThread(const FAnimationUpdateContext& Context) override;
    virtual void Evaluate_AnyThread(FPoseContext& Output) override;
    virtual bool HasPreUpdate() const override { return true; }
    virtual void PreUpdate(const UAnimInstance* InAnimInstance) override;

private:
    // Bone transforms em component space recebidos do Python (via AnimInstance)
    TArray<FTransform> CachedBoneTransforms;

    // Bone transforms em LOCAL space após conversão CS→LS pelo FCSPose
    TArray<FTransform> CurrentLocalTransforms;
    TArray<FTransform> LastValidLocalTransforms;

    bool  bHasValidTransforms   = false;
    bool  bInferenceFallback    = false;
    bool  bLocalTransformsReady = false;
    float CachedDeltaTime       = 0.f;
};
