#pragma once

#include "CoreMinimal.h"
#include "CognitiveBehaviorTypes.h"
#include "CognitiveBoneTypes.generated.h"

// ─────────────────────────────────────────────────────────────────────────────
// FCognitiveBoneFrame — frame enviado do NPC ao Python
// Todos os bone transforms do skeleton + contexto de comportamento
// ─────────────────────────────────────────────────────────────────────────────
USTRUCT(BlueprintType)
struct COGNITIVEMOTIONINTELLIGENCE_API FCognitiveBoneFrame
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadWrite, Category = "Cognitive|Bone")
    TArray<FName> BoneNames;

    UPROPERTY(BlueprintReadWrite, Category = "Cognitive|Bone")
    TArray<FTransform> BoneTransforms;

    UPROPERTY(BlueprintReadWrite, Category = "Cognitive|Bone")
    double Timestamp = 0.0;

    UPROPERTY(BlueprintReadWrite, Category = "Cognitive|Bone")
    int64 SequenceId = 0;

    UPROPERTY(BlueprintReadWrite, Category = "Cognitive|Bone")
    FCognitiveBehaviorContext BehaviorContext;

    bool IsValid() const
    {
        return BoneNames.Num() > 0 && BoneNames.Num() == BoneTransforms.Num();
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// FCognitiveBoneResponse — resposta do Python com bone transforms para o NPC
// ─────────────────────────────────────────────────────────────────────────────
USTRUCT(BlueprintType)
struct COGNITIVEMOTIONINTELLIGENCE_API FCognitiveBoneResponse
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadWrite, Category = "Cognitive|Bone")
    TArray<FTransform> BoneTransforms;

    UPROPERTY(BlueprintReadWrite, Category = "Cognitive|Bone")
    bool bValid = false;

    UPROPERTY(BlueprintReadWrite, Category = "Cognitive|Bone")
    float Confidence = 0.f;

    UPROPERTY(BlueprintReadWrite, Category = "Cognitive|Bone")
    float LatencyMs = 0.f;

    UPROPERTY(BlueprintReadWrite, Category = "Cognitive|Bone")
    int64 SequenceId = -1;

    UPROPERTY(BlueprintReadWrite, Category = "Cognitive|Bone")
    bool bApplyRootMotion = false;

    UPROPERTY(BlueprintReadWrite, Category = "Cognitive|Bone")
    FVector RootLocation = FVector::ZeroVector;

    UPROPERTY(BlueprintReadWrite, Category = "Cognitive|Bone")
    FQuat RootRotation = FQuat::Identity;
};

// Delegate — declarado depois dos structs que usa como parâmetro
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(
    FCognitiveBoneResponseDelegate,
    FCognitiveBehaviorContext, BehaviorContext
);
