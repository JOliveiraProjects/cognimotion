#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "CognitiveMotionTypes.h"
#include "CognitiveRuntimePoseMemory.generated.h"

UCLASS()
class COGNITIVEMOTIONINTELLIGENCE_API UCognitiveRuntimePoseMemory : public UObject
{
    GENERATED_BODY()

public:
    void Initialize(int32 Capacity, int32 EmbeddingDim);

    void Store(const FCognitiveGeneratedMotionFragment& Fragment);

    TArray<FCognitiveGeneratedMotionFragment> QuerySimilar(
        const FCognitiveMotionEmbedding& Query,
        int32 K = 8,
        float MinSimilarity = 0.5f,
        ECognitiveMotionStyle StyleFilter = ECognitiveMotionStyle::Neutral,
        bool bUseStyleFilter = false) const;

    FCognitiveGeneratedMotionFragment BlendFragments(
        const TArray<FCognitiveGeneratedMotionFragment>& Fragments,
        const TArray<float>& Weights) const;

    void SaveToDisk(const FString& FilePath) const;
    bool LoadFromDisk(const FString& FilePath);

    UFUNCTION(BlueprintPure, Category = "Cognitive|Memory")
    int32 GetSize() const;

    UFUNCTION(BlueprintPure, Category = "Cognitive|Memory")
    float GetAverageQuality() const;

    UFUNCTION(BlueprintCallable, Category = "Cognitive|Memory")
    void Clear();

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cognitive|Memory")
    int32 MaxCapacity = 10000;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cognitive|Memory")
    float PruneQualityThreshold = 0.1f;

private:
    float CosineSimilarity(const TArray<float>& A, const TArray<float>& B) const;
    void PruneIfNeeded();

    TArray<FCognitiveGeneratedMotionFragment> Entries;
    mutable FRWLock MemoryLock;
    int32 EmbeddingDimension = 256;
};
