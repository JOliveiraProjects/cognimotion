#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "CognitiveMotionTypes.h"
#include "CognitiveTrajectoryGenerator.generated.h"

USTRUCT(BlueprintType)
struct COGNITIVEMOTIONINTELLIGENCE_API FCognitiveTrajectoryConfig
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite) int32 PastSamples      = 6;
    UPROPERTY(EditAnywhere, BlueprintReadWrite) int32 FutureSamples    = 6;
    UPROPERTY(EditAnywhere, BlueprintReadWrite) float SampleInterval   = 0.05f;
    UPROPERTY(EditAnywhere, BlueprintReadWrite) float MaxPredictSpeed  = 600.f;
    UPROPERTY(EditAnywhere, BlueprintReadWrite) float SmoothingFactor  = 0.85f;
    UPROPERTY(EditAnywhere, BlueprintReadWrite) float TurnSmoothing    = 0.7f;
    UPROPERTY(EditAnywhere, BlueprintReadWrite) bool  bUsePythonRefined = true;
};

UCLASS(BlueprintType)
class COGNITIVEMOTIONINTELLIGENCE_API UCognitiveTrajectoryGenerator : public UObject
{
    GENERATED_BODY()

public:
    UFUNCTION(BlueprintCallable, Category = "Cognitive|Trajectory")
    void Initialize(const FCognitiveTrajectoryConfig& InConfig);

    UFUNCTION(BlueprintCallable, Category = "Cognitive|Trajectory")
    FCognitiveTrajectory GenerateFutureTrajectory(
        const FVector& CurrentLocation,
        const FVector& CurrentVelocity,
        const FQuat&   CurrentFacing,
        const FVector& DesiredDirection,
        float          DesiredSpeed,
        ECognitiveMovementMode MovementMode) const;

    UFUNCTION(BlueprintCallable, Category = "Cognitive|Trajectory")
    FCognitiveTrajectory BuildPastTrajectory(
        const TArray<FVector>& LocationHistory,
        const TArray<FVector>& VelocityHistory,
        const TArray<FQuat>&   FacingHistory,
        const TArray<float>&   TimeHistory) const;

    UFUNCTION(BlueprintCallable, Category = "Cognitive|Trajectory")
    FCognitiveTrajectory BlendWithPythonResponse(
        const FCognitiveTrajectory& Generated,
        const FCognitiveTrajectory& PythonRefined,
        float BlendAlpha) const;

    UFUNCTION(BlueprintCallable, Category = "Cognitive|Trajectory")
    FCognitiveTrajectory MakeIdleTrajectory() const;

    UFUNCTION(BlueprintCallable, Category = "Cognitive|Trajectory")
    void RecordFrame(const FVector& Location, const FVector& Velocity, const FQuat& Facing, double Timestamp);

    UFUNCTION(BlueprintCallable, Category = "Cognitive|Trajectory")
    FCognitiveTrajectory GetRecordedPastTrajectory() const;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cognitive|Trajectory")
    FCognitiveTrajectoryConfig Config;

private:
    FVector SmoothVector(const FVector& Current, const FVector& Target, float Alpha) const;
    FQuat   SmoothRotation(const FQuat& Current, const FQuat& Target, float Alpha) const;

    struct FHistoryFrame { FVector Location; FVector Velocity; FQuat Facing; double Time; };
    TArray<FHistoryFrame> FrameHistory;
    static constexpr int32 MaxHistory = 128;
    // TECH DEBT FIX: mutable permite uso de FScopeLock em métodos const (GetRecordedPastTrajectory)
    // sem const_cast, que era semanticamente confuso mesmo sendo tecnicamente válido.
    mutable FCriticalSection HistoryLock;
    // Ring buffer: HistoryHead é o próximo slot a escrever (O(1) RecordFrame).
    int32 HistoryHead = 0;
};
