#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "CognitiveMotionTypes.h"
#include "CognitiveMotionDebugVisualizerComponent.generated.h"

UCLASS(ClassGroup=(Cognitive), meta=(BlueprintSpawnableComponent))
class COGNITIVEMOTIONINTELLIGENCEEDITOR_API UCognitiveMotionDebugVisualizerComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    UCognitiveMotionDebugVisualizerComponent();

    virtual void TickComponent(float DeltaTime, ELevelTick TickType,
        FActorComponentTickFunction* ThisTickFunction) override;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Debug|Trajectory")
    bool bShowFutureTrajectory = true;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Debug|Trajectory")
    bool bShowPastTrajectory = true;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Debug|Trajectory")
    FColor FutureTrajectoryColor = FColor::Cyan;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Debug|Trajectory")
    FColor PastTrajectoryColor = FColor(100, 100, 100);

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Debug|Embedding")
    bool bShowEmbeddingConfidence = true;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Debug|Embedding")
    bool bShowMotionStyle = true;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Debug|State")
    bool bShowNPCState = true;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Debug|State")
    bool bShowFallbackState = true;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Debug|Latency")
    bool bShowLatency = true;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Debug|Quality")
    bool bShowFootSliding = true;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Debug")
    float DrawDuration = 0.f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Debug")
    float TrajectoryLineThickness = 2.f;

private:
    void DrawTrajectory(const FCognitiveTrajectory& Traj, const FColor& Color) const;
    void DrawStateText() const;
    void DrawConfidenceBar() const;
    void DrawFootSlidingIndicator() const;
};
