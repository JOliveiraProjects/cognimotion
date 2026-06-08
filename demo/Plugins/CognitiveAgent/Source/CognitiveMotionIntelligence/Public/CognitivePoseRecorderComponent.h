#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "CognitiveMotionTypes.h"
#include "CognitiveAnimInstance.h"
#include "CognitivePoseRecorderComponent.generated.h"

UCLASS(ClassGroup=(Cognitive), meta=(BlueprintSpawnableComponent))
class COGNITIVEMOTIONINTELLIGENCE_API UCognitivePoseRecorderComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    UCognitivePoseRecorderComponent();

    virtual void BeginPlay() override;
    virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

    UFUNCTION(BlueprintCallable, Category = "Cognitive|Pose")
    void StartRecording();

    UFUNCTION(BlueprintCallable, Category = "Cognitive|Pose")
    void StopRecording();

    UFUNCTION(BlueprintPure, Category = "Cognitive|Pose")
    bool IsRecording() const { return bIsRecording; }

    UFUNCTION(BlueprintCallable, Category = "Cognitive|Pose")
    bool GetLatestFrame(FCognitivePoseFrame& OutFrame) const;

    UFUNCTION(BlueprintCallable, Category = "Cognitive|Pose")
    void GetRecentFrames(int32 Count, TArray<FCognitivePoseFrame>& OutFrames) const;

    UFUNCTION(BlueprintCallable, Category = "Cognitive|Pose")
    void BuildTrajectoryFromBuffer(FCognitiveTrajectory& OutPast, FCognitiveTrajectory& OutFuture, int32 PastSamples = 6, int32 FutureSamples = 6) const;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cognitive|Pose")
    float SamplingRate = 30.f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cognitive|Pose")
    int32 BufferCapacity = 256;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cognitive|Pose")
    bool bCaptureBoneTransforms = true;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cognitive|Pose")
    TArray<FName> BonesToCapture;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Cognitive|Debug")
    int32 TotalFramesCaptured = 0;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Cognitive|Debug")
    float ActualSamplingRate = 0.f;

private:
    void CaptureFrame();
    void PopulateFrameBones(FCognitivePoseFrame& Frame, const USkeletalMeshComponent* Mesh) const;
    void PushFrame(FCognitivePoseFrame&& Frame);

    TArray<FCognitivePoseFrame>  CircularBuffer;
    int32                        BufferHead = 0;
    int32                        BufferSize = 0;
    mutable FRWLock              BufferLock;

    bool  bIsRecording     = false;
    float SampleAccumulator = 0.f;
    float SampleInterval    = 1.f / 30.f;
    // BM-08 FIX: double em vez de float — evita cancelamento catastrófico de
    // ponto flutuante ao calcular DT com timestamps de wall-clock grandes (1.7e9+s).
    double LastSampleTime    = 0.0;

    TWeakObjectPtr<USkeletalMeshComponent> CachedMesh;
    TWeakObjectPtr<UCognitiveAnimInstance> CachedAnimInstance;
    int32 FrameIndex = 0;

    FVector PrevLocation  = FVector::ZeroVector;
    FQuat   PrevRotation  = FQuat::Identity;
    double  PrevSampleTime = 0.0;  // BM-08 FIX: double para evitar perda de precisão
};
