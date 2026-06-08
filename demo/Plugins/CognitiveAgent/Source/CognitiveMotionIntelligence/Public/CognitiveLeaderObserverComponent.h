#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "CognitiveMotionTypes.h"
#include "CognitiveMotionProtocol.h"
#include "CognitiveLeaderObserverComponent.generated.h"

class UCognitiveInferenceSubsystem;
class USkeletalMeshComponent;

UCLASS(ClassGroup=(Cognitive), meta=(BlueprintSpawnableComponent),
       DisplayName="Cognitive Leader Observer")
class COGNITIVEMOTIONINTELLIGENCE_API UCognitiveLeaderObserverComponent
    : public UActorComponent
{
    GENERATED_BODY()

public:
    UCognitiveLeaderObserverComponent();

    // TargetLeader: sete na instância do level (selecione o NPC no viewport → Details Panel).
    // EditInstanceOnly = só aceita no Details Panel da INSTÂNCIA no level, não nos Class Defaults.
    // Se deixar null, o componente usa automaticamente PlayerPawn(0) no BeginPlay.
    UPROPERTY(EditInstanceOnly, BlueprintReadWrite, Category="Cognitive|Leader|Setup",
              meta=(DisplayName="Target Leader (deixe null para usar PlayerPawn automático)"))
    AActor* TargetLeader = nullptr;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Leader|Setup")
    int64 LeaderNPCId = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Leader|Setup")
    int64 FollowerNPCId = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Leader|Timing",
              meta=(ClampMin="1.0", ClampMax="120.0"))
    float SamplingRate = 30.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Leader|Timing",
              meta=(ClampMin="0.5", ClampMax="10.0"))
    float SequenceIntervalSeconds = 0.1f;  // bones chegam em 100ms

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Leader|Timing",
              meta=(ClampMin="4", ClampMax="120"))
    int32 MaxFramesPerSequence = 30;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Leader|Setup")
    bool bEnabled = true;

    // BC-03 FIX: Lista de bones a capturar — deve ser idêntica à lista do
    // UCognitivePoseRecorderComponent para que os BoneTransforms de líder e
    // seguidor sejam estruturalmente compatíveis no servidor Python.
    // Default: os mesmos 9 bones do PoseRecorderComponent.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Leader|Setup")
    TArray<FName> BonesToCapture = {
        FName("pelvis"),   FName("spine_01"), FName("spine_02"),
        FName("spine_03"), FName("foot_l"),   FName("foot_r"),
        FName("hand_l"),   FName("hand_r"),   FName("head")
    };

    // SetTargetLeader: passa o ator líder (ex: via GetPlayerPawn(0) no BP_CognitiveNPC)
    UFUNCTION(BlueprintCallable, Category="Cognitive|Leader")
    void SetTargetLeader(AActor* NewLeader, int64 NewLeaderNPCId);

    // SetLeaderAsPlayer: atalho — define automaticamente o líder como PlayerPawn(PlayerIndex).
    // Use este no BeginPlay do BP_CognitiveNPC em vez de arrastar referências.
    UFUNCTION(BlueprintCallable, Category="Cognitive|Leader")
    void SetLeaderAsPlayer(int32 PlayerIndex = 0, int64 NewLeaderNPCId = 1001);

    UFUNCTION(BlueprintPure, Category="Cognitive|Leader")
    int32 GetBufferedFrameCount() const { return FrameBuffer.Num(); }

    UFUNCTION(BlueprintCallable, Category="Cognitive|Leader")
    void FlushSequence();

    UFUNCTION(BlueprintPure, Category="Cognitive|Leader")
    FString GetDiagnostics() const;

    virtual void BeginPlay()  override;
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
    virtual void TickComponent(float DeltaTime, ELevelTick TickType,
                               FActorComponentTickFunction* ThisTickFunction) override;

private:
    void CaptureLeaderFrame();
    bool FillPoseFrame(FCognitivePoseFrame& OutFrame);
    void ExtractBoneTransforms(USkeletalMeshComponent* SkeletalMesh,
                               FCognitivePoseFrame& OutFrame) const;
    void SendSequence();

    TArray<FCognitivePoseFrame> FrameBuffer;
    float CaptureAccumulator = 0.0f;
    float CaptureInterval   = 0.0f;
    float SendAccumulator   = 0.0f;
    int32 NextSequenceId    = 1;
    double BufferStartTimestamp = 0.0;

    UPROPERTY(Transient)
    TWeakObjectPtr<UCognitiveInferenceSubsystem> InferenceSubsystem;
    TWeakObjectPtr<class UCognitiveAnimInstance> CachedNPCAnimInstance;

    TWeakObjectPtr<USkeletalMeshComponent> CachedLeaderMesh;
    FVector LastLeaderLocation   = FVector::ZeroVector;
    FVector LeaderLinearVelocity = FVector::ZeroVector;
    bool bFirstCapture    = true;
    int32 TotalFramesCaptured = 0;
    int32 TotalSequencesSent  = 0;
    int32 FailedCaptures      = 0;
};
