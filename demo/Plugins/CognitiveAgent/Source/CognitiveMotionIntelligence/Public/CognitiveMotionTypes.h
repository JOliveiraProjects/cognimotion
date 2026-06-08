#pragma once

#include "CoreMinimal.h"
#include "Engine/EngineTypes.h"
#include "Animation/AnimationAsset.h"
#include "CognitiveMotionTypes.generated.h"

UENUM(BlueprintType)
enum class ECognitiveNPCState : uint8
{
    Idle,
    CasualMovement,
    Alert,
    Investigate,
    Combat,
    Stealth,
    Flee,
    Hide,
    TakeCover,
    Surrender,
    Healing,
    Incapacitated,
    Dead,
    React,
    ContextualAction,
    Mounted,
    Driving
};

UENUM(BlueprintType)
enum class ECognitiveMovementMode : uint8
{
    Idle,
    Walk,
    Jog,
    Sprint,
    Crouch,
    Prone,
    Swim,
    Fly,
    Mounted
};

UENUM(BlueprintType)
enum class ECognitivePosture : uint8
{
    Standing,
    Crouching,
    Prone
};

UENUM(BlueprintType)
enum class ECognitiveEmotionalState : uint8
{
    Calm,
    Nervous,
    Panicked,
    Angry,
    Fearful,
    Confident,
    Suspicious,
    Alert,
    Drunk
};

UENUM(BlueprintType)
enum class ECognitiveMotionStyle : uint8
{
    Neutral,
    Aggressive,
    Relaxed,
    Injured,
    Fatigued,
    Stealth,
    Military,
    Civilian,
    Criminal
};

UENUM(BlueprintType)
enum class ECognitiveSensorEvent : uint8
{
    None,
    VisualContact,
    HearingEvent,
    TactileEvent,
    AllyDeathNearby,
    ExplosionNearby,
    ProjectileNearby,
    VehicleNearby
};

USTRUCT(BlueprintType)
struct COGNITIVEMOTIONINTELLIGENCE_API FCognitiveBlackboard
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadWrite) float Health = 100.f;
    UPROPERTY(BlueprintReadWrite) float Stamina = 100.f;
    UPROPERTY(BlueprintReadWrite) float Alertness = 0.f;
    UPROPERTY(BlueprintReadWrite) float FearLevel = 0.f;
    UPROPERTY(BlueprintReadWrite) float AggressionLevel = 0.f;
    UPROPERTY(BlueprintReadWrite) float Fatigue = 0.f;
    UPROPERTY(BlueprintReadWrite) float ThreatLevel = 0.f;
    UPROPERTY(BlueprintReadWrite) float SuspicionTimer = 0.f;
    UPROPERTY(BlueprintReadWrite) float TimeSinceLastThreat = 0.f;
    UPROPERTY(BlueprintReadWrite) int32 AmmoCount = 0;
    UPROPERTY(BlueprintReadWrite) bool bIsInCover = false;
    UPROPERTY(BlueprintReadWrite) bool bIsAiming = false;
    UPROPERTY(BlueprintReadWrite) bool bIsReloading = false;
    UPROPERTY(BlueprintReadWrite) FVector LastKnownTargetPosition = FVector::ZeroVector;
    UPROPERTY(BlueprintReadWrite) ECognitiveNPCState CurrentState = ECognitiveNPCState::Idle;
    UPROPERTY(BlueprintReadWrite) ECognitivePosture Posture = ECognitivePosture::Standing;
    UPROPERTY(BlueprintReadWrite) ECognitiveEmotionalState EmotionalState = ECognitiveEmotionalState::Calm;
    UPROPERTY(BlueprintReadWrite) ECognitiveMovementMode MovementMode = ECognitiveMovementMode::Idle;
    UPROPERTY(BlueprintReadWrite) ECognitiveMotionStyle MotionStyle = ECognitiveMotionStyle::Neutral;
};

USTRUCT(BlueprintType)
struct COGNITIVEMOTIONINTELLIGENCE_API FCognitiveTrajectorySample
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadWrite) FVector Position = FVector::ZeroVector;
    UPROPERTY(BlueprintReadWrite) FVector LinearVelocity = FVector::ZeroVector;
    UPROPERTY(BlueprintReadWrite) FVector AngularVelocity = FVector::ZeroVector;
    UPROPERTY(BlueprintReadWrite) FQuat Facing = FQuat::Identity;
    UPROPERTY(BlueprintReadWrite) float TimeInSeconds = 0.f;
    UPROPERTY(BlueprintReadWrite) float Speed = 0.f;
};

USTRUCT(BlueprintType)
struct COGNITIVEMOTIONINTELLIGENCE_API FCognitiveTrajectory
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadWrite) TArray<FCognitiveTrajectorySample> Samples;

    bool IsValid() const { return Samples.Num() > 0; }
    void Reset() { Samples.Reset(); }
};

USTRUCT(BlueprintType)
struct COGNITIVEMOTIONINTELLIGENCE_API FCognitivePoseFrame
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadWrite) double Timestamp = 0.0;
    UPROPERTY(BlueprintReadWrite) FVector LinearVelocity = FVector::ZeroVector;
    UPROPERTY(BlueprintReadWrite) FVector AngularVelocity = FVector::ZeroVector;
    UPROPERTY(BlueprintReadWrite) FVector RootLocation = FVector::ZeroVector;
    UPROPERTY(BlueprintReadWrite) FQuat RootRotation = FQuat::Identity;
    UPROPERTY(BlueprintReadWrite) TArray<FTransform> BoneTransforms;
    UPROPERTY(BlueprintReadWrite) FCognitiveTrajectory PastTrajectory;
    UPROPERTY(BlueprintReadWrite) FCognitiveTrajectory FutureTrajectory;
    UPROPERTY(BlueprintReadWrite) TMap<FName, float> CurveValues;
    UPROPERTY(BlueprintReadWrite) TArray<FName> Tags;
    UPROPERTY(BlueprintReadWrite) ECognitiveMovementMode MovementMode = ECognitiveMovementMode::Idle;
    UPROPERTY(BlueprintReadWrite) ECognitiveMotionStyle MotionStyle = ECognitiveMotionStyle::Neutral;
    UPROPERTY(BlueprintReadWrite) int32 FrameIndex = 0;
};

USTRUCT(BlueprintType)
struct COGNITIVEMOTIONINTELLIGENCE_API FCognitiveMotionEmbedding
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadWrite) TArray<float> Values;
    UPROPERTY(BlueprintReadWrite) float Confidence = 0.f;
    UPROPERTY(BlueprintReadWrite) ECognitiveMotionStyle Style = ECognitiveMotionStyle::Neutral;
    UPROPERTY(BlueprintReadWrite) int64 SequenceId = -1;

    bool IsValid() const { return Values.Num() == 256 && Confidence > 0.f; }
};

USTRUCT(BlueprintType)
struct COGNITIVEMOTIONINTELLIGENCE_API FCognitiveGeneratedMotionFragment
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadWrite) FCognitivePoseFrame PoseFrame;
    UPROPERTY(BlueprintReadWrite) FCognitiveMotionEmbedding Embedding;
    UPROPERTY(BlueprintReadWrite) float SimilarityScore = 0.f;
    UPROPERTY(BlueprintReadWrite) float QualityScore = 0.f;
    UPROPERTY(BlueprintReadWrite) double GeneratedAt = 0.0;
};

USTRUCT(BlueprintType)
struct COGNITIVEMOTIONINTELLIGENCE_API FCognitiveMotionIdentity
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadWrite) ECognitiveMotionStyle Style = ECognitiveMotionStyle::Neutral;
    UPROPERTY(BlueprintReadWrite) float StyleBlendWeight = 1.f;
    UPROPERTY(BlueprintReadWrite) float ConfidenceThreshold = 0.5f;
    UPROPERTY(BlueprintReadWrite) float EntropyWeight = 0.1f;
    UPROPERTY(BlueprintReadWrite) TArray<float> StyleEmbedding;
};

USTRUCT(BlueprintType)
struct COGNITIVEMOTIONINTELLIGENCE_API FCognitiveMotionQualityMetrics
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadWrite) float FootSliding = 0.f;
    UPROPERTY(BlueprintReadWrite) float Smoothness = 1.f;
    UPROPERTY(BlueprintReadWrite) float ImitationScore = 0.f;
    UPROPERTY(BlueprintReadWrite) float TrajectoryError = 0.f;
    UPROPERTY(BlueprintReadWrite) float VelocityError = 0.f;
    UPROPERTY(BlueprintReadWrite) float PredictionError = 0.f;
    UPROPERTY(BlueprintReadWrite) float Confidence = 0.f;
    UPROPERTY(BlueprintReadWrite) float LatencyMs = 0.f;
};

USTRUCT(BlueprintType)
struct COGNITIVEMOTIONINTELLIGENCE_API FCognitiveMotionRequest
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadWrite) int64 SequenceId = 0;
    UPROPERTY(BlueprintReadWrite) FCognitivePoseFrame CurrentPose;
    UPROPERTY(BlueprintReadWrite) FCognitiveTrajectory DesiredTrajectory;
    UPROPERTY(BlueprintReadWrite) FCognitiveBlackboard Blackboard;
    UPROPERTY(BlueprintReadWrite) ECognitiveMotionStyle RequestedStyle = ECognitiveMotionStyle::Neutral;
    UPROPERTY(BlueprintReadWrite) float MaxLatencyMs = 80.f;
    // BehaviorContext (Mode/Type) é enviado separadamente via FCognitiveBoneFrame
    // pelo UCognitiveNPCBoneDriver — evita dependência circular de headers.
};

// Estado físico do NPC decidido pelo Python (espelha PHYSICAL_STATE_MAP).
// Diz à AnimInstance/BoneDriver qual animação tocar.
UENUM(BlueprintType)
enum class ECognitivePhysicalState : uint8
{
    Alive    = 0  UMETA(DisplayName = "Alive"),
    Dead     = 1  UMETA(DisplayName = "Dead (morte)"),
    Falling  = 2  UMETA(DisplayName = "Falling (caindo)"),
    Swimming = 3  UMETA(DisplayName = "Swimming (nadando)"),
    Landing  = 4  UMETA(DisplayName = "Landing (pouso)"),
    Attack   = 5  UMETA(DisplayName = "Attack (atacar)"),
    Flee     = 6  UMETA(DisplayName = "Flee (fugir)"),
    Hide     = 7  UMETA(DisplayName = "Hide (esconder)"),
    PickUp   = 8  UMETA(DisplayName = "PickUp (pegar)"),
    Enter    = 9  UMETA(DisplayName = "Enter (entrar em veículo)"),
};

USTRUCT(BlueprintType)
struct COGNITIVEMOTIONINTELLIGENCE_API FCognitiveMotionResponse
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadWrite) int64 SequenceId = -1;
    UPROPERTY(BlueprintReadWrite) FCognitiveMotionEmbedding Embedding;
    UPROPERTY(BlueprintReadWrite) FCognitiveTrajectory RefinedTrajectory;
    UPROPERTY(BlueprintReadWrite) ECognitiveMotionStyle SelectedStyle = ECognitiveMotionStyle::Neutral;
    UPROPERTY(BlueprintReadWrite) bool bValid = false;
    UPROPERTY(BlueprintReadWrite) float LatencyMs = 0.f;

    // Bone transforms gerados pelo Python para TODOS os bones do skeleton do NPC.
    // O AnimNode aplica estes transforms diretamente — sem Motion Matching, sem PoseSearch.
    UPROPERTY(BlueprintReadWrite) TArray<FTransform> BoneTransforms;

    // Estado físico decidido pelo Python (vida/morte/queda/natação).
    UPROPERTY(BlueprintReadWrite) ECognitivePhysicalState PhysicalState = ECognitivePhysicalState::Alive;
};

USTRUCT(BlueprintType)
struct COGNITIVEMOTIONINTELLIGENCE_API FCognitiveSensorData
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadWrite) ECognitiveSensorEvent EventType = ECognitiveSensorEvent::None;
    UPROPERTY(BlueprintReadWrite) FVector EventLocation = FVector::ZeroVector;
    UPROPERTY(BlueprintReadWrite) float Intensity = 0.f;
    UPROPERTY(BlueprintReadWrite) float Distance = 0.f;
    UPROPERTY(BlueprintReadWrite) AActor* SourceActor = nullptr;
    UPROPERTY(BlueprintReadWrite) double Timestamp = 0.0;
};

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FCognitiveMotionResponseDelegate, const FCognitiveMotionResponse&, Response);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FCognitiveSensorEventDelegate, const FCognitiveSensorData&, SensorData);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FCognitiveStateChangeDelegate, ECognitiveNPCState, NewState);
