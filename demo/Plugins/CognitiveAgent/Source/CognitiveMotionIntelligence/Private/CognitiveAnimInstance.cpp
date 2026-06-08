#include "CognitiveAnimInstance.h"
#include "GameFramework/Character.h"
#include "GameFramework/CharacterMovementComponent.h"
#include "Components/SkeletalMeshComponent.h"
#include "Animation/AnimInstanceProxy.h"
#include "DrawDebugHelpers.h"

UCognitiveAnimInstance* UCognitiveAnimInstance::GetCognitiveAnimInstance(const ACharacter* Character)
{
    if (!Character) return nullptr;
    if (const USkeletalMeshComponent* Mesh = Character->GetMesh())
        return Cast<UCognitiveAnimInstance>(Mesh->GetAnimInstance());
    return nullptr;
}

void UCognitiveAnimInstance::NativeInitializeAnimation()
{
    Super::NativeInitializeAnimation();
    CacheOwnerData();
}

void UCognitiveAnimInstance::CacheOwnerData()
{
    if (AActor* Owner = GetOwningActor())
    {
        OwnerCharacter = Cast<ACharacter>(Owner);
        if (OwnerCharacter.IsValid())
            OwnerMovement = OwnerCharacter->GetCharacterMovement();
    }
}

void UCognitiveAnimInstance::NativeUpdateAnimation(float DeltaSeconds)
{
    Super::NativeUpdateAnimation(DeltaSeconds);

    if (!OwnerCharacter.IsValid())
    {
        CacheOwnerData();
        return;
    }

    const FVector Velocity = OwnerMovement.IsValid()
        ? OwnerMovement->Velocity
        : FVector::ZeroVector;

    // BM-04 FIX: ThreadSafeSpeed era escrito sem lock de ambos os threads.
    // Agora usa SpeedLock (FRWLock) para escrita segura.
    {
        FRWScopeLock Lock(VelocityLock, SLT_Write);
        ThreadSafeRootVelocity = Velocity;
        // BM-04: Speed também protegido pelo mesmo lock de velocidade
        ThreadSafeSpeed = (float)Velocity.Size2D();
    }

    // BM-03 FIX: NativeUpdateAnimation e NativeThreadSafeUpdateAnimation escreviam
    // MovementMode com lógicas de threshold diferentes (280 vs 200/400 cm/s),
    // causando valor não-determinístico. Removida a escrita de MovementMode daqui —
    // toda a lógica de MovementMode fica centralizada em NativeThreadSafeUpdateAnimation
    // onde pode ler ThreadSafeRootVelocity de forma sincronizada.

    UpdateQualityMetrics(DeltaSeconds);
}

void UCognitiveAnimInstance::NativeThreadSafeUpdateAnimation(float DeltaSeconds)
{
    Super::NativeThreadSafeUpdateAnimation(DeltaSeconds);

    FVector RootVelCopy;
    {
        FRWScopeLock Lock(VelocityLock, SLT_ReadOnly);
        RootVelCopy = ThreadSafeRootVelocity;
    }

    // BM-04 FIX: ThreadSafeSpeed lido/escrito com lock
    const float Speed2D = (float)RootVelCopy.Size2D();
    {
        FRWScopeLock Lock(VelocityLock, SLT_Write);
        ThreadSafeSpeed = Speed2D;
    }

    // BM-03 FIX: Única fonte de verdade para MovementMode — thresholds unificados.
    // (Removida a escrita duplicada em NativeUpdateAnimation)
    ECognitiveMovementMode NewMode;
    if (Speed2D < 10.f)
        NewMode = ECognitiveMovementMode::Idle;
    else if (Speed2D < 200.f)
        NewMode = ECognitiveMovementMode::Walk;
    else if (Speed2D < 400.f)
        NewMode = ECognitiveMovementMode::Jog;
    else
        NewMode = ECognitiveMovementMode::Sprint;

    // Postura sobrescreve modo de velocidade
    FCognitiveBlackboard BBCopy;
    {
        FRWScopeLock Lock(BlackboardLock, SLT_ReadOnly);
        BBCopy = ThreadSafeBlackboard;
    }
    if (BBCopy.Posture == ECognitivePosture::Crouching)
        NewMode = ECognitiveMovementMode::Crouch;
    else if (BBCopy.Posture == ECognitivePosture::Prone)
        NewMode = ECognitiveMovementMode::Prone;

    {
        FRWScopeLock Lock(BlackboardLock, SLT_Write);
        ThreadSafeBlackboard.MovementMode = NewMode;
    }
}

void UCognitiveAnimInstance::SetBlackboard(const FCognitiveBlackboard& InBB)
{
    FRWScopeLock Lock(BlackboardLock, SLT_Write);
    ThreadSafeBlackboard = InBB;
}

void UCognitiveAnimInstance::SetTrajectory(const FCognitiveTrajectory& InTraj)
{
    FRWScopeLock Lock(TrajectoryLock, SLT_Write);
    ThreadSafeTrajectory = InTraj;
}

void UCognitiveAnimInstance::SetEmbedding(const FCognitiveMotionEmbedding& InEmbedding)
{
    FRWScopeLock Lock(EmbeddingLock, SLT_Write);
    ThreadSafeEmbedding = InEmbedding;
}

void UCognitiveAnimInstance::SetBlendWeight(float InWeight)
{
    FRWScopeLock Lock(VelocityLock, SLT_Write);
    ThreadSafeBlendWeight = FMath::Clamp(InWeight, 0.f, 1.f);
}

void UCognitiveAnimInstance::UpdateQualityMetrics(float DeltaSeconds)
{
    FVector LeftFoot, RightFoot;
    SampleFootPositions(LeftFoot, RightFoot);

    // BM-09 FIX: quando o NPC está parado (Speed ≈ 0), o SpeedFactor era
    // KINDA_SMALL_NUMBER (~1e-4), causando FootSliding = slide_delta / 1e-4 = valores
    // absurdos que se acumulavam e corrompiam a métrica durante idle.
    // Correção: só acumula FootSliding quando há movimento real (Speed > 10 cm/s).
    float SpeedForMetric;
    {
        FRWScopeLock Lock(VelocityLock, SLT_ReadOnly);
        SpeedForMetric = ThreadSafeSpeed;
    }

    if (SpeedForMetric > 10.f)
    {
        const float SpeedFactor    = FMath::Max(SpeedForMetric * DeltaSeconds, KINDA_SMALL_NUMBER);
        const float LeftSlide      = (LeftFoot - PrevLeftFootPos).Size();
        const float RightSlide     = (RightFoot - PrevRightFootPos).Size();
        const float FootSlide      = (LeftSlide + RightSlide) * 0.5f / SpeedFactor;

        AccumulatedFootSliding     += FootSlide;
        ++FootSlidingFrames;

        if (FootSlidingFrames >= 30)
        {
            QualityMetrics.FootSliding  = AccumulatedFootSliding / FootSlidingFrames;
            AccumulatedFootSliding      = 0.f;
            FootSlidingFrames           = 0;
        }
    }

    {
        FRWScopeLock Lock(EmbeddingLock, SLT_ReadOnly);
        QualityMetrics.Confidence = ThreadSafeEmbedding.Confidence;
    }
    QualityMetrics.LatencyMs    = InferenceLatencyMs;

    PrevLeftFootPos  = LeftFoot;
    PrevRightFootPos = RightFoot;
}

void UCognitiveAnimInstance::SampleFootPositions(FVector& OutLeft, FVector& OutRight) const
{
    OutLeft  = FVector::ZeroVector;
    OutRight = FVector::ZeroVector;

    if (!OwnerCharacter.IsValid()) return;

    const USkeletalMeshComponent* Mesh = OwnerCharacter->GetMesh();
    if (!Mesh) return;

    const FName LeftBone  = FName("foot_l");
    const FName RightBone = FName("foot_r");
    OutLeft  = Mesh->GetBoneLocation(LeftBone,  EBoneSpaces::WorldSpace);
    OutRight = Mesh->GetBoneLocation(RightBone, EBoneSpaces::WorldSpace);
}
