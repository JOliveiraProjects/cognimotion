#include "CognitiveMotionDebugVisualizerComponent.h"
#include "CognitiveAnimInstance.h"
#include "CognitiveMotionLearnerComponent.h"
#include "DrawDebugHelpers.h"
#include "GameFramework/Character.h"
#include "Engine/Canvas.h"

UCognitiveMotionDebugVisualizerComponent::UCognitiveMotionDebugVisualizerComponent()
{
    PrimaryComponentTick.bCanEverTick = true;
    PrimaryComponentTick.TickInterval = 0.033f;
    // BB-05 FIX: O componente está no módulo Editor — não deve persistir em builds
    // cooked (Shipping/Development sem editor). bIsEditorOnly = true garante que
    // seja removido automaticamente pelo cooker, economizando memória em runtime.
    bIsEditorOnly = true;
}

void UCognitiveMotionDebugVisualizerComponent::TickComponent(
    float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

#if ENABLE_DRAW_DEBUG
    AActor* Owner = GetOwner();
    if (!Owner || !GetWorld()) return;

    UCognitiveAnimInstance* AnimInst = nullptr;
    UCognitiveMotionLearnerComponent* Learner = Owner->FindComponentByClass<UCognitiveMotionLearnerComponent>();

    if (ACharacter* Char = Cast<ACharacter>(Owner))
        if (USkeletalMeshComponent* Mesh = Char->GetMesh())
            AnimInst = Cast<UCognitiveAnimInstance>(Mesh->GetAnimInstance());

    if (!AnimInst) return;

    if (bShowFutureTrajectory)
    {
        const FCognitiveTrajectory& FTraj = AnimInst->GetFutureTrajectory();
        if (FTraj.IsValid()) DrawTrajectory(FTraj, FutureTrajectoryColor);
    }

    if (bShowEmbeddingConfidence)
        DrawConfidenceBar();

    if (bShowNPCState || bShowFallbackState)
        DrawStateText();

    if (bShowFootSliding)
        DrawFootSlidingIndicator();
#endif
}

void UCognitiveMotionDebugVisualizerComponent::DrawTrajectory(
    const FCognitiveTrajectory& Traj, const FColor& Color) const
{
#if ENABLE_DRAW_DEBUG
    const UWorld* World = GetWorld();
    if (!World || Traj.Samples.IsEmpty()) return;

    const float ZOffset = 5.f;

    FVector Prev = Traj.Samples[0].Position + FVector(0, 0, ZOffset);
    for (int32 i = 1; i < Traj.Samples.Num(); ++i)
    {
        const FVector Curr = Traj.Samples[i].Position + FVector(0, 0, ZOffset);
        DrawDebugLine(World, Prev, Curr, Color, false, DrawDuration, 0, TrajectoryLineThickness);
        DrawDebugSphere(World, Curr, 4.f, 6, Color, false, DrawDuration);
        Prev = Curr;
    }
#endif
}

void UCognitiveMotionDebugVisualizerComponent::DrawStateText() const
{
#if ENABLE_DRAW_DEBUG
    const UWorld* World = GetWorld();
    AActor* Owner = GetOwner();
    if (!World || !Owner) return;

    ACharacter* Char = Cast<ACharacter>(Owner);
    if (!Char) return;

    UCognitiveAnimInstance* AnimInst = nullptr;
    if (USkeletalMeshComponent* Mesh = Char->GetMesh())
        AnimInst = Cast<UCognitiveAnimInstance>(Mesh->GetAnimInstance());

    if (!AnimInst) return;

    const FVector TextLoc = Owner->GetActorLocation() + FVector(0, 0, 120.f);

    if (bShowNPCState)
    {
        const UEnum* StateEnum = StaticEnum<ECognitiveNPCState>();
        const FString StateStr = StateEnum
            ? StateEnum->GetNameStringByValue((int64)AnimInst->GetNPCState())
            : TEXT("?");
        DrawDebugString(World, TextLoc, FString::Printf(TEXT("State: %s"), *StateStr),
            nullptr, FColor::White, DrawDuration, false, 1.0f);
    }

    if (bShowFallbackState)
    {
        const bool bFallback = AnimInst->bInferenceFallbackActive;
        DrawDebugString(World, TextLoc + FVector(0, 0, 16.f),
            bFallback ? TEXT("[FALLBACK]") : TEXT("[INFERENCE]"),
            nullptr, bFallback ? FColor::Red : FColor::Green, DrawDuration, false, 1.0f);
    }

    if (bShowLatency)
    {
        DrawDebugString(World, TextLoc + FVector(0, 0, 32.f),
            FString::Printf(TEXT("Latency: %.1f ms"), AnimInst->InferenceLatencyMs),
            nullptr, FColor::Yellow, DrawDuration, false, 1.0f);
    }
#endif
}

void UCognitiveMotionDebugVisualizerComponent::DrawConfidenceBar() const
{
#if ENABLE_DRAW_DEBUG
    AActor* Owner = GetOwner();
    if (!Owner) return;

    UCognitiveMotionLearnerComponent* Learner = Owner->FindComponentByClass<UCognitiveMotionLearnerComponent>();
    if (!Learner) return;

    const float Confidence = Learner->GetResponseConfidence();
    const FColor BarColor  = FColor::MakeRedToGreenColorFromScalar(Confidence);
    const FVector Origin   = Owner->GetActorLocation() + FVector(0, 0, 160.f);

    DrawDebugLine(GetWorld(), Origin, Origin + FVector(0, 0, Confidence * 40.f),
        BarColor, false, DrawDuration, 0, 4.f);
#endif
}

void UCognitiveMotionDebugVisualizerComponent::DrawFootSlidingIndicator() const
{
#if ENABLE_DRAW_DEBUG
    ACharacter* Char = Cast<ACharacter>(GetOwner());
    if (!Char) return;

    UCognitiveAnimInstance* AnimInst = nullptr;
    if (USkeletalMeshComponent* Mesh = Char->GetMesh())
        AnimInst = Cast<UCognitiveAnimInstance>(Mesh->GetAnimInstance());
    if (!AnimInst) return;

    const float Sliding    = AnimInst->QualityMetrics.FootSliding;
    const FColor SlideColor = Sliding > 0.3f ? FColor::Red : (Sliding > 0.1f ? FColor::Yellow : FColor::Green);
    const FVector FootLoc  = Char->GetActorLocation();

    DrawDebugSphere(GetWorld(), FootLoc, Sliding * 30.f + 5.f, 8, SlideColor, false, DrawDuration);
#endif
}
