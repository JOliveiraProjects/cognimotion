#include "CognitiveTrajectoryGenerator.h"

void UCognitiveTrajectoryGenerator::Initialize(const FCognitiveTrajectoryConfig& InConfig)
{
    Config = InConfig;
    FScopeLock Lock(&HistoryLock);
    FrameHistory.Reserve(MaxHistory);
}

FCognitiveTrajectory UCognitiveTrajectoryGenerator::GenerateFutureTrajectory(
    const FVector& CurrentLocation,
    const FVector& CurrentVelocity,
    const FQuat&   CurrentFacing,
    const FVector& DesiredDirection,
    float          DesiredSpeed,
    ECognitiveMovementMode MovementMode) const
{
    FCognitiveTrajectory Traj;
    Traj.Samples.Reserve(Config.FutureSamples);

    const float ClampedSpeed = FMath::Clamp(DesiredSpeed, 0.f, Config.MaxPredictSpeed);
    const FVector TargetVel  = DesiredDirection.GetSafeNormal() * ClampedSpeed;

    FVector   Pos   = CurrentLocation;
    FVector   Vel   = CurrentVelocity;
    FQuat     Rot   = CurrentFacing;

    for (int32 i = 0; i < Config.FutureSamples; ++i)
    {
        Vel = SmoothVector(Vel, TargetVel, 1.f - FMath::Pow(Config.SmoothingFactor, (float)(i + 1)));
        Pos += Vel * Config.SampleInterval;

        FQuat TargetRot = Rot;
        if (!DesiredDirection.IsNearlyZero())
        {
            const FVector FlatDir = FVector(DesiredDirection.X, DesiredDirection.Y, 0.f).GetSafeNormal();
            TargetRot = FlatDir.ToOrientationQuat();
        }
        Rot = SmoothRotation(Rot, TargetRot, 1.f - FMath::Pow(Config.TurnSmoothing, (float)(i + 1)));

        FCognitiveTrajectorySample Sample;
        Sample.Position       = Pos;
        Sample.LinearVelocity = Vel;
        Sample.Facing         = Rot;
        Sample.TimeInSeconds  = (i + 1) * Config.SampleInterval;
        Sample.Speed          = (float)Vel.Size2D();
        Traj.Samples.Add(Sample);
    }
    return Traj;
}

FCognitiveTrajectory UCognitiveTrajectoryGenerator::BuildPastTrajectory(
    const TArray<FVector>& LocationHistory,
    const TArray<FVector>& VelocityHistory,
    const TArray<FQuat>&   FacingHistory,
    const TArray<float>&   TimeHistory) const
{
    FCognitiveTrajectory Traj;
    const int32 N = FMath::Min(Config.PastSamples,
        FMath::Min(TimeHistory.Num(),
            FMath::Min(FacingHistory.Num(),
                FMath::Min(LocationHistory.Num(), VelocityHistory.Num()))));
    Traj.Samples.Reserve(N);
    for (int32 i = 0; i < N; ++i)
    {
        FCognitiveTrajectorySample S;
        S.Position       = LocationHistory[i];
        S.LinearVelocity = VelocityHistory[i];
        S.Facing         = FacingHistory[i];
        S.TimeInSeconds  = TimeHistory[i];
        S.Speed          = (float)VelocityHistory[i].Size2D();
        Traj.Samples.Add(S);
    }
    return Traj;
}

FCognitiveTrajectory UCognitiveTrajectoryGenerator::BlendWithPythonResponse(
    const FCognitiveTrajectory& Generated,
    const FCognitiveTrajectory& PythonRefined,
    float BlendAlpha) const
{
    if (!PythonRefined.IsValid() || BlendAlpha < KINDA_SMALL_NUMBER)
        return Generated;
    if (!Generated.IsValid())
        return PythonRefined;

    FCognitiveTrajectory Result;
    const int32 N = FMath::Min(Generated.Samples.Num(), PythonRefined.Samples.Num());
    Result.Samples.Reserve(N);

    for (int32 i = 0; i < N; ++i)
    {
        const FCognitiveTrajectorySample& G = Generated.Samples[i];
        const FCognitiveTrajectorySample& P = PythonRefined.Samples[i];
        FCognitiveTrajectorySample Blended;
        Blended.Position       = FMath::Lerp(G.Position,       P.Position,       BlendAlpha);
        Blended.LinearVelocity = FMath::Lerp(G.LinearVelocity, P.LinearVelocity, BlendAlpha);
        Blended.Facing         = FQuat::Slerp(G.Facing,        P.Facing,         BlendAlpha);
        Blended.TimeInSeconds  = G.TimeInSeconds;
        Blended.Speed          = FMath::Lerp(G.Speed,          P.Speed,          BlendAlpha);
        Result.Samples.Add(Blended);
    }
    return Result;
}

FCognitiveTrajectory UCognitiveTrajectoryGenerator::MakeIdleTrajectory() const
{
    FCognitiveTrajectory Traj;
    Traj.Samples.Reserve(Config.FutureSamples);
    for (int32 i = 0; i < Config.FutureSamples; ++i)
    {
        FCognitiveTrajectorySample S;
        S.Position       = FVector::ZeroVector;
        S.LinearVelocity = FVector::ZeroVector;
        S.Facing         = FQuat::Identity;
        S.TimeInSeconds  = (i + 1) * Config.SampleInterval;
        S.Speed          = 0.f;
        Traj.Samples.Add(S);
    }
    return Traj;
}

void UCognitiveTrajectoryGenerator::RecordFrame(
    const FVector& Location, const FVector& Velocity, const FQuat& Facing, double Timestamp)
{
    FScopeLock Lock(&HistoryLock);
    // TECH DEBT FIX: RemoveAt(0) em TArray é O(n) — desloca todos os elementos
    // a cada frame. Para MaxHistory=128 a 30 fps é 128 ops/frame = 3840 ops/s.
    // Correção: sobrescrever o slot mais antigo usando índice circular (O(1)).
    // FrameHistory mantém capacidade fixa; HistoryHead aponta para o próximo slot.
    if (FrameHistory.Num() < MaxHistory)
    {
        FrameHistory.Add({ Location, Velocity, Facing, Timestamp });
    }
    else
    {
        FrameHistory[HistoryHead] = { Location, Velocity, Facing, Timestamp };
        HistoryHead = (HistoryHead + 1) % MaxHistory;
    }
}

FCognitiveTrajectory UCognitiveTrajectoryGenerator::GetRecordedPastTrajectory() const
{
    // TECH DEBT FIX: mutable HistoryLock — const_cast removido
    FScopeLock Lock(&HistoryLock);

    FCognitiveTrajectory Traj;
    const int32 Total = FrameHistory.Num();
    const int32 N     = FMath::Min(Total, Config.PastSamples);
    if (N == 0) return Traj;

    Traj.Samples.Reserve(N);
    const double Now = FPlatformTime::Seconds();

    // TECH DEBT FIX: com ring buffer, os frames estão em ordem circular.
    // HistoryHead aponta para o próximo slot a escrever.
    // Último N frames em ordem cronológica: lemos do mais antigo ao mais recente.
    for (int32 i = N - 1; i >= 0; --i)
    {
        // Índice no ring buffer: conta N-1 slots atrás do HistoryHead
        // (ou do final do array se ainda não wrappou)
        const int32 Idx = Total < MaxHistory
            ? (Total - 1 - (N - 1 - i))                        // sem wrap
            : (HistoryHead - 1 - i + MaxHistory) % MaxHistory;  // com wrap

        if (Idx < 0 || Idx >= Total) continue;
        const FHistoryFrame& F = FrameHistory[Idx];
        FCognitiveTrajectorySample S;
        S.Position       = F.Location;
        S.LinearVelocity = F.Velocity;
        S.Facing         = F.Facing;
        S.TimeInSeconds  = F.Time - Now;
        S.Speed          = (float)F.Velocity.Size2D();
        Traj.Samples.Add(S);
    }
    return Traj;
}

FVector UCognitiveTrajectoryGenerator::SmoothVector(
    const FVector& Current, const FVector& Target, float Alpha) const
{
    return FMath::Lerp(Current, Target, FMath::Clamp(Alpha, 0.f, 1.f));
}

FQuat UCognitiveTrajectoryGenerator::SmoothRotation(
    const FQuat& Current, const FQuat& Target, float Alpha) const
{
    return FQuat::Slerp(Current, Target, FMath::Clamp(Alpha, 0.f, 1.f));
}
