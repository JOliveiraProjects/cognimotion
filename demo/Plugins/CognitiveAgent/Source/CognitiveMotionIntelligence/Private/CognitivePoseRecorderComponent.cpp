#include "CognitivePoseRecorderComponent.h"
#include "CognitiveAnimInstance.h"
#include "GameFramework/Character.h"
#include "Components/SkeletalMeshComponent.h"
#include "Engine/World.h"

UCognitivePoseRecorderComponent::UCognitivePoseRecorderComponent()
{
    PrimaryComponentTick.bCanEverTick = true;
    PrimaryComponentTick.bStartWithTickEnabled = false;
    PrimaryComponentTick.TickInterval = 1.f / 30.f;
    BonesToCapture = { FName("pelvis"), FName("spine_01"), FName("spine_02"), FName("spine_03"),
                       FName("foot_l"), FName("foot_r"), FName("hand_l"), FName("hand_r"),
                       FName("head") };
}

void UCognitivePoseRecorderComponent::BeginPlay()
{
    Super::BeginPlay();

    CircularBuffer.SetNum(BufferCapacity);
    SampleInterval = 1.f / FMath::Max(SamplingRate, 1.f);

    if (AActor* Owner = GetOwner())
    {
        if (ACharacter* Char = Cast<ACharacter>(Owner))
        {
            CachedMesh = Char->GetMesh();
            if (CachedMesh.IsValid())
            {
                CachedAnimInstance = Cast<UCognitiveAnimInstance>(
                    CachedMesh->GetAnimInstance());
            }
        }
    }
}

void UCognitivePoseRecorderComponent::TickComponent(
    float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

    if (!bIsRecording || !CachedMesh.IsValid()) return;

    SampleAccumulator += DeltaTime;
    while (SampleAccumulator >= SampleInterval)
    {
        CaptureFrame();
        SampleAccumulator -= SampleInterval;
    }
}

void UCognitivePoseRecorderComponent::StartRecording()
{
    bIsRecording = true;
    PrimaryComponentTick.SetTickFunctionEnable(true);
    SampleAccumulator = 0.f;
}

void UCognitivePoseRecorderComponent::StopRecording()
{
    bIsRecording = false;
    PrimaryComponentTick.SetTickFunctionEnable(false);
}

void UCognitivePoseRecorderComponent::CaptureFrame()
{
    if (!CachedMesh.IsValid()) return;

    AActor* Owner = GetOwner();
    if (!Owner) return;

    const double Now = FPlatformTime::Seconds();

    // BM-08 FIX: PrevSampleTime e LastSampleTime eram float, causando perda de
    // precisão para timestamps grandes (wall-clock em segundos pode ser 1e6+ após
    // horas de runtime). Cast float→double perde ~7 dígitos significativos.
    // Com Now ≈ 1.7e9 (Unix timestamp relativo) e DT ≈ 0.033s, o DT calculado
    // sofre cancelamento catastrófico: (1700000001.033 - 1700000001.000) como float
    // pode dar 0.0 ou valores aleatórios. Correção: manter PrevSampleTime como double.
    const double DT_d = (PrevSampleTime > 0.0) ? (Now - PrevSampleTime) : (double)SampleInterval;
    const float  DT   = (DT_d > (double)KINDA_SMALL_NUMBER) ? (float)DT_d : SampleInterval;

    const FVector Loc   = Owner->GetActorLocation();
    const FQuat   Rot   = Owner->GetActorQuat();
    const FVector LinVel = DT > KINDA_SMALL_NUMBER ? (Loc - PrevLocation) / DT : FVector::ZeroVector;
    const FQuat   DeltaQ = PrevRotation.Inverse() * Rot;
    FVector AngAxis; float AngAngle;
    DeltaQ.ToAxisAndAngle(AngAxis, AngAngle);
    const FVector AngVel = DT > KINDA_SMALL_NUMBER ? (AngAxis * AngAngle) / DT : FVector::ZeroVector;

    FCognitivePoseFrame Frame;
    Frame.Timestamp      = Now;
    Frame.FrameIndex     = FrameIndex++;
    Frame.RootLocation   = Loc;
    Frame.RootRotation   = Rot;
    Frame.LinearVelocity = LinVel;
    Frame.AngularVelocity = AngVel;

    if (CachedAnimInstance.IsValid())
    {
        Frame.MovementMode = CachedAnimInstance->GetMovementMode();
        Frame.MotionStyle  = CachedAnimInstance->GetMotionStyle();
    }

    if (bCaptureBoneTransforms && CachedMesh.IsValid())
    {
        PopulateFrameBones(Frame, CachedMesh.Get());
    }

    // BM-08 FIX: LastSampleTime e PrevSampleTime mantidos como double
    const double DeltaFromLast = Now - LastSampleTime;
    if (DeltaFromLast > 0.0)
        ActualSamplingRate = (float)(1.0 / DeltaFromLast);

    PrevLocation   = Loc;
    PrevRotation   = Rot;
    PrevSampleTime = Now;    // double — sem perda de precisão
    LastSampleTime = Now;    // double — sem perda de precisão

    PushFrame(MoveTemp(Frame));
    ++TotalFramesCaptured;
}

void UCognitivePoseRecorderComponent::PopulateFrameBones(
    FCognitivePoseFrame& Frame, const USkeletalMeshComponent* Mesh) const
{
    // DISTORÇÃO FIX: usar GetComponentSpaceTransforms() em vez de
    // GetBoneTransform(i, ComponentTransform) que retorna world space.
    // AnimNode Output.Pose e GetComponentSpaceTransforms() usam o mesmo espaço.
    Frame.BoneTransforms = Mesh->GetComponentSpaceTransforms();
}

void UCognitivePoseRecorderComponent::PushFrame(FCognitivePoseFrame&& Frame)
{
    FRWScopeLock Lock(BufferLock, SLT_Write);
    CircularBuffer[BufferHead] = MoveTemp(Frame);
    BufferHead = (BufferHead + 1) % BufferCapacity;
    BufferSize = FMath::Min(BufferSize + 1, BufferCapacity);
}

bool UCognitivePoseRecorderComponent::GetLatestFrame(FCognitivePoseFrame& OutFrame) const
{
    FRWScopeLock Lock(BufferLock, SLT_ReadOnly);
    if (BufferSize == 0) return false;
    const int32 LatestIdx = (BufferHead - 1 + BufferCapacity) % BufferCapacity;
    OutFrame = CircularBuffer[LatestIdx];
    return true;
}

void UCognitivePoseRecorderComponent::GetRecentFrames(int32 Count, TArray<FCognitivePoseFrame>& OutFrames) const
{
    FRWScopeLock Lock(BufferLock, SLT_ReadOnly);
    const int32 Available = FMath::Min(Count, BufferSize);
    OutFrames.Reserve(Available);
    for (int32 i = Available - 1; i >= 0; --i)
    {
        const int32 Idx = (BufferHead - 1 - i + BufferCapacity) % BufferCapacity;
        OutFrames.Add(CircularBuffer[Idx]);
    }
}

void UCognitivePoseRecorderComponent::BuildTrajectoryFromBuffer(
    FCognitiveTrajectory& OutPast, FCognitiveTrajectory& OutFuture,
    int32 PastSamples, int32 FutureSamples) const
{
    TArray<FCognitivePoseFrame> Recent;
    GetRecentFrames(PastSamples, Recent);

    OutPast.Samples.Reset();
    OutPast.Samples.Reserve(Recent.Num());

    for (int32 i = 0; i < Recent.Num(); ++i)
    {
        const FCognitivePoseFrame& F = Recent[i];
        FCognitiveTrajectorySample S;
        S.Position        = F.RootLocation;
        S.LinearVelocity  = F.LinearVelocity;
        S.AngularVelocity = F.AngularVelocity;
        S.Facing          = F.RootRotation;
        S.TimeInSeconds   = (float)(i - Recent.Num() + 1) * SampleInterval;
        S.Speed           = (float)F.LinearVelocity.Size2D();
        OutPast.Samples.Add(S);
    }

    if (Recent.Num() > 0)
    {
        const FCognitivePoseFrame& Latest = Recent.Last();
        const FVector Vel = Latest.LinearVelocity;
        FVector Pos = Latest.RootLocation;

        OutFuture.Samples.Reset();
        OutFuture.Samples.Reserve(FutureSamples);
        for (int32 i = 1; i <= FutureSamples; ++i)
        {
            Pos += Vel * SampleInterval;
            FCognitiveTrajectorySample S;
            S.Position        = Pos;
            S.LinearVelocity  = Vel;
            S.Facing          = Latest.RootRotation;
            S.TimeInSeconds   = i * SampleInterval;
            S.Speed           = (float)Vel.Size2D();
            OutFuture.Samples.Add(S);
        }
    }
}
