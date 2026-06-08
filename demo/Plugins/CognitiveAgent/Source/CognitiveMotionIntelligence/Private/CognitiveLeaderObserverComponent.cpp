#include "CognitiveLeaderObserverComponent.h"
#include "CognitiveInferenceSubsystem.h"
#include "CognitiveAnimInstance.h"
#include "CognitiveNPCBoneDriver.h"
#include "Components/SkeletalMeshComponent.h"
#include "Engine/GameInstance.h"
#include "GameFramework/Actor.h"
#include "GameFramework/Pawn.h"
#include "GameFramework/PlayerController.h"
#include "Engine/World.h"
#include "Kismet/GameplayStatics.h"

UCognitiveLeaderObserverComponent::UCognitiveLeaderObserverComponent()
{
    PrimaryComponentTick.bCanEverTick = true;
    PrimaryComponentTick.TickGroup   = TG_PostPhysics;
}

void UCognitiveLeaderObserverComponent::BeginPlay()
{
    Super::BeginPlay();
    CaptureInterval = (SamplingRate > 0.0f) ? (1.0f / SamplingRate) : (1.0f / 30.0f);
    FrameBuffer.Reserve(MaxFramesPerSequence);

    if (UGameInstance* GI = GetWorld() ? GetWorld()->GetGameInstance() : nullptr)
    {
        if (UCognitiveInferenceSubsystem* Sub =
            GI->GetSubsystem<UCognitiveInferenceSubsystem>())
        {
            InferenceSubsystem = Sub;
        }
    }

    // Se TargetLeader não foi setado no editor, tenta automaticamente:
    // 1. PlayerPawn(0) — funciona quando o líder é o personagem do jogador
    if (!TargetLeader && GetWorld())
    {
        APlayerController* PC = UGameplayStatics::GetPlayerController(GetWorld(), 0);
        if (PC && PC->GetPawn())
        {
            TargetLeader = PC->GetPawn();
            UE_LOG(LogTemp, Log,
                TEXT("[CognitiveLeaderObserver] TargetLeader não configurado — "
                     "usando PlayerPawn automático: %s"),
                *TargetLeader->GetName());
        }
    }

    // Cache NPC AnimInstance para aplicação direta de bones
    if (ACharacter* NPCOwner = Cast<ACharacter>(GetOwner()))
        if (USkeletalMeshComponent* NPCMesh = NPCOwner->GetMesh())
            CachedNPCAnimInstance = Cast<UCognitiveAnimInstance>(NPCMesh->GetAnimInstance());

    if (TargetLeader)
    {
        CachedLeaderMesh = Cast<USkeletalMeshComponent>(
            TargetLeader->GetComponentByClass(USkeletalMeshComponent::StaticClass()));

        UE_LOG(LogTemp, Log,
            TEXT("[CognitiveLeaderObserver] Leader: %s | Mesh: %s"),
            *TargetLeader->GetName(),
            CachedLeaderMesh.IsValid() ? *CachedLeaderMesh->GetName() : TEXT("NOT FOUND"));
    }
    else
    {
        UE_LOG(LogTemp, Warning,
            TEXT("[CognitiveLeaderObserver] Nenhum TargetLeader encontrado. "
                 "Configure via Details Panel (instância no level) ou "
                 "chame SetTargetLeader() no BeginPlay do BP_CognitiveNPC."));
    }
}

void UCognitiveLeaderObserverComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    if (FrameBuffer.Num() >= 2)
        SendSequence();
    FrameBuffer.Empty();
    Super::EndPlay(EndPlayReason);
}

void UCognitiveLeaderObserverComponent::TickComponent(
    float DeltaTime, ELevelTick TickType,
    FActorComponentTickFunction* ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
    if (!bEnabled || !TargetLeader) return;  // bone capture não depende de InferenceSubsystem

    if (!CachedLeaderMesh.IsValid())
    {
        CachedLeaderMesh = Cast<USkeletalMeshComponent>(
            TargetLeader->GetComponentByClass(USkeletalMeshComponent::StaticClass()));
    }

    CaptureAccumulator += DeltaTime;
    if (CaptureAccumulator >= CaptureInterval)
    {
        CaptureAccumulator -= CaptureInterval;
        CaptureLeaderFrame();
    }

    // Envio ao Python somente quando conectado
    if (InferenceSubsystem.IsValid())
    {
        SendAccumulator += DeltaTime;
        if (SendAccumulator >= SequenceIntervalSeconds && FrameBuffer.Num() >= 2)
        {
            SendAccumulator -= SequenceIntervalSeconds;
            SendSequence();
        }
    }
}

void UCognitiveLeaderObserverComponent::CaptureLeaderFrame()
{
    if (!TargetLeader || !TargetLeader->IsValidLowLevel())
    {
        ++FailedCaptures;
        return;
    }

    FCognitivePoseFrame Frame;
    if (!FillPoseFrame(Frame))
    {
        ++FailedCaptures;
        return;
    }

    // Aplica bones direto no NPC somente em Observing
    // Em Inferring: Python controla autonomamente via LearnerComponent
    if (!Frame.BoneTransforms.IsEmpty() && CachedNPCAnimInstance.IsValid())
    {
        ECognitiveObservationState CurState = ECognitiveObservationState::Observing;
        if (UCognitiveNPCBoneDriver* BD =
            GetOwner() ? GetOwner()->FindComponentByClass<UCognitiveNPCBoneDriver>() : nullptr)
            CurState = BD->ObservationState;

        if (CurState == ECognitiveObservationState::Observing)
        {
            CachedNPCAnimInstance->SetBoneTransforms(Frame.BoneTransforms);
            CachedNPCAnimInstance->bInferenceFallbackActive = false;
        }
    }

    if (FrameBuffer.Num() >= MaxFramesPerSequence)
        FrameBuffer.RemoveAt(0, 1, EAllowShrinking::No);

    if (FrameBuffer.Num() == 0)
        BufferStartTimestamp = Frame.Timestamp;

    FrameBuffer.Add(Frame);
    ++TotalFramesCaptured;
}

bool UCognitiveLeaderObserverComponent::FillPoseFrame(FCognitivePoseFrame& OutFrame)
{
    if (!TargetLeader) return false;

    // BM-02 FIX: O código anterior usava GetWorld()->GetTimeSeconds() (tempo do mundo,
    // afetado por time dilation e pausas) enquanto UCognitivePoseRecorderComponent usa
    // FPlatformTime::Seconds() (wall-clock real). Timestamps incompatíveis entre frames
    // de líder e seguidor causam erros de interpolação temporal no servidor Python.
    // Correção: usar FPlatformTime::Seconds() em ambos os lados.
    const double  Now             = FPlatformTime::Seconds();
    const FVector CurrentLocation = TargetLeader->GetActorLocation();
    const FQuat   CurrentRotation = TargetLeader->GetActorQuat();

    if (!bFirstCapture)
        LeaderLinearVelocity = (CurrentLocation - LastLeaderLocation) / FMath::Max(CaptureInterval, SMALL_NUMBER);
    else
    {
        LeaderLinearVelocity = FVector::ZeroVector;
        bFirstCapture        = false;
    }
    LastLeaderLocation = CurrentLocation;

    OutFrame.Timestamp       = Now;
    OutFrame.FrameIndex      = TotalFramesCaptured;
    OutFrame.RootLocation    = CurrentLocation;
    OutFrame.RootRotation    = CurrentRotation;
    OutFrame.LinearVelocity  = LeaderLinearVelocity;
    OutFrame.AngularVelocity = FVector::ZeroVector;
    OutFrame.MovementMode    = ECognitiveMovementMode::Idle;
    OutFrame.MotionStyle     = ECognitiveMotionStyle::Neutral;

    if (UPrimitiveComponent* Prim = Cast<UPrimitiveComponent>(TargetLeader->GetRootComponent()))
        OutFrame.AngularVelocity = Prim->GetPhysicsAngularVelocityInRadians();

    if (CachedLeaderMesh.IsValid())
        ExtractBoneTransforms(CachedLeaderMesh.Get(), OutFrame);

    return true;
}

void UCognitiveLeaderObserverComponent::ExtractBoneTransforms(
    USkeletalMeshComponent* Mesh, FCognitivePoseFrame& OutFrame) const
{
    if (!Mesh) return;

    // DISTORÇÃO FIX: GetBoneTransform(i, ComponentTransform) retorna world space.
    // Output.Pose do AnimNode espera component space.
    // GetComponentSpaceTransforms() retorna o array de transforms já em component space
    // (atualizado após o animation tick — disponível em TG_PostPhysics).
    const TArray<FTransform>& CSTransforms = Mesh->GetComponentSpaceTransforms();
    OutFrame.BoneTransforms = CSTransforms;
}

void UCognitiveLeaderObserverComponent::SendSequence()
{
    if (!InferenceSubsystem.IsValid() || FrameBuffer.Num() < 2) return;

    CognitiveMotionProtocol::FLeaderSequencePayload Payload;
    Payload.Frames          = FrameBuffer;
    Payload.SequenceId      = NextSequenceId++;
    Payload.StartTimestamp  = BufferStartTimestamp;
    Payload.EndTimestamp    = FrameBuffer.Last().Timestamp;
    Payload.LeaderNPCId     = LeaderNPCId;
    Payload.FollowerNPCId   = FollowerNPCId;

    const TArray<uint8> Data = CognitiveMotionProtocol::SerializeLeaderSequence(Payload);
    if (Data.Num() > 0)
    {
        InferenceSubsystem->SendRawMessage(Data);
        ++TotalSequencesSent;
    }

    const int32 Keep = FMath::Max(MaxFramesPerSequence / 4, 1);
    if (FrameBuffer.Num() > Keep)
    {
        FrameBuffer.RemoveAt(0, FrameBuffer.Num() - Keep, EAllowShrinking::No);
        if (FrameBuffer.Num() > 0)
            BufferStartTimestamp = FrameBuffer[0].Timestamp;
    }
}

void UCognitiveLeaderObserverComponent::SetTargetLeader(AActor* NewLeader, int64 NewLeaderNPCId)
{
    TargetLeader         = NewLeader;
    LeaderNPCId          = NewLeaderNPCId;
    CachedLeaderMesh     = nullptr;
    bFirstCapture        = true;
    FrameBuffer.Empty();
    BufferStartTimestamp = 0.0;

    if (NewLeader)
        CachedLeaderMesh = Cast<USkeletalMeshComponent>(
            NewLeader->GetComponentByClass(USkeletalMeshComponent::StaticClass()));
}

void UCognitiveLeaderObserverComponent::SetLeaderAsPlayer(int32 PlayerIndex, int64 NewLeaderNPCId)
{
    // Busca o PlayerPawn pelo índice — funciona para qualquer tipo de Character (BP_ThirdPersonCharacter, etc.)
    // Chame no BeginPlay do BP_CognitiveNPC: GetComponentByClass → SetLeaderAsPlayer(0, 1001)
    if (!GetWorld()) return;

    APlayerController* PC = UGameplayStatics::GetPlayerController(GetWorld(), PlayerIndex);
    if (!PC || !PC->GetPawn())
    {
        UE_LOG(LogTemp, Warning,
            TEXT("[CognitiveLeaderObserver] SetLeaderAsPlayer: PlayerController(%d) ou Pawn não encontrado."),
            PlayerIndex);
        return;
    }

    SetTargetLeader(PC->GetPawn(), NewLeaderNPCId);

    UE_LOG(LogTemp, Log,
        TEXT("[CognitiveLeaderObserver] Líder setado como PlayerPawn: %s"),
        *PC->GetPawn()->GetName());
}

void UCognitiveLeaderObserverComponent::FlushSequence()
{
    if (FrameBuffer.Num() >= 2)
        SendSequence();
}

FString UCognitiveLeaderObserverComponent::GetDiagnostics() const
{
    return FString::Printf(
        TEXT("LeaderObserver | Leader=%s | LId=%lld | FId=%lld | Buf=%d/%d | Cap=%d | Sent=%d | Fail=%d"),
        TargetLeader ? *TargetLeader->GetName() : TEXT("None"),
        LeaderNPCId, FollowerNPCId,
        FrameBuffer.Num(), MaxFramesPerSequence,
        TotalFramesCaptured, TotalSequencesSent, FailedCaptures);
}
