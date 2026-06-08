#include "CognitiveMotionLearnerComponent.h"
#include "Components/SkeletalMeshComponent.h"
#include "CognitiveDebugLog.h"
#include "CognitiveWorldPerceptionComponent.h"
#include "CognitiveEntityTagComponent.h"
#include "CognitiveInferenceSubsystem.h"
#include "CognitivePoseRecorderComponent.h"
#include "CognitiveAnimInstance.h"
#include "GameFramework/Character.h"
#include "DrawDebugHelpers.h"
#include "Engine/World.h"
#include "Engine/GameInstance.h"

UCognitiveMotionLearnerComponent::UCognitiveMotionLearnerComponent()
{
    PrimaryComponentTick.bCanEverTick = true;
    PrimaryComponentTick.TickInterval = 0.016f;
}

void UCognitiveMotionLearnerComponent::BeginPlay()
{
    Super::BeginPlay();

    if (UGameInstance* GI = GetWorld()->GetGameInstance())
    {
        InferenceSubsystem = GI->GetSubsystem<UCognitiveInferenceSubsystem>();
        if (InferenceSubsystem && bAutoConnect)
            InferenceSubsystem->Connect(PythonHost, PythonPort);
    }

    if (AActor* Owner = GetOwner())
    {
        PoseRecorder = Owner->FindComponentByClass<UCognitivePoseRecorderComponent>();
        if (PoseRecorder.IsValid())
        {
            // Ensure recording starts automatically so BoneTransforms are populated
            // when the first motion request is sent.
            PoseRecorder->StartRecording();
        }

        if (ACharacter* Char = Cast<ACharacter>(Owner))
        {
            if (USkeletalMeshComponent* Mesh = Char->GetMesh())
            {
                AnimInstance = Cast<UCognitiveAnimInstance>(Mesh->GetAnimInstance());
                if (!AnimInstance.IsValid())
                {
                    UE_LOG(LogTemp, Warning,
                        TEXT("CognitiveMotionLearnerComponent: AnimInstance on '%s' is not a "
                             "UCognitiveAnimInstance. Assign ABP_Npc (or any AnimBP derived from "
                             "UCognitiveAnimInstance) to the Skeletal Mesh."),
                        *Owner->GetName());
                }
            }
        }
    }

    // BM-01 FIX: TrajectoryGenerator e PoseMemory eram declarados como UPROPERTY()
    // mas nunca instanciados — o bloco if(TrajectoryGenerator) nunca executava,
    // a trajetória futura alternativa nunca era gerada, e PoseMemory era dead code.
    TrajectoryGenerator = NewObject<UCognitiveTrajectoryGenerator>(this);
    {
        FCognitiveTrajectoryConfig TrajConfig;
        TrajConfig.PastSamples    = 6;
        TrajConfig.FutureSamples  = 6;
        TrajConfig.SampleInterval = RequestIntervalSeconds;
        TrajectoryGenerator->Initialize(TrajConfig);
    }

    PoseMemory = NewObject<UCognitiveRuntimePoseMemory>(this);
    PoseMemory->Initialize(10000, 256);
}

void UCognitiveMotionLearnerComponent::EndPlay(const EEndPlayReason::Type Reason)
{
    Super::EndPlay(Reason);
}

void UCognitiveMotionLearnerComponent::TickComponent(
    float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* Func)
{
    Super::TickComponent(DeltaTime, TickType, Func);

    PollInferenceResponses();
    UpdateFallbackBlend(DeltaTime);

    RequestAccumulator += DeltaTime;
    if (RequestAccumulator >= RequestIntervalSeconds)
    {
        RequestAccumulator = 0.f;

        if (InferenceSubsystem && InferenceSubsystem->IsReady())
        {
            FCognitiveBlackboard BB;
            if (AnimInstance.IsValid())
            {
                BB = AnimInstance->GetBlackboard();
            }

            // Vida externa (do Health component) sobrescreve a do blackboard,
            // para a decisão reativa saber quando fugir/morrer.
            if (bHasExternalHealth)
            {
                BB.Health = ExternalHealth;
            }

            // Enriquece o blackboard com percepção semântica do mundo, se houver.
            // Assim o conhecimento de inimigo/ameaça/alvo chega ao Python e
            // influencia a política aprendida.
            if (UCognitiveWorldPerceptionComponent* Perception =
                    GetOwner()->FindComponentByClass<UCognitiveWorldPerceptionComponent>())
            {
                const FCognitivePerceivedEntity Threat = Perception->GetNearestThreat();
                if (Threat.Actor.IsValid())
                {
                    BB.ThreatLevel = FMath::Max(BB.ThreatLevel, Threat.ThreatWeight);
                    BB.Alertness   = FMath::Max(BB.Alertness, 0.5f);
                    BB.LastKnownTargetPosition = Threat.Actor->GetActorLocation();
                    if (Threat.SuggestedReaction == ECognitiveReaction::Flee ||
                        Threat.SuggestedReaction == ECognitiveReaction::Hide)
                        BB.FearLevel = FMath::Max(BB.FearLevel, Threat.ThreatWeight);
                    else if (Threat.SuggestedReaction == ECognitiveReaction::Attack)
                        BB.AggressionLevel = FMath::Max(BB.AggressionLevel, 0.6f);
                }
            }

            FCognitiveTrajectory PastTraj, FutureTraj;
            if (PoseRecorder.IsValid())
            {
                PoseRecorder->BuildTrajectoryFromBuffer(PastTraj, FutureTraj, 6, 6);
            }
            if (TrajectoryGenerator)
            {
                TrajectoryGenerator->RecordFrame(
                    GetOwner()->GetActorLocation(),
                    AnimInstance.IsValid() ? AnimInstance->GetRootVelocity() : FVector::ZeroVector,
                    GetOwner()->GetActorQuat(),
                    (float)FPlatformTime::Seconds());
                if (!FutureTraj.IsValid())
                    FutureTraj = TrajectoryGenerator->GenerateFutureTrajectory(
                        GetOwner()->GetActorLocation(),
                        AnimInstance.IsValid() ? AnimInstance->GetRootVelocity() : FVector::ZeroVector,
                        GetOwner()->GetActorQuat(),
                        GetOwner()->GetActorForwardVector(),
                        AnimInstance.IsValid() ? AnimInstance->GetSpeed() : 0.f,
                        AnimInstance.IsValid() ? AnimInstance->GetMovementMode() : ECognitiveMovementMode::Idle);
            }
            RequestMotionInference(BB, FutureTraj);
        }
        else if (!bFallbackActive)
        {
            ActivateFallback(TEXT("Inference not available"));
        }
    }

    EmitDebugDraw();
}

void UCognitiveMotionLearnerComponent::RequestMotionInference(
    const FCognitiveBlackboard& Blackboard, const FCognitiveTrajectory& DesiredTrajectory)
{
    if (!InferenceSubsystem || !InferenceSubsystem->IsReady()) return;

    FCognitivePoseFrame CurrentPose;
    bool bGotFrame = false;
    if (PoseRecorder.IsValid())
        bGotFrame = PoseRecorder->GetLatestFrame(CurrentPose);

    // If no recorded frame yet (recording just started or bCaptureBoneTransforms was off),
    // fill from the live skeletal mesh so Python never receives an empty BoneTransforms array.
    if (!bGotFrame || CurrentPose.BoneTransforms.IsEmpty())
    {
        if (AActor* Owner = GetOwner())
        {
            if (ACharacter* Char = Cast<ACharacter>(Owner))
            {
                if (USkeletalMeshComponent* Mesh = Char->GetMesh())
                {
                    const TArray<FName> BonesToFill = PoseRecorder.IsValid()
                        ? PoseRecorder->BonesToCapture
                        : TArray<FName>{ FName("pelvis"), FName("spine_01"), FName("spine_02"),
                                         FName("spine_03"), FName("foot_l"), FName("foot_r"),
                                         FName("hand_l"), FName("hand_r"), FName("head") };

                    CurrentPose.BoneTransforms.Reserve(BonesToFill.Num());
                    for (const FName& BoneName : BonesToFill)
                    {
                        if (Mesh->GetBoneIndex(BoneName) != INDEX_NONE)
                            CurrentPose.BoneTransforms.Add(
                                Mesh->GetBoneTransform(BoneName, RTS_Component));
                        else
                            CurrentPose.BoneTransforms.Add(FTransform::Identity);
                    }
                }
            }
        }
    }

    FCognitiveMotionRequest Req;
    Req.SequenceId        = NextSequenceId++;
    Req.CurrentPose       = CurrentPose;
    Req.DesiredTrajectory = DesiredTrajectory;
    Req.Blackboard        = Blackboard;
    Req.RequestedStyle    = CurrentIdentity.Style;
    Req.MaxLatencyMs      = MaxInferenceLatencyMs;

    LastRequestTime = FPlatformTime::Seconds();
    InferenceSubsystem->SendMotionRequest(Req);
}

void UCognitiveMotionLearnerComponent::PollInferenceResponses()
{
    if (!InferenceSubsystem) return;

    // Limita quantas respostas processamos por tick. Durante o treino, o Python
    // envia em rajadas e a fila pode acumular; drenar TUDO de uma vez num único
    // tick causa picos de frame (o editor "trava" por instantes). Processamos no
    // máximo N por tick — só a MAIS RECENTE importa para a animação; as antigas
    // são descartadas rapidamente sem aplicar bones.
    static constexpr int32 MaxResponsesPerTick = 4;
    int32 Processed = 0;

    FCognitiveMotionResponse Resp;
    while (Processed < MaxResponsesPerTick && InferenceSubsystem->TryGetResponse(Resp))
    {
        ++Processed;
        const double Now = FPlatformTime::Seconds();
        Resp.LatencyMs = (float)((Now - LastRequestTime) * 1000.0);

        if (Resp.LatencyMs > MaxInferenceLatencyMs)
        {
            ActivateFallback(TEXT("Latency exceeded"));
            continue;
        }

        LatestResponse = Resp;
        LastResponseTime = Now;

        // Atualiza estado físico e notifica a AnimBP/ator se mudou.
        if (Resp.PhysicalState != PhysicalState)
        {
            PhysicalState = Resp.PhysicalState;
            OnPhysicalStateChanged.Broadcast(PhysicalState);
            CMI_DBG("[Learner] estado físico → %d", (int32)PhysicalState);
        }

        if (AnimInstance.IsValid())
        {
            AnimInstance->SetEmbedding(Resp.Embedding);
            if (Resp.RefinedTrajectory.IsValid())
                AnimInstance->SetTrajectory(Resp.RefinedTrajectory);

            // Aplica bone transforms do Python diretamente no skeleton do NPC.
            // O AnimNode lê esses transforms e os aplica em Evaluate_AnyThread.
            if (Resp.BoneTransforms.Num() > 0)
            {
                AnimInstance->SetBoneTransforms(Resp.BoneTransforms);
                LastSelectedStyle = static_cast<int32>(Resp.SelectedStyle);
                CMI_DBG("[RECEBIDO←Python] ação=%d | conf=%.2f | bones=%d | "
                        "estado_físico=%d | latência=%.1fms",
                        LastSelectedStyle, Resp.Embedding.Confidence,
                        Resp.BoneTransforms.Num(), (int32)Resp.PhysicalState,
                        Resp.LatencyMs);
            }
        }

        if (bFallbackActive) DeactivateFallback();
        OnResponseReceived.Broadcast(Resp);

        MotionQuality.Confidence  = Resp.Embedding.Confidence;
        MotionQuality.LatencyMs   = Resp.LatencyMs;
    }

    // Se a fila acumulou além do que processamos (rajada de treino), descarta o
    // excesso SEM aplicar — evita crescimento ilimitado de memória e mantém só
    // o estado mais recente. Só a resposta mais nova importa para a animação.
    if (Processed >= MaxResponsesPerTick)
    {
        FCognitiveMotionResponse Discard;
        int32 Drained = 0;
        while (Drained < 64 && InferenceSubsystem->TryGetResponse(Discard))
            ++Drained;
    }

    const double Now = FPlatformTime::Seconds();
    if (LastResponseTime > 0.0 && (Now - LastResponseTime) > (MaxInferenceLatencyMs * 0.002))
    {
        if (!bFallbackActive)
            ActivateFallback(TEXT("Response timeout"));
    }
}

void UCognitiveMotionLearnerComponent::ActivateFallback(const FString& Reason)
{
    if (bFallbackActive) return;
    bFallbackActive = true;
    if (AnimInstance.IsValid())
    {
        AnimInstance->bInferenceFallbackActive = true;
    }
}

void UCognitiveMotionLearnerComponent::DeactivateFallback()
{
    bFallbackActive = false;
    if (AnimInstance.IsValid())
    {
        AnimInstance->bInferenceFallbackActive = false;
    }
}

void UCognitiveMotionLearnerComponent::UpdateFallbackBlend(float DeltaTime)
{
    const float Target = bFallbackActive ? 0.f : 1.f;
    FallbackBlendAlpha = FMath::FInterpTo(FallbackBlendAlpha, Target, DeltaTime, FallbackBlendSpeed);
    if (AnimInstance.IsValid())
        AnimInstance->SetBlendWeight(FallbackBlendAlpha);
}

void UCognitiveMotionLearnerComponent::SetMotionStyle(ECognitiveMotionStyle Style)
{
    CurrentIdentity.Style = Style;
}

void UCognitiveMotionLearnerComponent::EmitDebugDraw() const
{
#if ENABLE_DRAW_DEBUG
    if (!GetOwner() || !GetWorld()) return;

    const FVector Origin = GetOwner()->GetActorLocation() + FVector(0, 0, 100.f);
    const FColor StateColor = bFallbackActive ? FColor::Red : FColor::Green;
    DrawDebugSphere(GetWorld(), Origin, 10.f, 8, StateColor, false, 0.05f);

    if (LatestResponse.bValid && LatestResponse.RefinedTrajectory.IsValid())
    {
        const TArray<FCognitiveTrajectorySample>& Samples = LatestResponse.RefinedTrajectory.Samples;
        for (int32 i = 1; i < Samples.Num(); ++i)
        {
            DrawDebugLine(GetWorld(),
                Samples[i-1].Position, Samples[i].Position,
                FColor::Cyan, false, 0.05f, 0, 1.5f);
        }
    }
#endif
}
