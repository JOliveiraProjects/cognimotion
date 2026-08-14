#include "CognitiveNPCBoneDriver.h"
#include "CognitiveDebugLog.h"
#include "CognitiveLeaderObserverComponent.h"
#include "CognitiveAnimInstance.h"
#include "CognitiveMotionLearnerComponent.h"
#include "CognitiveNativeInferenceComponent.h"
#include "GameFramework/Character.h"
#include "GameFramework/CharacterMovementComponent.h"
#include "Engine/GameInstance.h"
#include "HAL/PlatformTime.h"
#include "NavigationSystem.h"
#include "Blueprint/AIBlueprintHelperLibrary.h"
#include "CognitiveInferenceSubsystem.h"
#include "CognitiveAnimInstance.h"
#include "CognitiveLeaderObserverComponent.h"
#include "GameFramework/Character.h"
#include "GameFramework/CharacterMovementComponent.h"
#include "Components/SkeletalMeshComponent.h"
#include "Engine/World.h"
#include "HAL/PlatformTime.h"

UCognitiveNPCBoneDriver::UCognitiveNPCBoneDriver()
{
    PrimaryComponentTick.bCanEverTick = true;
    PrimaryComponentTick.TickInterval = 0.f;
}

void UCognitiveNPCBoneDriver::BeginPlay()
{
    Super::BeginPlay();
    SendInterval = SendRateHz > 0.f ? 1.f / SendRateHz : 1.f / 30.f;

    // Proteção: GetWorld()/GetGameInstance() podem ser nulos no editor ou em
    // PIE inicial. Sem isso, o acesso ao subsystem causa crash.
    if (UWorld* World = GetWorld())
    {
        if (UGameInstance* GI = World->GetGameInstance())
        {
            InferenceSubsystem = GI->GetSubsystem<UCognitiveInferenceSubsystem>();
        }
    }
    if (!InferenceSubsystem.IsValid())
    {
        CMI_DBG("[BoneDriver] AVISO: subsistema de inferência indisponível "
                "(GameInstance nulo?). Envio ao Python desabilitado neste NPC.");
    }

    if (ACharacter* Char = Cast<ACharacter>(GetOwner()))
    {
        CachedMesh = Char->GetMesh();
        if (CachedMesh.IsValid())
        {
            CachedAnimInstance = Cast<UCognitiveAnimInstance>(
                CachedMesh->GetAnimInstance());

            // Captura nomes de todos os bones do skeleton
            const int32 N = CachedMesh->GetNumBones();
            CachedBoneNames.SetNum(N);
            for (int32 i = 0; i < N; ++i)
                CachedBoneNames[i] = CachedMesh->GetBoneName(i);
        }
    }

    // Cache LeaderObserver e RemoveRecvQueue competition
    CachedLeaderObserver = GetOwner()
        ? GetOwner()->FindComponentByClass<UCognitiveLeaderObserverComponent>()
        : nullptr;

    if (bAutoConnect && InferenceSubsystem.IsValid())
    {
        // Evita duplo Connect: LearnerComponent pode ter conectado primeiro.
        // Se já está Ready ou Connecting, não reconecta (derrubaria a conexão existente).
        const ECognitiveInferenceState State = InferenceSubsystem->GetConnectionState();
        if (State == ECognitiveInferenceState::Disconnected)
        {
            InferenceSubsystem->Connect(PythonHost, PythonPort, CachedBoneNames);
        }
        else
        {
            // Apenas atualiza os bone names no subsystem sem reconectar
            InferenceSubsystem->Connect(TEXT(""), 0, CachedBoneNames); // overload só armazena names
        }
    }

    // Cache do LeaderObserver — usado para replicar movimento físico do líder
    CachedLeaderObserver = GetOwner()
        ? GetOwner()->FindComponentByClass<UCognitiveLeaderObserverComponent>()
        : nullptr;

    // ── Promoção automática para modo Imported ───────────────────────────────
    // Se o NPC tem um Cognitive Native Inference Component com o .pt carregado,
    // o caminho de inferência nativa (RunInference) só roda em Imported/Inferring.
    // Sem isto, o default Observing nunca chama o modelo e o NPC só replica o
    // líder (flutuando/deslizando). Promover aqui evita depender do checkbox no
    // Details e garante que o worker isolado seja efetivamente usado.
    if (ObservationState == ECognitiveObservationState::Observing && GetOwner())
    {
        if (UCognitiveNativeInferenceComponent* Native =
                GetOwner()->FindComponentByClass<UCognitiveNativeInferenceComponent>())
        {
            if (Native->IsModelLoaded())
            {
                ObservationState = ECognitiveObservationState::Imported;
                CMI_DBG("[BoneDriver] modelo nativo (.pt) carregado — "
                        "ObservationState promovido para Imported.");
            }
        }
    }
}

void UCognitiveNPCBoneDriver::EndPlay(const EEndPlayReason::Type Reason)
{
    Super::EndPlay(Reason);
}

void UCognitiveNPCBoneDriver::TickComponent(
    float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
    if (!InferenceSubsystem.IsValid()) return;

    // ── PAINEL DE STATUS (lado Unreal) a cada ~5s — espelha o painel Python ──
    StatusPanelAccumulator += DeltaTime;
    if (StatusPanelAccumulator >= 5.0f)
    {
        StatusPanelAccumulator = 0.f;
        const TCHAR* ConnStr = InferenceSubsystem->IsReady() ? TEXT("CONECTADO ✓")
                                                             : TEXT("DESCONECTADO ✗");
        CMI_DBG("┌── COGNITIVE NPC (lado Unreal) ──\n"
                "  Conexão Python: %s (%s:%d)\n"
                "  Estado: %s | treino=%s | locomoção=%d\n"
                "  → Enviados: %d frames | bones/frame=%d\n"
                "  ← Última resposta: ação aplicada, %d bones | conf=%.2f | %.1fms\n"
                "└────────────────────────────────",
                ConnStr, *PythonHost, PythonPort,
                *GetObservationStateString(), *BehaviorContext.ToKey(),
                (int32)BehaviorContext.LocomotionState,
                TotalRequestsSent, CachedBoneNames.Num(),
                BonesApplied, LatestResponse.Confidence, LastLatencyMs);
    }

    // RecvQueue lida EXCLUSIVAMENTE pelo LearnerComponent
    // BoneDriver só lê estado do AnimInstance para debug
    if (CachedAnimInstance.IsValid())
        BonesApplied = CachedAnimInstance->HasValidBoneTransforms()
                       ? CachedAnimInstance->GetBoneTransforms().Num() : 0;

    // ── Envia frame do NPC ao Python ─────────────────────────────────────────
    if (ObservationState == ECognitiveObservationState::Observing)
    {
        SendAccumulator += DeltaTime;
        if (SendAccumulator >= SendInterval)
        {
            SendAccumulator -= SendInterval;

            FCognitiveBoneFrame Frame = BuildNPCFrame();
            if (Frame.IsValid())
            {
                InferenceSubsystem->SendBoneFrame(Frame);
                LastRequestTime = FPlatformTime::Seconds();
                ++TotalRequestsSent;

                // TELEMETRIA: o que o plugin ENVIA ao Python (a cada 30 envios
                // para não poluir, ~1x/s a 30Hz).
                if ((TotalRequestsSent % 30) == 0)
                {
                    CMI_DBG("[ENVIO→Python] treino=%s | estado_loco=%d | bones=%d | "
                            "enviados=%d",
                            *BehaviorContext.ToKey(),
                            (int32)BehaviorContext.LocomotionState,
                            Frame.BoneTransforms.Num(),
                            TotalRequestsSent);
                }
            }
        }
    }

    // ── Movimento e rotação — state-aware ───────────────────────────────────
    // Inferring  = Python controla em tempo real (teste durante o treino).
    // Imported   = modelo .pt roda NATIVO via LibTorch (sem rede, produção).
    if (ObservationState == ECognitiveObservationState::Inferring ||
        ObservationState == ECognitiveObservationState::Imported)
    {
        // ── MODO AUTÔNOMO: Python controla o NPC ─────────────────────────────
        // O world model GERA as poses dos 89 bones (PoseDecoder) e o
        // LearnerComponent as aplica via SetBoneTransforms — o NPC anima sozinho,
        // com o movimento aprendido do líder. Aqui só traduzimos a ação discreta
        // em deslocamento físico da cápsula (locomoção no mundo).
        // OBS: se o PoseDecoder ainda não foi treinado, o Python faz fallback
        // para os bones do líder; e o AnimNode tem SourcePose como rede de
        // segurança caso nenhum bone chegue.
        ACharacter* NPCChar = Cast<ACharacter>(GetOwner());
        if (NPCChar)
        {
            UCharacterMovementComponent* NPCMove = NPCChar->GetCharacterMovement();
            if (NPCMove && CachedAnimInstance.IsValid())
            {
                // Obtém a última ação do Python via LearnerComponent
                UCognitiveMotionLearnerComponent* Learner =
                    GetOwner()->FindComponentByClass<UCognitiveMotionLearnerComponent>();
                int32 Action = Learner ? Learner->LastSelectedStyle : 0;

                // ── INFERÊNCIA NATIVA (LibTorch, sem rede/Python) ──────────────
                // Em Imported é o caminho principal; em Inferring é aceleração.
                UCognitiveNativeInferenceComponent* Native =
                    GetOwner()->FindComponentByClass<UCognitiveNativeInferenceComponent>();
                if (Native && Native->IsModelLoaded())
                {
                    TArray<FTransform> GenBones;
                    const TArray<float> EmptyObs;
                    const int32 NativeAction = Native->RunInference(EmptyObs, GenBones);
                    if (NativeAction >= 0)
                    {
                        Action = NativeAction;
                        if (GenBones.Num() > 0)
                            CachedAnimInstance->SetBoneTransforms(GenBones);
                    }
                }
                else if (ObservationState == ECognitiveObservationState::Imported)
                {
                    // Em Imported sem modelo nativo, nada anima. Diagnóstico claro.
                    if ((TotalRequestsSent++ % 120) == 0)
                    {
                        if (!Native)
                            CMI_DBG("[NativeInfer] FALTA o componente 'Cognitive Native "
                                    "Inference' no NPC para o modo Imported funcionar.");
                        else
                            CMI_DBG("[NativeInfer] modelo .pt NÃO carregado. Confira "
                                    "Content/Models/CognitiveModel.pt ou o ModelPath.");
                    }
                }

                // Mapa de ações: deve espelhar ActionExecutor no Python
                // 0=idle, 1=fwd, 2=back, 3=left, 4=right, 5=run_fwd, 6=jump, 7=crouch, 8=stop
                const float Walk = 300.f;
                const float Run  = 600.f;
                const FVector Fwd   = NPCChar->GetActorForwardVector();
                const FVector Right = NPCChar->GetActorRightVector();

                NPCMove->bOrientRotationToMovement = false;
                if (NPCMove->MaxWalkSpeed < Run)
                    NPCMove->MaxWalkSpeed = Run;

                // Resolve a direção desejada da ação (em espaço do mundo).
                FVector MoveDir = FVector::ZeroVector;
                float   MoveSpeed = Walk;
                switch (Action)
                {
                    case 1: MoveDir = Fwd;    MoveSpeed = Walk; break;
                    case 2: MoveDir = -Fwd;   MoveSpeed = Walk; break;
                    case 3: MoveDir = -Right; MoveSpeed = Walk; break;
                    case 4: MoveDir = Right;  MoveSpeed = Walk; break;
                    case 5: MoveDir = Fwd;    MoveSpeed = Run;  break;
                    case 7: MoveDir = Fwd;    MoveSpeed = 100.f; break;
                    default: break;
                }

                if (Action == 6)
                {
                    if (!NPCMove->IsFalling() && !bJumpTriggered)
                    {
                        NPCChar->Jump();
                        bJumpTriggered = true;
                    }
                }
                else if (Action == 0 || Action == 8)
                {
                    bJumpTriggered = false;  // reset no idle/stop
                    NPCMove->StopMovementImmediately();
                }
                else if (!MoveDir.IsNearlyZero())
                {
                    // NAVMESH: interpreta a ação como destino e navega evitando
                    // obstáculos. Se o NavMesh não estiver disponível ou o ponto
                    // não projetar, cai no movimento direto (comportamento legado).
                    bool bNavHandled = false;
                    if (bUseNavMesh)
                    {
                        bNavHandled = NavigateByAction(NPCChar, MoveDir, MoveSpeed);
                    }
                    if (!bNavHandled)
                    {
                        NPCMove->RequestDirectMove(MoveDir * MoveSpeed, false);
                    }
                }
            }
        }
    }
    else if (bReplicateLeaderMovement
             && CachedLeaderObserver.IsValid()
             && (ObservationState == ECognitiveObservationState::Observing ||
              ObservationState == ECognitiveObservationState::Inferring ||
              ObservationState == ECognitiveObservationState::Imported))
    {
        // ── MODO OBSERVAÇÃO/REPLICAÇÃO: copia líder ───────────────────────────
        AActor*     Leader  = CachedLeaderObserver->TargetLeader;
        ACharacter* NPCChar = Cast<ACharacter>(GetOwner());

        if (Leader && NPCChar)
        {
            const FVector LeaderVelocity = Leader->GetVelocity();
            const float   Speed2D        = LeaderVelocity.Size2D();

            UCharacterMovementComponent* NPCMove = NPCChar->GetCharacterMovement();
            if (NPCMove)
            {
                const bool bBonesReady = CachedAnimInstance.IsValid()
                                         && CachedAnimInstance->HasValidBoneTransforms();

                // Rotação: copia a do líder
                if (bBonesReady)
                {
                    NPCMove->bOrientRotationToMovement = false;
                    NPCChar->SetActorRotation(Leader->GetActorQuat());
                }

                // Velocidade
                if (bBonesReady && Speed2D > MovementThreshold)
                {
                    if (Speed2D > NPCMove->MaxWalkSpeed)
                        NPCMove->MaxWalkSpeed = Speed2D * 1.1f;
                    NPCMove->RequestDirectMove(
                        FVector(LeaderVelocity.X, LeaderVelocity.Y, 0.f), false);
                }
                else
                {
                    NPCMove->StopMovementImmediately();
                }

                // Pulo
                ACharacter* LeaderChar = Cast<ACharacter>(Leader);
                if (LeaderChar)
                {
                    UCharacterMovementComponent* LM = LeaderChar->GetCharacterMovement();
                    const bool bLeaderFalling = LM && LM->IsFalling();
                    const bool bNPCFalling    = NPCMove->IsFalling();

                    // bJumpTriggered evita múltiplos Jump() enquanto líder
                    // está no ar (seria chamado 60x/s sem a flag)
                    if (bLeaderFalling && !bNPCFalling && !bJumpTriggered)
                    {
                        NPCChar->Jump();
                        bJumpTriggered = true;
                    }
                    else if (!bLeaderFalling)
                    {
                        bJumpTriggered = false;  // reset quando líder pousa
                    }
                }
            }
        }
    }
}

FCognitiveBoneFrame UCognitiveNPCBoneDriver::BuildNPCFrame() const
{
    FCognitiveBoneFrame Frame;
    if (!CachedMesh.IsValid()) return Frame;

    Frame.Timestamp       = FPlatformTime::Seconds();
    Frame.SequenceId      = NextSequenceId;
    Frame.BehaviorContext = BehaviorContext;
    Frame.BoneNames       = CachedBoneNames;

    Frame.BoneTransforms = CachedMesh->GetComponentSpaceTransforms();
    const_cast<UCognitiveNPCBoneDriver*>(this)->NextSequenceId++;
    return Frame;
}

void UCognitiveNPCBoneDriver::ApplyBoneTransforms(const FCognitiveBoneResponse& Response)
{
    if (!Response.bValid || Response.BoneTransforms.IsEmpty()) return;

    // Aplica via UCognitiveAnimInstance → AnimNode lê em Evaluate_AnyThread.
    // SetBoneTransformByName não existe em USkeletalMeshComponent no UE5.
    // A forma correta é alimentar a AnimInstance que já está conectada ao AnimNode.
    if (CachedAnimInstance.IsValid())
    {
        CachedAnimInstance->SetBoneTransforms(Response.BoneTransforms);
    }

    // Root motion (opcional)
    if (Response.bApplyRootMotion && GetOwner())
    {
        // Origem do root motion:
        //  - Online (servidor): usa Response.RootLocation/RootRotation.
        //  - Offline/.pt: o servidor não manda root explícito, então extrai do
        //    root bone (índice 0) das poses geradas. Assim a cápsula segue a
        //    trajetória real gerada pelo modelo, sem deslizar.
        FVector TargetLoc = Response.RootLocation;
        FQuat   TargetRot = Response.RootRotation;

        const bool bHasServerRoot =
            !Response.RootLocation.IsNearlyZero() || !Response.RootRotation.IsIdentity();
        if (!bHasServerRoot && Response.BoneTransforms.Num() > 0)
        {
            // Root bone = índice 0 (convenção do PoseDecoder: bone 0 é root/pelvis).
            // A pose é local ao ator; compõe com a transform atual para obter
            // o deslocamento no mundo.
            const FTransform& RootBone = Response.BoneTransforms[0];
            TargetLoc = GetOwner()->GetActorTransform().TransformPosition(
                RootBone.GetLocation());
            TargetRot = GetOwner()->GetActorQuat() * RootBone.GetRotation();
        }

        const float Alpha = FMath::Clamp(BlendAlpha, 0.f, 1.f);
        const FVector CurLoc = GetOwner()->GetActorLocation();
        const FQuat   CurRot = GetOwner()->GetActorQuat();
        const FVector NewLoc = FMath::Lerp(CurLoc, TargetLoc, Alpha);
        const FQuat   NewRot = FQuat::Slerp(CurRot, TargetRot, Alpha);
        GetOwner()->SetActorLocationAndRotation(NewLoc, NewRot, /*bSweep=*/false,
            nullptr, ETeleportType::TeleportPhysics);
    }
}

void UCognitiveNPCBoneDriver::SetObservationState(ECognitiveObservationState NewState)
{
    ObservationState = NewState;
    UE_LOG(LogTemp, Log, TEXT("[CognitiveNPCBoneDriver] State → %s"),
        *GetObservationStateString());
}

void UCognitiveNPCBoneDriver::SetTrainingContext(
    ECognitiveTrainingCategory Category, const FString& Subtype)
{
    BehaviorContext.Category = Category;
    BehaviorContext.Subtype  = Subtype;
    BehaviorContext.CustomCategoryName.Empty();
}

void UCognitiveNPCBoneDriver::SetLocomotionState(ECognitiveLocomotionState State)
{
    BehaviorContext.LocomotionState = State;
}

FString UCognitiveNPCBoneDriver::GetObservationStateString() const
{
    switch (ObservationState)
    {
    case ECognitiveObservationState::Observing:   return TEXT("Observing Leader");
    case ECognitiveObservationState::Inferring:   return TEXT("Inferring from Python");
    case ECognitiveObservationState::Imported:    return TEXT("Imported Model");
    default: return TEXT("Unknown");
    }
}

FString UCognitiveNPCBoneDriver::GetDiagnostics() const
{
    return FString::Printf(
        TEXT("NPCBoneDriver | State=%s | Bones=%d | Applied=%d | Latency=%.1fms | Requests=%d | Confidence=%.2f"),
        *GetObservationStateString(),
        CachedBoneNames.Num(),
        BonesApplied,
        LastLatencyMs,
        TotalRequestsSent,
        LatestResponse.Confidence);
}

// ─────────────────────────────────────────────────────────────────────────────
// NavigateByAction — converte a ação direcional num destino e navega pelo
// NavMesh do UE5, respeitando obstáculos. Retorna true se a navegação foi
// despachada; false se o NavMesh/Controller não estiverem disponíveis (o
// chamador então usa o movimento direto como fallback).
bool UCognitiveNPCBoneDriver::NavigateByAction(
    ACharacter* NPCChar, const FVector& MoveDir, float Speed)
{
    if (!NPCChar) return false;

    UWorld* World = NPCChar->GetWorld();
    if (!World) return false;

    UNavigationSystemV1* NavSys = FNavigationSystem::GetCurrent<UNavigationSystemV1>(World);
    if (!NavSys) return false;

    // Destino desejado: à frente da direção da ação, a NavProbeDistance.
    const FVector Origin = NPCChar->GetActorLocation();
    const FVector Destination = Origin + MoveDir.GetSafeNormal() * NavProbeDistance;

    // Projeta o destino no NavMesh (acha o ponto navegável mais próximo).
    FNavLocation NavDest;
    const FVector Extent(150.f, 150.f, 300.f);
    if (!NavSys->ProjectPointToNavigation(Destination, NavDest, Extent))
    {
        return false;  // sem ponto navegável — fallback para movimento direto
    }

    // Precisa de um controlador para emitir o move. Sem controller, fallback.
    AController* Ctrl = NPCChar->GetController();
    if (!Ctrl)
    {
        return false;
    }

    if (UCharacterMovementComponent* Move = NPCChar->GetCharacterMovement())
    {
        Move->MaxWalkSpeed = Speed;
        Move->SetMovementMode(MOVE_Walking);
    }

    // SimpleMoveToLocation usa o NavMesh para gerar o caminho e seguir,
    // evitando obstáculos automaticamente.
    UAIBlueprintHelperLibrary::SimpleMoveToLocation(Ctrl, NavDest.Location);
    return true;
}
