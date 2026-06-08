#include "CognitiveWorldPerceptionComponent.h"
#include "CognitiveEntityTagComponent.h"
#include "CognitiveDebugLog.h"
#include "CognitiveInferenceSubsystem.h"
#include "CognitiveMotionProtocol.h"

#include "GameFramework/Actor.h"
#include "GameFramework/Character.h"
#include "Components/SkeletalMeshComponent.h"
#include "Engine/GameInstance.h"
#include "Engine/World.h"
#include "Engine/OverlapResult.h"
#include "CollisionQueryParams.h"

// ─────────────────────────────────────────────────────────────────────────────
UCognitiveWorldPerceptionComponent::UCognitiveWorldPerceptionComponent()
{
    PrimaryComponentTick.bCanEverTick = true;
}

void UCognitiveWorldPerceptionComponent::BeginPlay()
{
    Super::BeginPlay();

    if (UGameInstance* GI = GetWorld() ? GetWorld()->GetGameInstance() : nullptr)
    {
        if (UCognitiveInferenceSubsystem* Sub =
            GI->GetSubsystem<UCognitiveInferenceSubsystem>())
        {
            InferenceSubsystem = Sub;
        }
    }

    ScanWorld();
}

// ─────────────────────────────────────────────────────────────────────────────
void UCognitiveWorldPerceptionComponent::TickComponent(
    float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

    ScanAccumulator += DeltaTime;
    const float Interval = 1.f / FMath::Max(ScanRateHz, 0.5f);
    if (ScanAccumulator >= Interval)
    {
        ScanAccumulator = 0.f;
        ScanWorld();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
void UCognitiveWorldPerceptionComponent::ScanWorld()
{
    AActor* Self = GetOwner();
    UWorld* World = GetWorld();
    if (!Self || !World) return;

    const FVector SelfLoc = Self->GetActorLocation();
    const FVector SelfFwd = Self->GetActorForwardVector();
    const float   CosHalfFOV = FMath::Cos(FMath::DegreesToRadians(FieldOfViewDegrees * 0.5f));

    TArray<FOverlapResult> Overlaps;
    FCollisionQueryParams Params;
    Params.AddIgnoredActor(Self);

    World->OverlapMultiByObjectType(
        Overlaps,
        SelfLoc,
        FQuat::Identity,
        FCollisionObjectQueryParams(FCollisionObjectQueryParams::AllObjects),
        FCollisionShape::MakeSphere(PerceptionRadius),
        Params);

    PerceivedEntities.Reset();
    PerceivedEntities.Reserve(Overlaps.Num());
    TSet<AActor*> Seen;
    Seen.Reserve(Overlaps.Num());

    for (const FOverlapResult& Ov : Overlaps)
    {
        AActor* Other = Ov.GetActor();
        if (!IsValid(Other) || Other == Self) continue;
        if (Seen.Contains(Other)) continue;

        UCognitiveEntityTagComponent* Tag =
            Other->FindComponentByClass<UCognitiveEntityTagComponent>();
        if (!Tag || Tag->Category == ECognitiveEntityCategory::Ignore) continue;

        Seen.Add(Other);

        const FVector ToTarget = Other->GetActorLocation() - SelfLoc;
        const float   Dist     = ToTarget.Size();
        const FVector Dir      = ToTarget.GetSafeNormal();

        // Campo de visão
        if (FieldOfViewDegrees < 360.f)
        {
            const float Dot = FVector::DotProduct(SelfFwd, Dir);
            if (Dot < CosHalfFOV) continue;
        }

        // Linha de visão
        const bool bLOS = bRequireLineOfSight ? HasLineOfSight(Other) : true;
        if (bRequireLineOfSight && !bLOS) continue;

        FCognitivePerceivedEntity E;
        E.Actor             = Other;
        E.Category          = Tag->Category;
        E.Disposition       = ResolveDisposition(Tag);
        E.Role              = Tag->Role;
        E.Distance          = Dist;
        E.RelativeDirection = Self->GetActorTransform().InverseTransformVectorNoScale(Dir);
        E.bInLineOfSight    = bLOS;
        E.VehicleType       = Tag->VehicleType;
        E.TrafficState      = Tag->TrafficState;
        E.ThreatWeight      = Tag->ThreatWeight;
        E.SuggestedReaction = SuggestReaction(E, Tag);

        PerceivedEntities.Add(E);
        OnEntityPerceived.Broadcast(E);
    }

    // Ordena por distância (mais próximo primeiro)
    PerceivedEntities.Sort([](const FCognitivePerceivedEntity& A,
                              const FCognitivePerceivedEntity& B)
    {
        return A.Distance < B.Distance;
    });

    CMI_DBG("[Perception] %s percebeu %d entidades (raio=%.0f)",
            *Self->GetName(), PerceivedEntities.Num(), PerceptionRadius);

    // Diagnóstico: se nada foi percebido, explica o porquê provável. Atores só
    // são percebidos se tiverem UCognitiveEntityTagComponent. Durante o treino,
    // ver 0 é NORMAL — o líder e os objetos da cena geralmente não têm tag.
    if (PerceivedEntities.Num() == 0 && Overlaps.Num() > 0)
    {
        CMI_DBG("[Perception] 0 percebidas, mas %d atores no raio: nenhum tem "
                "'Cognitive Entity Tag' (ou estão fora do FOV de %.0f graus, ou "
                "Category=Ignore). Adicione o componente Entity Tag aos atores "
                "que o NPC deve perceber.", Overlaps.Num(), FieldOfViewDegrees);
    }
    else if (PerceivedEntities.Num() == 0)
    {
        CMI_DBG("[Perception] 0 percebidas e 0 atores no raio. Durante o treino "
                "isso é normal (sem objetos tagueados na cena).");
    }

    // Envia a percepção ao Python para que o world model/policy possam decidir
    // com base no ambiente (não só na dinâmica do próprio corpo).
    SendPerceptionToPython();
}

// ─────────────────────────────────────────────────────────────────────────────
void UCognitiveWorldPerceptionComponent::SendPerceptionToPython()
{
    if (!InferenceSubsystem.IsValid()) return;
    if (!InferenceSubsystem->IsReady()) return;   // só envia se conectado

    CognitiveMotionProtocol::FPerceptionPayload Payload;
    Payload.NPCId = 0;   // sessão TCP única; o Python associa pela conexão
    Payload.Entities.Reserve(PerceivedEntities.Num());

    for (const FCognitivePerceivedEntity& E : PerceivedEntities)
    {
        CognitiveMotionProtocol::FPerceivedEntityWire W;
        W.Category     = static_cast<int32>(E.Category);
        W.Disposition  = static_cast<int32>(E.Disposition);
        W.Role         = static_cast<int32>(E.Role);
        W.Reaction     = static_cast<int32>(E.SuggestedReaction);
        W.VehicleType  = static_cast<int32>(E.VehicleType);
        W.TrafficState = static_cast<int32>(E.TrafficState);
        W.Distance     = E.Distance;
        W.DirX         = E.RelativeDirection.X;
        W.DirY         = E.RelativeDirection.Y;
        W.DirZ         = E.RelativeDirection.Z;
        W.ThreatWeight = E.ThreatWeight;
        Payload.Entities.Add(W);
    }

    const TArray<uint8> Data = CognitiveMotionProtocol::SerializePerception(Payload);
    if (Data.Num() > 0)
    {
        InferenceSubsystem->SendRawMessage(Data);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
ECognitiveDisposition UCognitiveWorldPerceptionComponent::ResolveDisposition(
    const UCognitiveEntityTagComponent* Tag) const
{
    if (!Tag) return ECognitiveDisposition::Neutral;

    // Mesma facção = amigo automático
    if (SelfFaction != NAME_None && Tag->Faction == SelfFaction)
        return ECognitiveDisposition::Friend;

    return Tag->Disposition;
}

// ─────────────────────────────────────────────────────────────────────────────
ECognitiveReaction UCognitiveWorldPerceptionComponent::SuggestReaction(
    const FCognitivePerceivedEntity& E,
    const UCognitiveEntityTagComponent* Tag) const
{
    if (!Tag) return ECognitiveReaction::None;

    switch (E.Category)
    {
        case ECognitiveEntityCategory::Character:
            if (E.Disposition == ECognitiveDisposition::Enemy)
            {
                // Inimigo forte e perto → fugir/esconder; senão atacar
                if (E.ThreatWeight > 0.7f && E.Distance < 600.f)
                    return ECognitiveReaction::Flee;
                return ECognitiveReaction::Attack;
            }
            if (E.Disposition == ECognitiveDisposition::Friend ||
                E.Disposition == ECognitiveDisposition::Ally)
                return ECognitiveReaction::Approach;
            return ECognitiveReaction::None;

        case ECognitiveEntityCategory::Weapon:
        case ECognitiveEntityCategory::Pickup:
            return Tag->bCanPickUp ? ECognitiveReaction::PickUp : ECognitiveReaction::None;

        case ECognitiveEntityCategory::Vehicle:
            return Tag->bCanEnter ? ECognitiveReaction::Enter : ECognitiveReaction::None;

        case ECognitiveEntityCategory::TrafficLight:
            // Verde = pode atravessar; vermelho/amarelo = esperar
            return (Tag->TrafficState == ECognitiveTrafficState::Green)
                ? ECognitiveReaction::Cross
                : ECognitiveReaction::Wait;

        case ECognitiveEntityCategory::Hazard:
            return ECognitiveReaction::Flee;

        case ECognitiveEntityCategory::Cover:
            return ECognitiveReaction::Hide;

        case ECognitiveEntityCategory::Objective:
            return ECognitiveReaction::Approach;

        default:
            return ECognitiveReaction::None;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
bool UCognitiveWorldPerceptionComponent::HasLineOfSight(AActor* Target) const
{
    AActor* Self = GetOwner();
    UWorld* World = GetWorld();
    if (!Self || !World || !Target) return false;

    FHitResult Hit;
    FCollisionQueryParams Params;
    Params.AddIgnoredActor(Self);
    Params.AddIgnoredActor(Target);

    const FVector Start = Self->GetActorLocation();
    const FVector End   = Target->GetActorLocation();

    const bool bBlocked = World->LineTraceSingleByChannel(
        Hit, Start, End, ECollisionChannel::ECC_Visibility, Params);

    return !bBlocked;  // sem bloqueio = tem linha de visão
}

// ─────────────────────────────────────────────────────────────────────────────
FCognitivePerceivedEntity UCognitiveWorldPerceptionComponent::GetNearestThreat() const
{
    for (const FCognitivePerceivedEntity& E : PerceivedEntities)
    {
        if (E.Category == ECognitiveEntityCategory::Character &&
            E.Disposition == ECognitiveDisposition::Enemy)
            return E;
        if (E.Category == ECognitiveEntityCategory::Hazard)
            return E;
    }
    return FCognitivePerceivedEntity();
}

FCognitivePerceivedEntity UCognitiveWorldPerceptionComponent::GetNearestPickup() const
{
    for (const FCognitivePerceivedEntity& E : PerceivedEntities)
    {
        if ((E.Category == ECognitiveEntityCategory::Weapon ||
             E.Category == ECognitiveEntityCategory::Pickup) &&
            E.SuggestedReaction == ECognitiveReaction::PickUp)
            return E;
    }
    return FCognitivePerceivedEntity();
}

ECognitiveTrafficState UCognitiveWorldPerceptionComponent::GetNearestTrafficState() const
{
    for (const FCognitivePerceivedEntity& E : PerceivedEntities)
    {
        if (E.Category == ECognitiveEntityCategory::TrafficLight)
            return E.TrafficState;
    }
    return ECognitiveTrafficState::Unknown;
}

// ─────────────────────────────────────────────────────────────────────────────
// Interação: segurar objeto no skeleton
// ─────────────────────────────────────────────────────────────────────────────
bool UCognitiveWorldPerceptionComponent::AttachObjectToHand(AActor* Object, FName SocketName)
{
    AActor* Self = GetOwner();
    if (!IsValid(Self) || !IsValid(Object))
    {
        CMI_DBG("[Perception] AttachObjectToHand: ator inválido");
        return false;
    }

    // Aceita qualquer ator com SkeletalMeshComponent (não só ACharacter).
    // Prioriza o mesh do Character, mas cai para o primeiro skeletal mesh achado.
    USkeletalMeshComponent* Mesh = nullptr;
    if (ACharacter* NPC = Cast<ACharacter>(Self))
    {
        Mesh = NPC->GetMesh();
    }
    if (!Mesh)
    {
        Mesh = Self->FindComponentByClass<USkeletalMeshComponent>();
    }
    if (!Mesh)
    {
        CMI_DBG("[Perception] AttachObjectToHand: %s não tem SkeletalMeshComponent",
                *Self->GetName());
        return false;
    }

    // Verifica se o socket existe no skeleton
    if (SocketName != NAME_None && !Mesh->DoesSocketExist(SocketName))
    {
        CMI_DBG("[Perception] socket '%s' não existe no skeleton de %s",
                *SocketName.ToString(), *Self->GetName());
        return false;
    }

    USceneComponent* ObjRoot = Object->GetRootComponent();
    if (!ObjRoot) return false;

    // Desativa física do objeto antes de anexar (se for primitiva simulando)
    if (UPrimitiveComponent* Prim = Cast<UPrimitiveComponent>(ObjRoot))
    {
        Prim->SetSimulatePhysics(false);
        Prim->SetCollisionEnabled(ECollisionEnabled::NoCollision);
    }

    const FAttachmentTransformRules Rules(EAttachmentRule::SnapToTarget, true);
    const bool bOk = Object->AttachToComponent(Mesh, Rules, SocketName);
    if (bOk)
    {
        HeldObject = Object;
        CMI_DBG("[Perception] %s pegou %s no socket %s",
                *Self->GetName(), *Object->GetName(), *SocketName.ToString());
    }
    return bOk;
}

void UCognitiveWorldPerceptionComponent::DropHeldObject()
{
    AActor* Obj = HeldObject.Get();
    if (!IsValid(Obj)) return;

    const FDetachmentTransformRules Rules(EDetachmentRule::KeepWorld, true);
    Obj->DetachFromActor(Rules);

    if (UPrimitiveComponent* Prim = Cast<UPrimitiveComponent>(Obj->GetRootComponent()))
    {
        Prim->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
        Prim->SetSimulatePhysics(true);
    }

    CMI_DBG("[Perception] %s soltou %s",
            GetOwner() ? *GetOwner()->GetName() : TEXT("?"), *Obj->GetName());
    HeldObject = nullptr;
}
