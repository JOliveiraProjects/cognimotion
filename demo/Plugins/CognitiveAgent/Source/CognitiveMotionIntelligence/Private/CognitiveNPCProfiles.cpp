#include "CognitiveNPCProfiles.h"
#include "CognitiveNPCStateMachine.h"
#include "GameFramework/Actor.h"

UCognitiveNPCProfileComponent::UCognitiveNPCProfileComponent()
{
    PrimaryComponentTick.bCanEverTick = true;
    PrimaryComponentTick.TickInterval = 0.1f;
}

void UCognitiveNPCProfileComponent::BeginPlay()
{
    Super::BeginPlay();
    StateMachine = GetOwner()->FindComponentByClass<UCognitiveNPCStateMachine>();
    SetProfile(DefaultProfile);
}

void UCognitiveNPCProfileComponent::SetProfile(ECognitiveNPCProfile InProfile)
{
    ActiveProfile   = InProfile;
    CurrentSubState = 0;
    TimeInSubState  = 0.f;
}

bool UCognitiveNPCProfileComponent::TransitionSubState(int32 NewSubStateIndex)
{
    if (NewSubStateIndex == CurrentSubState) return false;
    CurrentSubState = NewSubStateIndex;
    TimeInSubState  = 0.f;

    // BM-06 FIX: o código original fazia Broadcast(ECognitiveNPCState::Idle) sempre,
    // independente do sub-estado que mudou — listeners recebiam informação incorreta.
    // FCognitiveStateChangeDelegate é OnParam(ECognitiveNPCState), cuja assinatura não
    // representa bem sub-estados (que são int32 por perfil). Soluções possíveis:
    //   1. Manter delegate e passar CurrentState da FSM (contexto correto).
    //   2. Criar um delegate OnSubStateChanged(int32, int32) mais preciso.
    // Optamos por (1) — passa o estado NPC atual da FSM como contexto,
    // que é o dado mais relevante para listeners (ex: AnimBP saber que o NPC
    // mudou sub-estado durante Combat vs Idle).
    ECognitiveNPCState CurrentNPCState = ECognitiveNPCState::Idle;
    if (StateMachine.IsValid())
    {
        CurrentNPCState = StateMachine->GetCurrentState();
    }
    OnSubStateChanged.Broadcast(CurrentNPCState);
    return true;
}

void UCognitiveNPCProfileComponent::TickComponent(
    float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* Func)
{
    Super::TickComponent(DeltaTime, TickType, Func);
    TimeInSubState += DeltaTime;
    SubStateAccum  += DeltaTime;

    switch (ActiveProfile)
    {
    case ECognitiveNPCProfile::CivilianUrban: EvaluateCivilianTransitions(DeltaTime); break;
    case ECognitiveNPCProfile::Driver:        EvaluateDriverTransitions(DeltaTime);   break;
    case ECognitiveNPCProfile::Criminal:      EvaluateCriminalTransitions(DeltaTime); break;
    case ECognitiveNPCProfile::Fighter:       EvaluateFighterTransitions(DeltaTime);  break;
    case ECognitiveNPCProfile::Soldier:       EvaluateSoldierTransitions(DeltaTime);  break;
    case ECognitiveNPCProfile::Survivor:      EvaluateSurvivorTransitions(DeltaTime); break;
    case ECognitiveNPCProfile::Infected:      EvaluateInfectedTransitions(DeltaTime); break;
    default: break;
    }
}

void UCognitiveNPCProfileComponent::SetCarrying(bool bIsCarrying, float CarryWeightKg)
{
    bCarrying   = bIsCarrying;
    CarryWeight = CarryWeightKg;
}

void UCognitiveNPCProfileComponent::SetWantedLevel(int32 Level)
{
    WantedLevel = FMath::Clamp(Level, 0, 5);
}

ECognitiveMotionStyle UCognitiveNPCProfileComponent::GetMotionStyleForCurrentState() const
{
    switch (ActiveProfile)
    {
    case ECognitiveNPCProfile::Soldier:  return ECognitiveMotionStyle::Military;
    case ECognitiveNPCProfile::Criminal: return ECognitiveMotionStyle::Stealth;
    case ECognitiveNPCProfile::Survivor: return ECognitiveMotionStyle::Stealth;
    case ECognitiveNPCProfile::Fighter:  return ECognitiveMotionStyle::Aggressive;
    case ECognitiveNPCProfile::Infected: return ECognitiveMotionStyle::Aggressive;
    default:
        return StateMachine.IsValid() && StateMachine->GetBlackboard().Health < 25.f
            ? ECognitiveMotionStyle::Injured
            : StateMachine.IsValid() && StateMachine->GetBlackboard().Fatigue > 70.f
                ? ECognitiveMotionStyle::Fatigued
                : ECognitiveMotionStyle::Neutral;
    }
}

ECognitiveMovementMode UCognitiveNPCProfileComponent::GetMovementModeForCurrentState() const
{
    if (!StateMachine.IsValid()) return ECognitiveMovementMode::Walk;

    switch (StateMachine->GetCurrentState())
    {
    case ECognitiveNPCState::Flee:
    case ECognitiveNPCState::Combat:     return ECognitiveMovementMode::Sprint;
    case ECognitiveNPCState::Stealth:    return ECognitiveMovementMode::Crouch;
    case ECognitiveNPCState::Idle:       return ECognitiveMovementMode::Idle;
    case ECognitiveNPCState::TakeCover:  return ECognitiveMovementMode::Crouch;
    case ECognitiveNPCState::Investigate: return ECognitiveMovementMode::Walk;
    default:                             return ECognitiveMovementMode::Walk;
    }
}

void UCognitiveNPCProfileComponent::EvaluateCivilianTransitions(float DeltaTime)
{
    if (!StateMachine.IsValid()) return;
    const ECognitiveNPCState NPCState = StateMachine->GetCurrentState();

    if (NPCState == ECognitiveNPCState::Flee)
    {
        TransitionSubState((int32)ECognitiveCivilianSubState::PanicFlee);
        return;
    }
    if (NPCState == ECognitiveNPCState::Surrender)
    {
        TransitionSubState((int32)ECognitiveCivilianSubState::Surrender);
        return;
    }
    if (NPCState == ECognitiveNPCState::Hide)
    {
        TransitionSubState((int32)ECognitiveCivilianSubState::HideBehindCar);
        return;
    }

    const float Health = StateMachine->GetBlackboard().Health;
    if (Health < 50.f && CurrentSubState != (int32)ECognitiveCivilianSubState::InjuredLimp)
        TransitionSubState((int32)ECognitiveCivilianSubState::InjuredLimp);

    if (NPCState == ECognitiveNPCState::Idle && TimeInSubState > 8.f)
    {
        const ECognitiveCivilianSubState IdleStates[] = {
            ECognitiveCivilianSubState::IdlePhone,
            ECognitiveCivilianSubState::IdleTalking,
            ECognitiveCivilianSubState::IdleEating,
            ECognitiveCivilianSubState::IdleSmoking,
            ECognitiveCivilianSubState::IdleSitting
        };
        const int32 Pick = FMath::RandRange(0, 4);
        TransitionSubState((int32)IdleStates[Pick]);
    }

    if (NPCState == ECognitiveNPCState::Alert && CurrentSubState == (int32)ECognitiveCivilianSubState::IdlePhone)
        TransitionSubState((int32)ECognitiveCivilianSubState::RecordEvent);
}

void UCognitiveNPCProfileComponent::EvaluateDriverTransitions(float DeltaTime)
{
    if (!StateMachine.IsValid()) return;
    const ECognitiveNPCState NPCState = StateMachine->GetCurrentState();

    if (NPCState == ECognitiveNPCState::Flee)
        TransitionSubState((int32)ECognitiveDriverSubState::DrivingFlee);
    else if (NPCState == ECognitiveNPCState::Combat)
        TransitionSubState((int32)ECognitiveDriverSubState::DrivingAggressive);
    else if (NPCState == ECognitiveNPCState::React)
        TransitionSubState((int32)ECognitiveDriverSubState::CrashReaction);
    else if (NPCState == ECognitiveNPCState::Idle)
        TransitionSubState((int32)ECognitiveDriverSubState::DrivingNormal);
}

void UCognitiveNPCProfileComponent::EvaluateCriminalTransitions(float DeltaTime)
{
    if (!StateMachine.IsValid()) return;
    const ECognitiveNPCState NPCState = StateMachine->GetCurrentState();

    if (WantedLevel > 2 && NPCState != ECognitiveNPCState::Flee)
        StateMachine->TransitionTo(ECognitiveNPCState::Flee, TEXT("Wanted"));

    if (NPCState == ECognitiveNPCState::Flee)
        TransitionSubState((int32)ECognitiveCriminalSubState::EscapeOnFoot);
    else if (NPCState == ECognitiveNPCState::Stealth)
        TransitionSubState((int32)ECognitiveCriminalSubState::LoiteringStealth);
    else if (NPCState == ECognitiveNPCState::Combat)
        TransitionSubState((int32)ECognitiveCriminalSubState::FightPolice);
    else if (NPCState == ECognitiveNPCState::Hide)
        TransitionSubState((int32)ECognitiveCriminalSubState::BlendIn);
    else if (NPCState == ECognitiveNPCState::ContextualAction)
        TransitionSubState((int32)ECognitiveCriminalSubState::Pickpocketing);
}

void UCognitiveNPCProfileComponent::EvaluateFighterTransitions(float DeltaTime)
{
    if (!StateMachine.IsValid()) return;
    const ECognitiveNPCState NPCState = StateMachine->GetCurrentState();
    const FCognitiveBlackboard& BB = StateMachine->GetBlackboard();

    if (NPCState != ECognitiveNPCState::Combat) return;

    if (BB.Health < 30.f && TimeInSubState > 1.f)
    {
        TransitionSubState((int32)ECognitiveFighterSubState::Dodge);
        return;
    }

    if (TimeInSubState > 2.f)
    {
        const int32 ComboStates[] = {
            (int32)ECognitiveFighterSubState::Jab,
            (int32)ECognitiveFighterSubState::Cross,
            (int32)ECognitiveFighterSubState::Hook,
            (int32)ECognitiveFighterSubState::LowKick,
            (int32)ECognitiveFighterSubState::MidKick,
        };
        TransitionSubState(ComboStates[FMath::RandRange(0, 4)]);
    }

    if (BB.Stamina < 20.f)
        TransitionSubState((int32)ECognitiveFighterSubState::Circling);
}

void UCognitiveNPCProfileComponent::EvaluateSoldierTransitions(float DeltaTime)
{
    if (!StateMachine.IsValid()) return;
    const ECognitiveNPCState NPCState = StateMachine->GetCurrentState();
    const FCognitiveBlackboard& BB = StateMachine->GetBlackboard();

    if (NPCState == ECognitiveNPCState::Idle || NPCState == ECognitiveNPCState::CasualMovement)
        TransitionSubState((int32)ECognitiveSoldierSubState::Patrol);
    else if (NPCState == ECognitiveNPCState::Alert)
        TransitionSubState((int32)ECognitiveSoldierSubState::WallPeek);
    else if (NPCState == ECognitiveNPCState::TakeCover)
        TransitionSubState(BB.AggressionLevel > 60.f
            ? (int32)ECognitiveSoldierSubState::SuppressiveFire
            : (int32)ECognitiveSoldierSubState::TakeCover);
    else if (NPCState == ECognitiveNPCState::Combat)
    {
        if (BB.AmmoCount == 0)
            TransitionSubState((int32)ECognitiveSoldierSubState::Reload);
        else if (BB.AggressionLevel > 70.f)
            TransitionSubState((int32)ECognitiveSoldierSubState::AdvancingUnderFire);
        else
            TransitionSubState((int32)ECognitiveSoldierSubState::SuppressiveFire);
    }
    else if (NPCState == ECognitiveNPCState::Healing)
        TransitionSubState((int32)ECognitiveSoldierSubState::ReviveAlly);
}

void UCognitiveNPCProfileComponent::EvaluateSurvivorTransitions(float DeltaTime)
{
    if (!StateMachine.IsValid()) return;
    const ECognitiveNPCState NPCState = StateMachine->GetCurrentState();
    const FCognitiveBlackboard& BB = StateMachine->GetBlackboard();

    if (NPCState == ECognitiveNPCState::Stealth)
        TransitionSubState((int32)ECognitiveSurvivorSubState::StealthIdle);
    else if (NPCState == ECognitiveNPCState::Healing)
        TransitionSubState((int32)ECognitiveSurvivorSubState::Healing);
    else if (NPCState == ECognitiveNPCState::ContextualAction)
        TransitionSubState((int32)ECognitiveSurvivorSubState::Looting);
    else if (NPCState == ECognitiveNPCState::Flee)
        TransitionSubState((int32)ECognitiveSurvivorSubState::PanicSprint);
    else if (NPCState == ECognitiveNPCState::Combat)
        TransitionSubState(BB.Health < 40.f
            ? (int32)ECognitiveSurvivorSubState::ScaredCombat
            : (int32)ECognitiveSurvivorSubState::AmbushSetup);
}

void UCognitiveNPCProfileComponent::EvaluateInfectedTransitions(float DeltaTime)
{
    if (!StateMachine.IsValid()) return;
    const ECognitiveNPCState NPCState = StateMachine->GetCurrentState();
    const FCognitiveBlackboard& BB = StateMachine->GetBlackboard();

    if (NPCState == ECognitiveNPCState::Idle || NPCState == ECognitiveNPCState::CasualMovement)
        TransitionSubState((int32)ECognitiveInfectedSubState::Wander);
    else if (NPCState == ECognitiveNPCState::Alert || NPCState == ECognitiveNPCState::Investigate)
        TransitionSubState((int32)ECognitiveInfectedSubState::SoundInvestigate);
    else if (NPCState == ECognitiveNPCState::Combat)
    {
        if (BB.AggressionLevel > 80.f)
            TransitionSubState((int32)ECognitiveInfectedSubState::Enraged);
        else if (TimeInSubState > 1.5f)
            TransitionSubState((int32)ECognitiveInfectedSubState::AttackBite);
        else
            TransitionSubState((int32)ECognitiveInfectedSubState::Chase);
    }
    else if (NPCState == ECognitiveNPCState::Flee)
        TransitionSubState((int32)ECognitiveInfectedSubState::Stagger);
    else if (NPCState == ECognitiveNPCState::Dead)
        TransitionSubState((int32)ECognitiveInfectedSubState::PlayDead);
}
