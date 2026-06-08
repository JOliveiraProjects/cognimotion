#include "CognitiveNPCStateMachine.h"
#include "GameFramework/Actor.h"   // BM-10 FIX: para GetOwner() e OnTakeAnyDamage

ECognitiveNPCState FCognitiveUtilityScores::GetHighestScoreState() const
{
    struct FPair { float Score; ECognitiveNPCState State; };
    const FPair Candidates[] = {
        { FleeScore,      ECognitiveNPCState::Flee      },
        { FightScore,     ECognitiveNPCState::Combat     },
        { HideScore,      ECognitiveNPCState::Hide       },
        { SurrenderScore, ECognitiveNPCState::Surrender  },
        { HealScore,      ECognitiveNPCState::Healing    },
    };
    const FPair* Best = &Candidates[0];
    for (const FPair& P : Candidates)
        if (P.Score > Best->Score) Best = &P;
    return Best->State;
}

UCognitiveNPCStateMachine::UCognitiveNPCStateMachine()
{
    PrimaryComponentTick.bCanEverTick = true;
    PrimaryComponentTick.TickInterval = 0.05f;
}

void UCognitiveNPCStateMachine::BeginPlay()
{
    Super::BeginPlay();
    Blackboard.CurrentState = ECognitiveNPCState::Idle;
    TimeInState = 0.f;

    // BM-10 FIX: vincula OnDamageReceived ao delegate UE de dano.
    if (AActor* Owner = GetOwner())
    {
        Owner->OnTakeAnyDamage.AddDynamic(this, &UCognitiveNPCStateMachine::OnDamageReceivedUE);
    }
}

void UCognitiveNPCStateMachine::EndPlay(const EEndPlayReason::Type Reason)
{
    // BM-10 FIX complementar: remove o binding ao destruir para evitar delegate dangling.
    // Sem isso, se o componente for destruído antes do Actor, o delegate chamaria
    // um objeto inválido e causaria crash.
    if (AActor* Owner = GetOwner())
    {
        Owner->OnTakeAnyDamage.RemoveDynamic(this, &UCognitiveNPCStateMachine::OnDamageReceivedUE);
    }
    Super::EndPlay(Reason);
}

float UCognitiveNPCStateMachine::GetTimeInCurrentState() const
{
    return TimeInState;
}

void UCognitiveNPCStateMachine::TickComponent(
    float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* Func)
{
    Super::TickComponent(DeltaTime, TickType, Func);
    TimeInState += DeltaTime;
    Blackboard.TimeSinceLastThreat += DeltaTime;

    UpdateDecayTimers(DeltaTime);
    EvaluateGlobalTriggers(DeltaTime);

    UtilityAccum += DeltaTime;
    if (UtilityAccum >= UtilityEvalInterval)
    {
        UtilityAccum = 0.f;
        const FCognitiveUtilityScores Scores = EvaluateUtilityScores();
        const ECognitiveNPCState TargetState  = Scores.GetHighestScoreState();

        if (TargetState != Blackboard.CurrentState)
            TransitionTo(TargetState, TEXT("Utility"));
    }
}

bool UCognitiveNPCStateMachine::IsValidTransition(
    ECognitiveNPCState From, ECognitiveNPCState To) const
{
    if (From == To) return false;
    if (To == ECognitiveNPCState::Dead) return true;
    if (To == ECognitiveNPCState::React) return true;

    switch (From)
    {
    case ECognitiveNPCState::Dead:
    case ECognitiveNPCState::Incapacitated:
        return false;

    case ECognitiveNPCState::Idle:
        return To == ECognitiveNPCState::CasualMovement
            || To == ECognitiveNPCState::Alert
            || To == ECognitiveNPCState::ContextualAction;

    case ECognitiveNPCState::Alert:
        return To == ECognitiveNPCState::Idle
            || To == ECognitiveNPCState::Investigate
            || To == ECognitiveNPCState::Combat
            || To == ECognitiveNPCState::Flee
            || To == ECognitiveNPCState::TakeCover;

    case ECognitiveNPCState::Combat:
        return To == ECognitiveNPCState::Flee
            || To == ECognitiveNPCState::TakeCover
            || To == ECognitiveNPCState::Healing
            || To == ECognitiveNPCState::Surrender
            || To == ECognitiveNPCState::Incapacitated;

    case ECognitiveNPCState::Flee:
        return To == ECognitiveNPCState::Hide
            || To == ECognitiveNPCState::Alert
            || To == ECognitiveNPCState::Surrender;

    default:
        return true;
    }
}

bool UCognitiveNPCStateMachine::TransitionTo(ECognitiveNPCState NewState, const FString& Reason)
{
    if (!IsValidTransition(Blackboard.CurrentState, NewState)) return false;
    ForceTransitionTo(NewState);
    return true;
}

void UCognitiveNPCStateMachine::ForceTransitionTo(ECognitiveNPCState NewState)
{
    const ECognitiveNPCState OldState = Blackboard.CurrentState;
    PreviousState            = OldState;
    Blackboard.CurrentState  = NewState;
    TimeInState              = 0.f;
    OnStateChanged.Broadcast(OldState, NewState);
}

void UCognitiveNPCStateMachine::ProcessSensorEvent(const FCognitiveSensorData& Event)
{
    OnSensorEvent.Broadcast(Event);

    switch (Event.EventType)
    {
    case ECognitiveSensorEvent::VisualContact:
        Blackboard.Alertness = FMath::Clamp(Blackboard.Alertness + Event.Intensity * 30.f, 0.f, 100.f);
        Blackboard.ThreatLevel = FMath::Max(Blackboard.ThreatLevel, Event.Intensity * 60.f);
        Blackboard.LastKnownTargetPosition = Event.EventLocation;
        Blackboard.TimeSinceLastThreat = 0.f;
        if (Blackboard.CurrentState == ECognitiveNPCState::Idle)
            TransitionTo(ECognitiveNPCState::Alert, TEXT("VisualContact"));
        else if (Blackboard.ThreatLevel > AlertCombatThreatThreshold)
            TransitionTo(ECognitiveNPCState::Combat, TEXT("HighThreat"));
        break;

    case ECognitiveSensorEvent::HearingEvent:
        Blackboard.SuspicionTimer = FMath::Clamp(Blackboard.SuspicionTimer + Event.Intensity * 10.f, 0.f, 100.f);
        if (Blackboard.SuspicionTimer > 50.f && Blackboard.CurrentState == ECognitiveNPCState::Idle)
            TransitionTo(ECognitiveNPCState::Investigate, TEXT("SuspiciousSound"));
        break;

    case ECognitiveSensorEvent::ExplosionNearby:
        Blackboard.FearLevel = FMath::Clamp(Blackboard.FearLevel + 50.f, 0.f, 100.f);
        Blackboard.Alertness = 100.f;
        TransitionTo(ECognitiveNPCState::Flee, TEXT("Explosion"));
        break;

    case ECognitiveSensorEvent::AllyDeathNearby:
        Blackboard.FearLevel += 25.f;
        Blackboard.AggressionLevel += 15.f;
        break;

    default: break;
    }
}

void UCognitiveNPCStateMachine::UpdateBlackboard(const FCognitiveBlackboard& NewValues)
{
    Blackboard.Health        = NewValues.Health;
    Blackboard.Stamina       = NewValues.Stamina;
    Blackboard.AmmoCount     = NewValues.AmmoCount;
    Blackboard.FearLevel     = NewValues.FearLevel;
    Blackboard.AggressionLevel = NewValues.AggressionLevel;
    Blackboard.Fatigue       = NewValues.Fatigue;
}

FCognitiveUtilityScores UCognitiveNPCStateMachine::EvaluateUtilityScores() const
{
    FCognitiveUtilityScores Scores;

    const float NormalizedHealth = Blackboard.Health / 100.f;
    const float NormalizedFear   = Blackboard.FearLevel / 100.f;
    const float NormalizedAggr   = Blackboard.AggressionLevel / 100.f;
    const float TeamSupport      = 0.f;
    const float CoverQuality     = Blackboard.bIsInCover ? 0.8f : 0.1f;

    Scores.FleeScore     = (NormalizedFear * 0.6f) + ((1.f - NormalizedHealth) * 0.4f) - (TeamSupport * 0.2f);
    Scores.FightScore    = (NormalizedAggr * 0.5f) + (NormalizedHealth * 0.3f) + (CoverQuality * 0.2f);
    Scores.HideScore     = (NormalizedFear * 0.4f) + ((1.f - CoverQuality) * 0.3f);
    Scores.SurrenderScore = NormalizedFear > 0.8f && NormalizedHealth < 0.2f ? 0.9f : 0.f;
    Scores.HealScore     = (1.f - NormalizedHealth) > 0.5f ? (1.f - NormalizedHealth) : 0.f;

    const ECognitiveNPCState S = Blackboard.CurrentState;
    if (S == ECognitiveNPCState::Combat || S == ECognitiveNPCState::Alert)
        Scores.FleeScore  *= (Blackboard.Health < CombatFleeHealthThreshold ? 2.f : 1.f);

    return Scores;
}

void UCognitiveNPCStateMachine::EvaluateGlobalTriggers(float DeltaTime)
{
    if (Blackboard.Health <= 0.f && Blackboard.CurrentState != ECognitiveNPCState::Dead)
    {
        ForceTransitionTo(ECognitiveNPCState::Dead);
        return;
    }

    if (Blackboard.CurrentState == ECognitiveNPCState::Flee)
    {
        TimeSinceLastSeen += DeltaTime;
        if (TimeSinceLastSeen >= FleeHideTimeThreshold)
            TransitionTo(ECognitiveNPCState::Hide, TEXT("LostSight"));
    }
    else
    {
        TimeSinceLastSeen = 0.f;
    }
}

void UCognitiveNPCStateMachine::UpdateDecayTimers(float DeltaTime)
{
    Blackboard.Alertness      = FMath::Max(0.f, Blackboard.Alertness      - AlertDecayRate     * DeltaTime);
    Blackboard.SuspicionTimer = FMath::Max(0.f, Blackboard.SuspicionTimer - SuspicionDecayRate * DeltaTime);
    Blackboard.FearLevel      = FMath::Max(0.f, Blackboard.FearLevel      - 2.f                * DeltaTime);
}

void UCognitiveNPCStateMachine::OnDamageReceivedUE(
    AActor* DamagedActor, float Damage, const UDamageType* DamageType,
    AController* InstigatedBy, AActor* DamageCauser)
{
    // BM-10 FIX: Wrapper UFUNCTION compatível com OnTakeAnyDamage signature.
    // Delega para a lógica interna com apenas o valor de dano relevante.
    OnDamageReceived(Damage);
}

void UCognitiveNPCStateMachine::OnDamageReceived(float Amount)
{
    Blackboard.Health = FMath::Max(0.f, Blackboard.Health - Amount);
    Blackboard.Alertness = 100.f;
    Blackboard.ThreatLevel = FMath::Max(Blackboard.ThreatLevel, Amount);
    if (Blackboard.CurrentState != ECognitiveNPCState::Combat)
        TransitionTo(ECognitiveNPCState::React, TEXT("Damaged"));
}
