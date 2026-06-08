#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "GameFramework/DamageType.h"   // BM-10 FIX: necessário para OnTakeAnyDamage signature
#include "CognitiveMotionTypes.h"
#include "CognitiveNPCStateMachine.generated.h"

DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FOnNPCStateChanged,
    ECognitiveNPCState, OldState, ECognitiveNPCState, NewState);

USTRUCT(BlueprintType)
struct COGNITIVEMOTIONINTELLIGENCE_API FCognitiveUtilityScores
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadWrite) float FleeScore     = 0.f;
    UPROPERTY(BlueprintReadWrite) float FightScore    = 0.f;
    UPROPERTY(BlueprintReadWrite) float HideScore     = 0.f;
    UPROPERTY(BlueprintReadWrite) float SurrenderScore = 0.f;
    UPROPERTY(BlueprintReadWrite) float HealScore     = 0.f;

    ECognitiveNPCState GetHighestScoreState() const;
};

UCLASS(ClassGroup=(Cognitive), meta=(BlueprintSpawnableComponent))
class COGNITIVEMOTIONINTELLIGENCE_API UCognitiveNPCStateMachine : public UActorComponent
{
    GENERATED_BODY()

public:
    UCognitiveNPCStateMachine();

    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type Reason) override;  // BM-10: remove OnTakeAnyDamage binding
    virtual void TickComponent(float DeltaTime, ELevelTick TickType,
        FActorComponentTickFunction* ThisTickFunction) override;

    UFUNCTION(BlueprintCallable, Category = "Cognitive|FSM")
    bool TransitionTo(ECognitiveNPCState NewState, const FString& Reason = TEXT(""));

    UFUNCTION(BlueprintCallable, Category = "Cognitive|FSM")
    void ForceTransitionTo(ECognitiveNPCState NewState);

    UFUNCTION(BlueprintPure, Category = "Cognitive|FSM")
    ECognitiveNPCState GetCurrentState() const { return Blackboard.CurrentState; }

    UFUNCTION(BlueprintPure, Category = "Cognitive|FSM")
    float GetTimeInCurrentState() const;

    UFUNCTION(BlueprintPure, Category = "Cognitive|FSM")
    ECognitiveNPCState GetPreviousState() const { return PreviousState; }

    UFUNCTION(BlueprintCallable, Category = "Cognitive|FSM")
    void ProcessSensorEvent(const FCognitiveSensorData& Event);

    UFUNCTION(BlueprintCallable, Category = "Cognitive|FSM")
    void UpdateBlackboard(const FCognitiveBlackboard& NewValues);

    UFUNCTION(BlueprintPure, Category = "Cognitive|FSM")
    const FCognitiveBlackboard& GetBlackboard() const { return Blackboard; }

    UFUNCTION(BlueprintCallable, Category = "Cognitive|FSM")
    FCognitiveUtilityScores EvaluateUtilityScores() const;

    UPROPERTY(BlueprintAssignable, Category = "Cognitive|FSM")
    FOnNPCStateChanged OnStateChanged;

    UPROPERTY(BlueprintAssignable, Category = "Cognitive|FSM")
    FCognitiveSensorEventDelegate OnSensorEvent;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cognitive|FSM")
    float UtilityEvalInterval = 0.2f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cognitive|FSM")
    float AlertDecayRate = 10.f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cognitive|FSM")
    float SuspicionDecayRate = 5.f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cognitive|FSM")
    float CombatFleeHealthThreshold = 25.f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cognitive|FSM")
    float CombatFleeAmmoThreshold = 0.f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cognitive|FSM")
    float AlertCombatThreatThreshold = 70.f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cognitive|FSM")
    float FleeHideTimeThreshold = 4.0f;

private:
    bool IsValidTransition(ECognitiveNPCState From, ECognitiveNPCState To) const;
    void EvaluateGlobalTriggers(float DeltaTime);
    void UpdateDecayTimers(float DeltaTime);

    // BM-10 FIX: OnDamageReceived era private e não vinculado a nenhum delegate.
    // OnDamageReceivedUE é o UFUNCTION wrapper vinculado a Owner->OnTakeAnyDamage.
    UFUNCTION()
    void OnDamageReceivedUE(AActor* DamagedActor, float Damage,
        const UDamageType* DamageType, AController* InstigatedBy, AActor* DamageCauser);

    void OnDamageReceived(float Amount);  // lógica interna

    FCognitiveBlackboard   Blackboard;
    ECognitiveNPCState     PreviousState    = ECognitiveNPCState::Idle;
    float                  TimeInState      = 0.f;
    float                  UtilityAccum     = 0.f;
    float                  TimeSinceLastSeen = 0.f;
};
