#pragma once

#include "CoreMinimal.h"
#include "CognitiveNPCStateMachine.h"
#include "CognitiveNPCProfiles.generated.h"

UENUM(BlueprintType)
enum class ECognitiveCivilianSubState : uint8
{
    IdlePhone, IdleTalking, IdleEating, IdleSmoking, IdleSitting,
    WalkPurposeful, WalkRelaxed, Jog, Sprint,
    CrossStreet, AvoidObstacle,
    PanicFlee, HideBehindCar, Surrender, RecordEvent,
    HelpCivilian, CallPolice, FreezeInFear,
    DrunkWalk, InjuredLimp
};

UENUM(BlueprintType)
enum class ECognitiveDriverSubState : uint8
{
    DrivingNormal, DrivingAggressive, DrivingFlee,
    Stopping, Parking, EnteringVehicle, ExitingVehicle,
    CrashReaction, VehicleOnFire, CarjackVictim,
    HonkShort, HonkLong, LaneChange, EmergencyBrake
};

UENUM(BlueprintType)
enum class ECognitiveCriminalSubState : uint8
{
    LoiteringStealth, Pickpocketing, Mugging,
    CarjackingBreakWindow, CarjackingPullOut, CarjackingHotwire,
    EscapeOnFoot, EscapeVehicle, BlendIn,
    Threaten, FightPolice
};

UENUM(BlueprintType)
enum class ECognitiveFighterSubState : uint8
{
    FightingStanceHigh, FightingStanceMid, FightingStanceLow,
    Circling, Feint, Jab, Cross, Hook, Uppercut,
    LowKick, MidKick, HighKick, Roundhouse,
    Grapple, Takedown, GroundAndPound,
    CounterAttack, Dodge, Block,
    Taunt, Victory, KnockdownRecovery
};

UENUM(BlueprintType)
enum class ECognitiveSoldierSubState : uint8
{
    Patrol, TakeCover, SuppressiveFire, AdvancingUnderFire,
    Flanking, Breaching, GrenadeThrow, Reload,
    MeleeMilitary, ReviveAlly, CallSupport,
    ProneSuppress, WallPeek, BlindFire,
    BoundingOverwatch, LeapfrogAdvance
};

UENUM(BlueprintType)
enum class ECognitiveSurvivorSubState : uint8
{
    StealthIdle, CautiousMove, Looting,
    Barricading, Crafting, Healing,
    ScaredCombat, PanicSprint,
    AmbushSetup, Scavenge
};

UENUM(BlueprintType)
enum class ECognitiveInfectedSubState : uint8
{
    Wander, SoundInvestigate, Chase,
    AttackBite, AttackSwipe, Enraged,
    PlayDead, WallClimb, GrabAndBite,
    Stagger, Alert
};

UENUM(BlueprintType)
enum class ECognitiveNPCProfile : uint8
{
    None,
    CivilianUrban,
    Driver,
    Criminal,
    Fighter,
    Soldier,
    Survivor,
    Infected
};

UCLASS(ClassGroup=(Cognitive), meta=(BlueprintSpawnableComponent))
class COGNITIVEMOTIONINTELLIGENCE_API UCognitiveNPCProfileComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    UCognitiveNPCProfileComponent();

    virtual void BeginPlay() override;
    virtual void TickComponent(float DeltaTime, ELevelTick TickType,
        FActorComponentTickFunction* ThisTickFunction) override;

    UFUNCTION(BlueprintCallable, Category = "Cognitive|Profile")
    void SetProfile(ECognitiveNPCProfile InProfile);

    UFUNCTION(BlueprintPure, Category = "Cognitive|Profile")
    ECognitiveNPCProfile GetProfile() const { return ActiveProfile; }

    UFUNCTION(BlueprintCallable, Category = "Cognitive|Profile")
    bool TransitionSubState(int32 NewSubStateIndex);

    UFUNCTION(BlueprintPure, Category = "Cognitive|Profile")
    int32 GetCurrentSubState() const { return CurrentSubState; }

    UFUNCTION(BlueprintPure, Category = "Cognitive|Profile")
    float GetTimeInSubState() const { return TimeInSubState; }

    UFUNCTION(BlueprintCallable, Category = "Cognitive|Profile")
    void SetCarrying(bool bIsCarrying, float CarryWeightKg = 0.f);

    UFUNCTION(BlueprintCallable, Category = "Cognitive|Profile")
    void SetWantedLevel(int32 Level);

    UFUNCTION(BlueprintPure, Category = "Cognitive|Profile")
    int32 GetWantedLevel() const { return WantedLevel; }

    UFUNCTION(BlueprintCallable, Category = "Cognitive|Profile")
    ECognitiveMotionStyle GetMotionStyleForCurrentState() const;

    UFUNCTION(BlueprintCallable, Category = "Cognitive|Profile")
    ECognitiveMovementMode GetMovementModeForCurrentState() const;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cognitive|Profile")
    ECognitiveNPCProfile DefaultProfile = ECognitiveNPCProfile::CivilianUrban;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Cognitive|Profile|Debug")
    FString CurrentSubStateName;

    UPROPERTY(BlueprintAssignable, Category = "Cognitive|Profile")
    FCognitiveStateChangeDelegate OnSubStateChanged;

private:
    void EvaluateCivilianTransitions(float DeltaTime);
    void EvaluateDriverTransitions(float DeltaTime);
    void EvaluateCriminalTransitions(float DeltaTime);
    void EvaluateFighterTransitions(float DeltaTime);
    void EvaluateSoldierTransitions(float DeltaTime);
    void EvaluateSurvivorTransitions(float DeltaTime);
    void EvaluateInfectedTransitions(float DeltaTime);

    ECognitiveNPCProfile ActiveProfile  = ECognitiveNPCProfile::None;
    int32  CurrentSubState  = 0;
    float  TimeInSubState   = 0.f;
    int32  WantedLevel      = 0;
    bool   bCarrying        = false;
    float  CarryWeight      = 0.f;
    float  SubStateAccum    = 0.f;

    TWeakObjectPtr<UCognitiveNPCStateMachine> StateMachine;
};
