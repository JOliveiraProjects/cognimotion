#pragma once

#include "CoreMinimal.h"
#include "CognitiveWorldPerceptionTypes.generated.h"

// ─────────────────────────────────────────────────────────────────────────────
// ECognitiveSocialRole
// Papel social específico de uma entidade numa cena, além da disposição.
// Usado em cenários complexos (ex.: sequestrador com refém) para o NPC decidir
// como reagir. None = sem papel especial.
// ─────────────────────────────────────────────────────────────────────────────
UENUM(BlueprintType)
enum class ECognitiveSocialRole : uint8
{
    None    = 0  UMETA(DisplayName = "None"),
    Hostage = 1  UMETA(DisplayName = "Hostage (refém)"),
    Captor  = 2  UMETA(DisplayName = "Captor (sequestrador)"),
    Civilian= 3  UMETA(DisplayName = "Civilian (civil)"),
    Wounded = 4  UMETA(DisplayName = "Wounded (ferido)"),
    Leader  = 5  UMETA(DisplayName = "Leader (líder/comandante)"),
};

// ─────────────────────────────────────────────────────────────────────────────
// ECognitiveEntityCategory
// O que uma entidade percebida É no mundo. Define como o NPC pode interagir.
// ─────────────────────────────────────────────────────────────────────────────
UENUM(BlueprintType)
enum class ECognitiveEntityCategory : uint8
{
    Unknown      = 0   UMETA(DisplayName = "Unknown"),
    Character    = 1   UMETA(DisplayName = "Character (pessoa/criatura)"),
    Weapon       = 2   UMETA(DisplayName = "Weapon (arma)"),
    Pickup       = 3   UMETA(DisplayName = "Pickup (objeto pegável)"),
    Vehicle      = 4   UMETA(DisplayName = "Vehicle (veículo)"),
    TrafficLight = 5   UMETA(DisplayName = "Traffic Light (semáforo)"),
    Cover        = 6   UMETA(DisplayName = "Cover (proteção)"),
    Hazard       = 7   UMETA(DisplayName = "Hazard (perigo)"),
    Objective    = 8   UMETA(DisplayName = "Objective (objetivo)"),
    Ignore       = 9   UMETA(DisplayName = "Ignore (ignorar)"),
};

// ─────────────────────────────────────────────────────────────────────────────
// ECognitiveDisposition
// Relação social do NPC com uma entidade (relevante p/ Characters).
// ─────────────────────────────────────────────────────────────────────────────
UENUM(BlueprintType)
enum class ECognitiveDisposition : uint8
{
    Neutral = 0  UMETA(DisplayName = "Neutral"),
    Friend  = 1  UMETA(DisplayName = "Friend (amigo)"),
    Enemy   = 2  UMETA(DisplayName = "Enemy (inimigo)"),
    Ally     = 3 UMETA(DisplayName = "Ally (aliado)"),
};

// ─────────────────────────────────────────────────────────────────────────────
// ECognitiveVehicleType
// ─────────────────────────────────────────────────────────────────────────────
UENUM(BlueprintType)
enum class ECognitiveVehicleType : uint8
{
    None     = 0  UMETA(DisplayName = "None"),
    Car      = 1  UMETA(DisplayName = "Car"),
    Motorcycle = 2 UMETA(DisplayName = "Motorcycle"),
    Bicycle  = 3  UMETA(DisplayName = "Bicycle"),
    Tank     = 4  UMETA(DisplayName = "Tank"),
    Boat     = 5  UMETA(DisplayName = "Boat"),
    Aircraft = 6  UMETA(DisplayName = "Aircraft"),
};

// ─────────────────────────────────────────────────────────────────────────────
// ECognitiveTrafficState
// ─────────────────────────────────────────────────────────────────────────────
UENUM(BlueprintType)
enum class ECognitiveTrafficState : uint8
{
    Unknown = 0  UMETA(DisplayName = "Unknown"),
    Red     = 1  UMETA(DisplayName = "Red (pare)"),
    Yellow  = 2  UMETA(DisplayName = "Yellow (atenção)"),
    Green   = 3  UMETA(DisplayName = "Green (siga)"),
};

// ─────────────────────────────────────────────────────────────────────────────
// ECognitiveReaction
// Decisão recomendada do NPC perante uma entidade. O sensor sugere; a política
// (Python) pode confirmar ou sobrescrever.
// ─────────────────────────────────────────────────────────────────────────────
UENUM(BlueprintType)
enum class ECognitiveReaction : uint8
{
    None     = 0  UMETA(DisplayName = "None"),
    Approach = 1  UMETA(DisplayName = "Approach (aproximar)"),
    Attack   = 2  UMETA(DisplayName = "Attack (atacar)"),
    Flee     = 3  UMETA(DisplayName = "Flee (fugir)"),
    Hide     = 4  UMETA(DisplayName = "Hide (esconder)"),
    PickUp   = 5  UMETA(DisplayName = "Pick Up (pegar)"),
    Enter    = 6  UMETA(DisplayName = "Enter (entrar - veículo)"),
    Wait     = 7  UMETA(DisplayName = "Wait (esperar)"),
    Cross    = 8  UMETA(DisplayName = "Cross (atravessar)"),
};

// ─────────────────────────────────────────────────────────────────────────────
// FCognitivePerceivedEntity
// Um item percebido pelo NPC no mundo, com sua classificação semântica.
// ─────────────────────────────────────────────────────────────────────────────
USTRUCT(BlueprintType)
struct COGNITIVEMOTIONINTELLIGENCE_API FCognitivePerceivedEntity
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly) TWeakObjectPtr<AActor> Actor = nullptr;
    UPROPERTY(BlueprintReadOnly) ECognitiveEntityCategory Category = ECognitiveEntityCategory::Unknown;
    UPROPERTY(BlueprintReadOnly) ECognitiveDisposition    Disposition = ECognitiveDisposition::Neutral;
    UPROPERTY(BlueprintReadOnly) ECognitiveSocialRole     Role = ECognitiveSocialRole::None;
    UPROPERTY(BlueprintReadOnly) ECognitiveReaction       SuggestedReaction = ECognitiveReaction::None;

    UPROPERTY(BlueprintReadOnly) float Distance = 0.f;
    UPROPERTY(BlueprintReadOnly) FVector RelativeDirection = FVector::ZeroVector; // local space do NPC
    UPROPERTY(BlueprintReadOnly) bool  bInLineOfSight = false;

    // Específicos por categoria
    UPROPERTY(BlueprintReadOnly) ECognitiveVehicleType  VehicleType  = ECognitiveVehicleType::None;
    UPROPERTY(BlueprintReadOnly) ECognitiveTrafficState TrafficState = ECognitiveTrafficState::Unknown;
    UPROPERTY(BlueprintReadOnly) float ThreatWeight = 0.f;  // 0..1 quão perigosa
};
