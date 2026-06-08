#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "CognitiveWorldPerceptionTypes.h"
#include "CognitiveEntityTagComponent.generated.h"

/**
 * UCognitiveEntityTagComponent
 *
 * Coloque este componente em QUALQUER ator do mundo para declarar o que ele é
 * para os NPCs Cognitive: arma, veículo, semáforo, inimigo, objeto pegável, etc.
 *
 * Configuração 100% no editor — sem código. O NPC lê estas tags via o
 * UCognitiveWorldPerceptionComponent e decide como agir.
 *
 * Exemplos:
 *   - Numa arma:     Category=Weapon,  bCanPickUp=true
 *   - Num carro:     Category=Vehicle, VehicleType=Car, bCanEnter=true
 *   - Num inimigo:   Category=Character, Disposition=Enemy
 *   - Num semáforo:  Category=TrafficLight (atualize TrafficState em runtime)
 */
UCLASS(ClassGroup=(Cognitive), meta=(BlueprintSpawnableComponent),
       DisplayName="Cognitive Entity Tag")
class COGNITIVEMOTIONINTELLIGENCE_API UCognitiveEntityTagComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    UCognitiveEntityTagComponent();

    // ── Classificação principal ───────────────────────────────────────────────
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Entity")
    ECognitiveEntityCategory Category = ECognitiveEntityCategory::Unknown;

    // Para Characters: amigo/inimigo/aliado/neutro
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Entity",
              meta=(EditCondition="Category==ECognitiveEntityCategory::Character"))
    ECognitiveDisposition Disposition = ECognitiveDisposition::Neutral;

    // Facção opcional (NPCs da mesma facção são amigos por padrão)
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Entity")
    FName Faction = NAME_None;

    // Papel social específico (refém, sequestrador, civil, ferido, líder).
    // Permite cenários complexos como sequestrador+refém. None = sem papel.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Entity")
    ECognitiveSocialRole Role = ECognitiveSocialRole::None;

    // ── Pickups / armas ───────────────────────────────────────────────────────
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Entity|Pickup")
    bool bCanPickUp = false;

    // Socket no skeleton do NPC onde o objeto é anexado ao pegar (ex: "hand_r")
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Entity|Pickup",
              meta=(EditCondition="bCanPickUp"))
    FName AttachSocket = TEXT("hand_r");

    // ── Veículos ──────────────────────────────────────────────────────────────
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Entity|Vehicle",
              meta=(EditCondition="Category==ECognitiveEntityCategory::Vehicle"))
    ECognitiveVehicleType VehicleType = ECognitiveVehicleType::None;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Entity|Vehicle",
              meta=(EditCondition="Category==ECognitiveEntityCategory::Vehicle"))
    bool bCanEnter = true;

    // Socket no veículo onde o NPC senta (ex: "DriverSeat")
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Entity|Vehicle",
              meta=(EditCondition="Category==ECognitiveEntityCategory::Vehicle"))
    FName SeatSocket = TEXT("DriverSeat");

    // ── Semáforo ──────────────────────────────────────────────────────────────
    // Atualize em runtime conforme o semáforo muda (Blueprint ou C++).
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Entity|Traffic",
              meta=(EditCondition="Category==ECognitiveEntityCategory::TrafficLight"))
    ECognitiveTrafficState TrafficState = ECognitiveTrafficState::Unknown;

    // ── Ameaça ────────────────────────────────────────────────────────────────
    // Peso 0..1 de quão perigosa esta entidade é (inimigo forte = perto de 1).
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Entity|Threat",
              meta=(ClampMin="0.0", ClampMax="1.0"))
    float ThreatWeight = 0.f;

    // ── Blueprint API ─────────────────────────────────────────────────────────
    UFUNCTION(BlueprintCallable, Category="Cognitive|Entity")
    void SetTrafficState(ECognitiveTrafficState NewState) { TrafficState = NewState; }

    UFUNCTION(BlueprintCallable, Category="Cognitive|Entity")
    void SetDisposition(ECognitiveDisposition NewDisposition) { Disposition = NewDisposition; }
};
