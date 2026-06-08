#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "CognitiveWorldPerceptionTypes.h"
#include "CognitiveWorldPerceptionComponent.generated.h"

class UCognitiveEntityTagComponent;

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(
    FOnCognitiveEntityPerceived, const FCognitivePerceivedEntity&, Entity);

/**
 * UCognitiveWorldPerceptionComponent
 *
 * Dá ao NPC consciência semântica do mundo. A cada intervalo varre atores
 * próximos que tenham UCognitiveEntityTagComponent e os classifica:
 *   - amigo / inimigo / neutro
 *   - arma ou objeto que pode pegar (ou ignorar)
 *   - veículo que pode entrar (carro/moto/bike/tanque)
 *   - semáforo (vermelho/amarelo/verde) — esperar ou atravessar
 *   - perigo — fugir ou esconder
 *
 * Para cada entidade sugere uma reação (Attack/Flee/Hide/PickUp/Enter/Wait/Cross).
 * A reação é heurística e serve como dica para a política do Python, que pode
 * confirmar ou sobrescrever.
 *
 * Também fornece anexar/soltar objeto no skeleton (segurar arma, p.ex.).
 *
 * Coloque este componente no MESMO ator do NPC (junto do BoneDriver).
 */
UCLASS(ClassGroup=(Cognitive), meta=(BlueprintSpawnableComponent),
       DisplayName="Cognitive World Perception")
class COGNITIVEMOTIONINTELLIGENCE_API UCognitiveWorldPerceptionComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    UCognitiveWorldPerceptionComponent();

    // ── Configuração ──────────────────────────────────────────────────────────
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Perception",
              meta=(ClampMin="100.0"))
    float PerceptionRadius = 2000.f;

    // Campo de visão em graus (0..360). 360 = onisciente ao redor.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Perception",
              meta=(ClampMin="0.0", ClampMax="360.0"))
    float FieldOfViewDegrees = 200.f;

    // Exige linha de visão (raycast) para perceber a entidade
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Perception")
    bool bRequireLineOfSight = false;

    // Frequência de varredura (Hz)
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Perception",
              meta=(ClampMin="0.5", ClampMax="30.0"))
    float ScanRateHz = 4.f;

    // Facção do próprio NPC (mesma facção = amigo automático)
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Perception")
    FName SelfFaction = NAME_None;

    // ── Resultado (read-only) ─────────────────────────────────────────────────
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category="Cognitive|Perception|Debug")
    TArray<FCognitivePerceivedEntity> PerceivedEntities;

    UPROPERTY(BlueprintAssignable, Category="Cognitive|Perception")
    FOnCognitiveEntityPerceived OnEntityPerceived;

    // ── Blueprint API ─────────────────────────────────────────────────────────
    // Entidade hostil mais próxima percebida (Actor nulo se nenhuma)
    UFUNCTION(BlueprintPure, Category="Cognitive|Perception")
    FCognitivePerceivedEntity GetNearestThreat() const;

    // Pickup/arma mais próximo que o NPC pode pegar
    UFUNCTION(BlueprintPure, Category="Cognitive|Perception")
    FCognitivePerceivedEntity GetNearestPickup() const;

    // Estado do semáforo mais próximo (Unknown se nenhum)
    UFUNCTION(BlueprintPure, Category="Cognitive|Perception")
    ECognitiveTrafficState GetNearestTrafficState() const;

    // Anexa um objeto (arma/item) ao skeleton do NPC no socket indicado
    UFUNCTION(BlueprintCallable, Category="Cognitive|Perception|Interaction")
    bool AttachObjectToHand(AActor* Object, FName SocketName);

    // Solta o objeto atualmente segurado
    UFUNCTION(BlueprintCallable, Category="Cognitive|Perception|Interaction")
    void DropHeldObject();

    UFUNCTION(BlueprintPure, Category="Cognitive|Perception|Interaction")
    AActor* GetHeldObject() const { return HeldObject.Get(); }

    // ── UActorComponent ───────────────────────────────────────────────────────
    virtual void BeginPlay() override;
    virtual void TickComponent(float DeltaTime, ELevelTick TickType,
                                FActorComponentTickFunction* ThisTickFunction) override;

private:
    void ScanWorld();
    void SendPerceptionToPython();   // NPC → Python (MSG_PERCEPTION 0x08)
    ECognitiveDisposition ResolveDisposition(const UCognitiveEntityTagComponent* Tag) const;
    ECognitiveReaction    SuggestReaction(const FCognitivePerceivedEntity& E,
                                          const UCognitiveEntityTagComponent* Tag) const;
    bool HasLineOfSight(AActor* Target) const;

    float ScanAccumulator = 0.f;
    TWeakObjectPtr<AActor> HeldObject;
    TWeakObjectPtr<class UCognitiveInferenceSubsystem> InferenceSubsystem;
};
