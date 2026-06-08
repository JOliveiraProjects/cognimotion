#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "CognitiveHealthComponent.generated.h"

DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(
    FOnCognitiveHealthChanged, float, NewHealth, float, Delta);

DECLARE_DYNAMIC_MULTICAST_DELEGATE(FOnCognitiveDeath);

/**
 * UCognitiveHealthComponent
 *
 * Vida configurável e pronta para uso de NPC/personagem. É a peça que liga
 * o gameplay (jogador bate, atira, atropela) ao sistema de estados físicos:
 * quando a vida chega a zero, o NPC entra em estado de MORTE e dispara eventos
 * que sua Anim Blueprint usa para tocar a animação de morte.
 *
 * Uso (Blueprint, em segundos):
 *   1. Add Component → Cognitive Health.
 *   2. Configure MaxHealth no Details.
 *   3. Chame ApplyDamage / Heal do seu gameplay.
 *   4. Ligue OnDeath / OnHealthChanged na sua lógica e Anim BP.
 *
 * Integra-se automaticamente com o Native Inference / Learner: o valor de vida
 * é exposto e pode alimentar a decisão reativa (fugir com vida baixa, morrer).
 */
UCLASS(ClassGroup=(Cognitive), meta=(BlueprintSpawnableComponent),
       DisplayName="Cognitive Health")
class COGNITIVEMOTIONINTELLIGENCE_API UCognitiveHealthComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    UCognitiveHealthComponent();

    // ── Configuração ──────────────────────────────────────────────────────────
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Health",
              meta=(ClampMin="1.0"))
    float MaxHealth = 100.f;

    // Vida inicial; se <= 0 usa MaxHealth.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Health")
    float StartHealth = -1.f;

    // Se true, a vida não cai abaixo de 0 nem sobe acima de MaxHealth.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Health")
    bool bClampHealth = true;

    // ── Estado (read-only) ────────────────────────────────────────────────────
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category="Cognitive|Health")
    float CurrentHealth = 100.f;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category="Cognitive|Health")
    bool bIsDead = false;

    // ── Eventos ───────────────────────────────────────────────────────────────
    // Dispara em qualquer mudança de vida (Delta negativo = dano, positivo = cura).
    UPROPERTY(BlueprintAssignable, Category="Cognitive|Health")
    FOnCognitiveHealthChanged OnHealthChanged;

    // Dispara uma vez quando a vida chega a zero. Ligue sua animação de morte aqui.
    UPROPERTY(BlueprintAssignable, Category="Cognitive|Health")
    FOnCognitiveDeath OnDeath;

    // ── API ───────────────────────────────────────────────────────────────────
    // Aplica dano (valor positivo reduz a vida). Use ao bater/atirar/atropelar.
    UFUNCTION(BlueprintCallable, Category="Cognitive|Health")
    void ApplyDamage(float Amount);

    // Cura (valor positivo aumenta a vida, até MaxHealth).
    UFUNCTION(BlueprintCallable, Category="Cognitive|Health")
    void Heal(float Amount);

    // Define a vida diretamente.
    UFUNCTION(BlueprintCallable, Category="Cognitive|Health")
    void SetHealth(float NewHealth);

    // Mata o NPC imediatamente.
    UFUNCTION(BlueprintCallable, Category="Cognitive|Health")
    void Kill();

    // Revive com vida cheia (ou valor informado).
    UFUNCTION(BlueprintCallable, Category="Cognitive|Health")
    void Revive(float WithHealth = -1.f);

    // Fração de vida 0..1 (para barras de vida).
    UFUNCTION(BlueprintPure, Category="Cognitive|Health")
    float GetHealthFraction() const;

    UFUNCTION(BlueprintPure, Category="Cognitive|Health")
    bool IsDead() const { return bIsDead; }

    virtual void BeginPlay() override;

private:
    void ApplyChange(float NewValue);
    void PropagateToLearner();
};
