#include "CognitiveHealthComponent.h"
#include "CognitiveMotionLearnerComponent.h"
#include "CognitiveDebugLog.h"
#include "GameFramework/Actor.h"

UCognitiveHealthComponent::UCognitiveHealthComponent()
{
    PrimaryComponentTick.bCanEverTick = false;
}

void UCognitiveHealthComponent::BeginPlay()
{
    Super::BeginPlay();
    CurrentHealth = (StartHealth > 0.f) ? FMath::Min(StartHealth, MaxHealth) : MaxHealth;
    bIsDead = (CurrentHealth <= 0.f);
    PropagateToLearner();
}

// ─────────────────────────────────────────────────────────────────────────────
float UCognitiveHealthComponent::GetHealthFraction() const
{
    return (MaxHealth > 0.f) ? FMath::Clamp(CurrentHealth / MaxHealth, 0.f, 1.f) : 0.f;
}

// ─────────────────────────────────────────────────────────────────────────────
void UCognitiveHealthComponent::ApplyDamage(float Amount)
{
    if (bIsDead || Amount <= 0.f) return;
    ApplyChange(CurrentHealth - Amount);
}

void UCognitiveHealthComponent::Heal(float Amount)
{
    if (bIsDead || Amount <= 0.f) return;
    ApplyChange(CurrentHealth + Amount);
}

void UCognitiveHealthComponent::SetHealth(float NewHealth)
{
    ApplyChange(NewHealth);
}

void UCognitiveHealthComponent::Kill()
{
    ApplyChange(0.f);
}

void UCognitiveHealthComponent::Revive(float WithHealth)
{
    bIsDead = false;
    const float HP = (WithHealth > 0.f) ? FMath::Min(WithHealth, MaxHealth) : MaxHealth;
    ApplyChange(HP);
}

// ─────────────────────────────────────────────────────────────────────────────
void UCognitiveHealthComponent::ApplyChange(float NewValue)
{
    const float Old = CurrentHealth;
    CurrentHealth = bClampHealth ? FMath::Clamp(NewValue, 0.f, MaxHealth) : NewValue;

    // Atualiza o flag de morte ANTES de propagar, para o estado físico refletir
    // corretamente já na jogada que mata o NPC.
    const bool bWasDead = bIsDead;
    if (!bIsDead && CurrentHealth <= 0.f)
    {
        bIsDead = true;
    }

    const float Delta = CurrentHealth - Old;
    if (!FMath::IsNearlyZero(Delta))
    {
        OnHealthChanged.Broadcast(CurrentHealth, Delta);
        PropagateToLearner();
    }

    if (bIsDead && !bWasDead)
    {
        CMI_DBG("[Health] %s morreu",
                GetOwner() ? *GetOwner()->GetName() : TEXT("?"));
        OnDeath.Broadcast();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Propaga a vida ao LearnerComponent (se houver) para alimentar o blackboard
// enviado à inferência — a decisão reativa usa a vida (fugir/morrer).
void UCognitiveHealthComponent::PropagateToLearner()
{
    if (AActor* Owner = GetOwner())
    {
        if (UCognitiveMotionLearnerComponent* Learner =
                Owner->FindComponentByClass<UCognitiveMotionLearnerComponent>())
        {
            Learner->SetExternalHealth(CurrentHealth);
            // Reflete morte/vida no estado físico mesmo sem inferência neural,
            // para a Anim BP tocar a animação de morte só com o núcleo.
            Learner->ForcePhysicalState(bIsDead
                ? ECognitivePhysicalState::Dead
                : ECognitivePhysicalState::Alive);
        }
    }
}
