#pragma once

#include "CoreMinimal.h"
#include "CognitiveBehaviorTypes.generated.h"

// ─────────────────────────────────────────────────────────────────────────────
// ECognitiveTrainingCategory
// A CATEGORIA de treino que o NPC está aprendendo. Define o "domínio" de
// comportamento. O Python usa isto para saber que tipo de movimento espera e
// como tomar decisões. O subtipo (texto livre) refina dentro da categoria.
//   Urbano   → pedestre, assaltante, policial, vendedor...
//   Luta     → MMA, karatê, boxe, muay thai...
//   Esporte  → futebol, basquete, futebol americano, tênis...
//   Zumbi    → andante, corredor, rastejante...
//   Corrida  → velocista, maratonista, parkour...
// ─────────────────────────────────────────────────────────────────────────────
UENUM(BlueprintType)
enum class ECognitiveTrainingCategory : uint8
{
    Urbano   = 0  UMETA(DisplayName = "Urbano"),
    Luta     = 1  UMETA(DisplayName = "Luta"),
    Esporte  = 2  UMETA(DisplayName = "Esporte"),
    Zumbi    = 3  UMETA(DisplayName = "Zumbi"),
    Corrida  = 4  UMETA(DisplayName = "Corrida"),
    Custom   = 255 UMETA(DisplayName = "Custom"),
};

// ─────────────────────────────────────────────────────────────────────────────
// ECognitiveLocomotionState
// Sinaliza ao TREINO o estado de locomoção do frame atual, para o Python
// rotular os dados corretamente. Idle quando parado, Dead quando a vida zera,
// etc. Tudo é aprendido no MESMO treino da categoria escolhida.
// ─────────────────────────────────────────────────────────────────────────────
UENUM(BlueprintType)
enum class ECognitiveLocomotionState : uint8
{
    Idle    = 0  UMETA(DisplayName = "Idle (parado)"),
    Walk    = 1  UMETA(DisplayName = "Walk"),
    Run     = 2  UMETA(DisplayName = "Run"),
    Action  = 3  UMETA(DisplayName = "Action (ação da categoria)"),
    Dead    = 4  UMETA(DisplayName = "Dead (sem vida)"),
};

// ─────────────────────────────────────────────────────────────────────────────
// ECognitiveObservationState
// Os modos de operação do NPC. Simplificado:
//   Observing → observa o líder e envia ao Python para TREINAR.
//   Inferring → o Python controla em tempo real (teste durante o treino).
//   Imported  → usa um modelo .pt já treinado e importado (produção, sem rede).
// ─────────────────────────────────────────────────────────────────────────────
UENUM(BlueprintType)
enum class ECognitiveObservationState : uint8
{
    Observing = 0  UMETA(DisplayName = "Observing Leader"),
    Inferring = 1  UMETA(DisplayName = "Inferring from Python"),
    Imported  = 2  UMETA(DisplayName = "Imported Model (.pt)"),
};

// ─────────────────────────────────────────────────────────────────────────────
// FCognitiveBehaviorContext
// O contexto de treino enviado ao Python: categoria + subtipo (texto livre) +
// estado de locomoção atual. O Python só precisa disto para saber o que está
// recebendo e gerar novas animações coerentes com aquele treino.
// ─────────────────────────────────────────────────────────────────────────────
USTRUCT(BlueprintType)
struct FCognitiveBehaviorContext
{
    GENERATED_BODY()

    // Categoria do treino (Urbano/Luta/Esporte/Zumbi/Corrida/Custom).
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cognitive|Behavior")
    ECognitiveTrainingCategory Category = ECognitiveTrainingCategory::Urbano;

    // Subtipo em texto livre: "MMA", "pedestre", "futebol", "policial", etc.
    // Você escreve; o Python recebe e associa ao treino daquela categoria.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cognitive|Behavior")
    FString Subtype;

    // Nome custom da categoria, usado só quando Category = Custom.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cognitive|Behavior",
              meta=(EditCondition="Category==ECognitiveTrainingCategory::Custom"))
    FString CustomCategoryName;

    // Estado de locomoção atual (idle/walk/run/action/dead) — sinaliza ao treino.
    // Atualizado em runtime (ex.: Dead quando a vida zera).
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cognitive|Behavior")
    ECognitiveLocomotionState LocomotionState = ECognitiveLocomotionState::Idle;

    // Chave única do treino: "Categoria|Subtipo". É o group_key que o Python usa
    // para separar os dados de cada treino.
    FString ToKey() const
    {
        const UEnum* CatEnum = StaticEnum<ECognitiveTrainingCategory>();
        FString C = (Category == ECognitiveTrainingCategory::Custom)
            ? CustomCategoryName
            : (CatEnum ? CatEnum->GetNameStringByValue((int64)Category) : TEXT("Urbano"));
        FString S = Subtype.IsEmpty() ? TEXT("default") : Subtype;
        return C + TEXT("|") + S;
    }
};
