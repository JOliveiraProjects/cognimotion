#pragma once

#include "CoreMinimal.h"
#include "CognitiveTrainingTypes.generated.h"

/**
 * Tipos do sistema de Treino & Ensino.
 *
 * TREINO  = registrar uma demonstração rotulada: "este tipo de treino tem esta
 *           reação, executada com esta animação". O Python guarda o catálogo.
 * ENSINO  = apresentar um cenário ("3 inimigos, um virado para você"), deixar o
 *           agente escolher uma reação do catálogo daquele tipo, e corrigir:
 *           certo/errado + reações sugeridas. Cada correção vira dado de treino.
 */

// ── Uma entrada de treino registrada ─────────────────────────────────────────
USTRUCT(BlueprintType)
struct COGNITIVEMOTIONINTELLIGENCE_API FCognitiveTrainingEntry
{
    GENERATED_BODY()

    // Tipo/categoria do treino (ex: "combate", "furtividade", "social")
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Training")
    FString TrainingType;

    // Nome da reação/treino (ex: "agachar com arma", "desviar do inimigo")
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Training")
    FString ReactionName;

    // Caminho do asset de animação (ex: /Game/Anims/Crouch_Rifle.Crouch_Rifle)
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Training")
    FString AnimationPath;

    // Notas livres opcionais (contexto, quando usar, etc.)
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Training")
    FString Notes;
};

// ── Uma entidade do cenário de ensino ────────────────────────────────────────
USTRUCT(BlueprintType)
struct COGNITIVEMOTIONINTELLIGENCE_API FCognitiveScenarioEntity
{
    GENERATED_BODY()

    // Categoria semântica em texto (ex: "enemy", "ally", "object", "danger")
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Teaching")
    FString Kind = TEXT("enemy");

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Teaching")
    int32 Count = 1;

    // Quantos deles estão virados para o agente
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Teaching")
    int32 FacingMe = 0;

    // Distância aproximada em metros (média)
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Teaching")
    float DistanceM = 5.f;
};

// ── Cenário de ensino ────────────────────────────────────────────────────────
USTRUCT(BlueprintType)
struct COGNITIVEMOTIONINTELLIGENCE_API FCognitiveTeachingScenario
{
    GENERATED_BODY()

    // Id único do cenário (preenchido pelo sistema)
    UPROPERTY(BlueprintReadOnly, Category="Cognitive|Teaching")
    int64 ScenarioId = 0;

    // Tipo de treino cujo catálogo de reações será usado (ex: "combate")
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Teaching")
    FString TrainingType;

    // Descrição textual livre da situação
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Teaching")
    FString Description;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Teaching")
    TArray<FCognitiveScenarioEntity> Entities;

    // Reações candidatas apresentadas ao agente (nomes do catálogo daquele tipo).
    // Vazio = Python usa todas as reações registradas para o TrainingType.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Teaching")
    TArray<FString> CandidateReactions;
};

// ── Escolha do agente (resposta do Python) ───────────────────────────────────
USTRUCT(BlueprintType)
struct COGNITIVEMOTIONINTELLIGENCE_API FCognitiveTeachingChoice
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category="Cognitive|Teaching")
    int64 ScenarioId = 0;

    UPROPERTY(BlueprintReadOnly, Category="Cognitive|Teaching")
    FString ChosenReaction;

    // Confiança 0..1 reportada pelo Python
    UPROPERTY(BlueprintReadOnly, Category="Cognitive|Teaching")
    float Confidence = 0.f;

    // Justificativa opcional que o Python pode devolver (debug/transparência)
    UPROPERTY(BlueprintReadOnly, Category="Cognitive|Teaching")
    FString Rationale;
};

// ── Feedback do humano sobre a escolha ───────────────────────────────────────
USTRUCT(BlueprintType)
struct COGNITIVEMOTIONINTELLIGENCE_API FCognitiveTeachingFeedback
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Teaching")
    int64 ScenarioId = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Teaching")
    bool bCorrect = false;

    // A reação que o agente tinha escolhido (eco, para o Python casar o registro)
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Teaching")
    FString ChosenReaction;

    // Se errado: quais reações o professor sugere como corretas (ordenadas)
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Teaching")
    TArray<FString> SuggestedReactions;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Teaching")
    FString Comment;
};
