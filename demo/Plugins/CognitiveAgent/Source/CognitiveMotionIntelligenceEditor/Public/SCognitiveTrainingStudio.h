#pragma once

#include "CoreMinimal.h"
#include "Widgets/SCompoundWidget.h"
#include "CognitiveTrainingTypes.h"

class UCognitiveInferenceSubsystem;
class SEditableTextBox;
class SMultiLineEditableTextBox;
class STextBlock;
template<typename T> class SComboBox;
class SCheckBox;

/**
 * SCognitiveTrainingStudio
 *
 * Tela de Treino & Ensino do plugin (aba dockável no editor).
 *
 * TREINO: registra demonstrações rotuladas no catálogo do Python —
 *   tipo de treino (ex: "combate") + reação (ex: "agachar com arma") +
 *   animação (asset do Content Browser) + notas.
 *
 * ENSINO: monta um cenário ("3 inimigos, 1 virado para você"), pede ao agente
 *   que escolha uma reação do catálogo daquele tipo, e corrige: Correto /
 *   Errado + reações sugeridas. Cada correção vira dado de aprendizado.
 *
 * Requer PIE rodando com o servidor Python conectado (o subsystem de
 * inferência vive no GameInstance do PIE).
 */
class SCognitiveTrainingStudio : public SCompoundWidget
{
public:
    SLATE_BEGIN_ARGS(SCognitiveTrainingStudio) {}
    SLATE_END_ARGS()

    void Construct(const FArguments& InArgs);

private:
    // ── Infra ─────────────────────────────────────────────────────────────────
    UCognitiveInferenceSubsystem* GetSubsystem() const;   // PIE GameInstance
    EActiveTimerReturnType PollTeachingChoice(double, float);
    FText GetConnectionStatusText() const;

    // ── Treino ────────────────────────────────────────────────────────────────
    FReply OnPickAnimFromContentBrowser();
    FReply OnRegisterTraining();

    TSharedPtr<SEditableTextBox> TrainType;
    TSharedPtr<SComboBox<TSharedPtr<FString>>> TrainReactionCombo;
    TSharedPtr<FString>          TrainReactionSel;   // reação canônica escolhida
    TSharedPtr<SEditableTextBox> TrainAnimPath;
    TSharedPtr<SEditableTextBox> TrainNotes;
    TSharedPtr<STextBlock>       TrainLog;
    FString                      TrainLogAccum;

    // Vocabulário canônico de reações (bate com REACTION_NAMES do Python,
    // excluindo "none"). Fonte única para combo e checklists.
    static const TArray<TSharedPtr<FString>>& ReactionVocabulary();

    // ── Ensino ────────────────────────────────────────────────────────────────
    FReply OnAskAgent();
    FReply OnFeedback(bool bCorrect);
    FReply OnCaptureSceneScenario();   // Fase 2a: lê as percepções reais do PIE

    TSharedPtr<SEditableTextBox>          TeachType;
    TSharedPtr<SMultiLineEditableTextBox> TeachDescription;
    // Checklists sobre o vocabulário canônico (substituem texto livre)
    TMap<FString, TSharedPtr<SCheckBox>>  CandidateChecks;   // reação → checkbox
    TMap<FString, TSharedPtr<SCheckBox>>  SuggestionChecks;  // reação → checkbox
    TSharedPtr<STextBlock>                TeachChoiceText;

    // Constrói uma coluna de checkboxes do vocabulário; preenche OutMap.
    TSharedRef<class SWidget> BuildReactionChecklist(
        TMap<FString, TSharedPtr<SCheckBox>>& OutMap);
    // Lê as reações marcadas de um checklist (preserva ordem do vocabulário).
    TArray<FString> CheckedReactions(
        const TMap<FString, TSharedPtr<SCheckBox>>& Map) const;

    // Cenário: contagens por tipo (linhas fixas enemy/ally/object/danger)
    int32 EnemyCount = 0,  EnemyFacing = 0;
    int32 AllyCount = 0,   ObjectCount = 0, DangerCount = 0;
    float AvgDistanceM = 5.f;

    int64                     PendingScenarioId = 0;
    FCognitiveTeachingChoice  LastChoice;
    bool                      bHasChoice = false;
};
