#include "SCognitiveTrainingStudio.h"
#include "CognitiveInferenceSubsystem.h"

#include "Editor.h"
#include "Engine/GameInstance.h"
#include "ContentBrowserModule.h"
#include "IContentBrowserSingleton.h"
#include "Animation/AnimSequence.h"
#include "EngineUtils.h"
#include "CognitiveWorldPerceptionComponent.h"
#include "CognitiveWorldPerceptionTypes.h"

#include "Widgets/Input/SEditableTextBox.h"
#include "Widgets/Input/SMultiLineEditableTextBox.h"
#include "Widgets/Input/SComboBox.h"
#include "Widgets/Input/SCheckBox.h"
#include "Widgets/Input/SButton.h"
#include "Widgets/Input/SNumericEntryBox.h"
#include "Widgets/Text/STextBlock.h"
#include "Widgets/Layout/SScrollBox.h"
#include "Widgets/Layout/SSeparator.h"
#include "Widgets/Layout/SExpandableArea.h"

#define LOCTEXT_NAMESPACE "CognitiveTrainingStudio"

// ─────────────────────────────────────────────────────────────────────────────
namespace
{
    // Linha rotulada compacta: [label 140px][widget]
    TSharedRef<SWidget> Row(const FText& Label, TSharedRef<SWidget> Widget)
    {
        return SNew(SHorizontalBox)
            + SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center)
              .Padding(0, 2, 8, 2)
              [ SNew(SBox).WidthOverride(150)
                [ SNew(STextBlock).Text(Label) ] ]
            + SHorizontalBox::Slot().FillWidth(1.f).Padding(0, 2)
              [ Widget ];
    }

}

// Vocabulário canônico de MOVIMENTOS — DEVE bater com VERB_NAMES no
// protocol/binary_protocol.py (idle/walk/run/jump/crouch/crawl/vault/pickup/
// flee/hide/attack/defend). São as ações que o world model executa; o ensino
// escolhe qual usar por situação. Movimentos fora desta lista caem no palpite
// em vez da política neural, por isso a UI só oferece estes.
const TArray<TSharedPtr<FString>>& SCognitiveTrainingStudio::ReactionVocabulary()
{
    static TArray<TSharedPtr<FString>> Vocab;
    if (Vocab.Num() == 0)
    {
        const TCHAR* Names[] = {
            TEXT("idle"),  TEXT("walk"),   TEXT("run"),    TEXT("jump"),
            TEXT("crouch"),TEXT("crawl"),  TEXT("vault"),  TEXT("pickup"),
            TEXT("flee"),  TEXT("hide"),   TEXT("attack"), TEXT("defend") };
        for (const TCHAR* N : Names)
            Vocab.Add(MakeShared<FString>(N));
    }
    return Vocab;
}

TSharedRef<SWidget> SCognitiveTrainingStudio::BuildReactionChecklist(
    TMap<FString, TSharedPtr<SCheckBox>>& OutMap)
{
    TSharedRef<SVerticalBox> Box = SNew(SVerticalBox);
    for (const TSharedPtr<FString>& R : ReactionVocabulary())
    {
        TSharedPtr<SCheckBox> Check;
        Box->AddSlot().AutoHeight().Padding(0, 1)
        [
            SAssignNew(Check, SCheckBox)
            .Content()[ SNew(STextBlock).Text(FText::FromString(*R)) ]
        ];
        OutMap.Add(*R, Check);
    }
    return Box;
}

TArray<FString> SCognitiveTrainingStudio::CheckedReactions(
    const TMap<FString, TSharedPtr<SCheckBox>>& Map) const
{
    TArray<FString> Out;
    // Preserva a ordem canônica do vocabulário
    for (const TSharedPtr<FString>& R : ReactionVocabulary())
    {
        const TSharedPtr<SCheckBox>* Check = Map.Find(*R);
        if (Check && Check->IsValid() && (*Check)->IsChecked())
            Out.Add(*R);
    }
    return Out;
}

// ─────────────────────────────────────────────────────────────────────────────
void SCognitiveTrainingStudio::Construct(const FArguments& InArgs)
{
    // Poll periódico da resposta do ensino (escolha do agente)
    RegisterActiveTimer(0.25f, FWidgetActiveTimerDelegate::CreateSP(
        this, &SCognitiveTrainingStudio::PollTeachingChoice));

    ChildSlot
    [
        SNew(SScrollBox)
        + SScrollBox::Slot().Padding(12)
        [
            SNew(SVerticalBox)

            // ── Status de conexão ─────────────────────────────────────────────
            + SVerticalBox::Slot().AutoHeight().Padding(0, 0, 0, 8)
            [
                SNew(STextBlock)
                .Text(this, &SCognitiveTrainingStudio::GetConnectionStatusText)
            ]

            // ══ SEÇÃO TREINO ═════════════════════════════════════════════════
            + SVerticalBox::Slot().AutoHeight()
            [
                SNew(SExpandableArea)
                .InitiallyCollapsed(false)
                .HeaderContent()
                [ SNew(STextBlock)
                  .Text(LOCTEXT("TrainHeader", "TREINO — registrar demonstração"))
                  .Font(FCoreStyle::GetDefaultFontStyle("Bold", 12)) ]
                .BodyContent()
                [
                    SNew(SVerticalBox)
                    + SVerticalBox::Slot().AutoHeight()
                    [ Row(LOCTEXT("TrainType", "Tipo de treino"),
                          SAssignNew(TrainType, SEditableTextBox)
                          .HintText(LOCTEXT("TrainTypeHint", "ex: combate"))) ]
                    + SVerticalBox::Slot().AutoHeight()
                    [ Row(LOCTEXT("TrainReaction", "Treino / reação"),
                          SAssignNew(TrainReactionCombo, SComboBox<TSharedPtr<FString>>)
                          .OptionsSource(&ReactionVocabulary())
                          .OnGenerateWidget_Lambda([](TSharedPtr<FString> In)
                              { return SNew(STextBlock).Text(FText::FromString(*In)); })
                          .OnSelectionChanged_Lambda(
                              [this](TSharedPtr<FString> In, ESelectInfo::Type)
                              { TrainReactionSel = In; })
                          .InitiallySelectedItem(ReactionVocabulary()[0])
                          [ SNew(STextBlock).Text_Lambda([this]
                              { return FText::FromString(
                                  TrainReactionSel.IsValid() ? *TrainReactionSel
                                  : *ReactionVocabulary()[0]); }) ]) ]
                    + SVerticalBox::Slot().AutoHeight()
                    [ Row(LOCTEXT("TrainAnim", "Animação"),
                          SNew(SHorizontalBox)
                          + SHorizontalBox::Slot().FillWidth(1.f)
                            [ SAssignNew(TrainAnimPath, SEditableTextBox)
                              .HintText(LOCTEXT("TrainAnimHint",
                                  "/Game/Anims/Crouch_Rifle.Crouch_Rifle")) ]
                          + SHorizontalBox::Slot().AutoWidth().Padding(4, 0, 0, 0)
                            [ SNew(SButton)
                              .Text(LOCTEXT("PickAnim", "Usar seleção"))
                              .ToolTipText(LOCTEXT("PickAnimTip",
                                  "Usa a AnimSequence selecionada no Content Browser"))
                              .OnClicked(this, &SCognitiveTrainingStudio::OnPickAnimFromContentBrowser) ]) ]
                    + SVerticalBox::Slot().AutoHeight()
                    [ Row(LOCTEXT("TrainNotes", "Notas (opcional)"),
                          SAssignNew(TrainNotes, SEditableTextBox)) ]
                    + SVerticalBox::Slot().AutoHeight().Padding(0, 8)
                    [ SNew(SButton)
                      .HAlign(HAlign_Center)
                      .Text(LOCTEXT("RegisterTraining", "Registrar treino no Python"))
                      .OnClicked(this, &SCognitiveTrainingStudio::OnRegisterTraining) ]
                    + SVerticalBox::Slot().AutoHeight()
                    [ SAssignNew(TrainLog, STextBlock)
                      .AutoWrapText(true)
                      .Text(LOCTEXT("TrainLogEmpty", "Nenhum treino registrado nesta sessão.")) ]
                ]
            ]

            + SVerticalBox::Slot().AutoHeight().Padding(0, 12)
            [ SNew(SSeparator) ]

            // ══ SEÇÃO ENSINO ═════════════════════════════════════════════════
            + SVerticalBox::Slot().AutoHeight()
            [
                SNew(SExpandableArea)
                .InitiallyCollapsed(false)
                .HeaderContent()
                [ SNew(STextBlock)
                  .Text(LOCTEXT("TeachHeader", "ENSINO — cenário, decisão e correção"))
                  .Font(FCoreStyle::GetDefaultFontStyle("Bold", 12)) ]
                .BodyContent()
                [
                    SNew(SVerticalBox)
                    + SVerticalBox::Slot().AutoHeight()
                    [ Row(LOCTEXT("TeachType", "Tipo de treino"),
                          SAssignNew(TeachType, SEditableTextBox)
                          .HintText(LOCTEXT("TeachTypeHint",
                              "ex: combate (usa o catálogo desse tipo)"))) ]
                    + SVerticalBox::Slot().AutoHeight()
                    [ Row(LOCTEXT("TeachDesc", "Descrição da situação"),
                          SAssignNew(TeachDescription, SMultiLineEditableTextBox)
                          .HintText(LOCTEXT("TeachDescHint",
                              "ex: 3 inimigos à frente; um deles está virado para você"))) ]

                    // Cenário: contagens
                    + SVerticalBox::Slot().AutoHeight()
                    [ Row(LOCTEXT("Enemies", "Inimigos / virados p/ mim"),
                          SNew(SHorizontalBox)
                          + SHorizontalBox::Slot().MaxWidth(90).Padding(0,0,6,0)
                            [ SNew(SNumericEntryBox<int32>)
                              .MinValue(0)
                              .Value_Lambda([this]{ return EnemyCount; })
                              .OnValueChanged_Lambda([this](int32 V){ EnemyCount = V; }) ]
                          + SHorizontalBox::Slot().MaxWidth(90)
                            [ SNew(SNumericEntryBox<int32>)
                              .MinValue(0)
                              .Value_Lambda([this]{ return EnemyFacing; })
                              .OnValueChanged_Lambda([this](int32 V){ EnemyFacing = V; }) ]) ]
                    + SVerticalBox::Slot().AutoHeight()
                    [ Row(LOCTEXT("Allies", "Aliados"),
                          SNew(SBox).MaxDesiredWidth(90)
                          [ SNew(SNumericEntryBox<int32>)
                            .MinValue(0)
                            .Value_Lambda([this]{ return AllyCount; })
                            .OnValueChanged_Lambda([this](int32 V){ AllyCount = V; }) ]) ]
                    + SVerticalBox::Slot().AutoHeight()
                    [ Row(LOCTEXT("Objects", "Objetos de interesse"),
                          SNew(SBox).MaxDesiredWidth(90)
                          [ SNew(SNumericEntryBox<int32>)
                            .MinValue(0)
                            .Value_Lambda([this]{ return ObjectCount; })
                            .OnValueChanged_Lambda([this](int32 V){ ObjectCount = V; }) ]) ]
                    + SVerticalBox::Slot().AutoHeight()
                    [ Row(LOCTEXT("Dangers", "Perigos (fogo, queda...)"),
                          SNew(SBox).MaxDesiredWidth(90)
                          [ SNew(SNumericEntryBox<int32>)
                            .MinValue(0)
                            .Value_Lambda([this]{ return DangerCount; })
                            .OnValueChanged_Lambda([this](int32 V){ DangerCount = V; }) ]) ]
                    + SVerticalBox::Slot().AutoHeight()
                    [ Row(LOCTEXT("Dist", "Distância média (m)"),
                          SNew(SBox).MaxDesiredWidth(90)
                          [ SNew(SNumericEntryBox<float>)
                            .MinValue(0.f)
                            .Value_Lambda([this]{ return AvgDistanceM; })
                            .OnValueChanged_Lambda([this](float V){ AvgDistanceM = V; }) ]) ]

                    + SVerticalBox::Slot().AutoHeight()
                    [ Row(LOCTEXT("Candidates", "Reações candidatas\n(marque; nenhuma = todas do tipo)"),
                          BuildReactionChecklist(CandidateChecks)) ]

                    + SVerticalBox::Slot().AutoHeight().Padding(0, 8, 0, 2)
                    [ SNew(SButton)
                      .HAlign(HAlign_Center)
                      .Text(LOCTEXT("Capture", "📷 Capturar cenário da cena (PIE)"))
                      .ToolTipText(LOCTEXT("CaptureTip",
                          "Preenche o cenário com o que o agente REALMENTE percebe agora no PIE (EntityTags via WorldPerception)"))
                      .OnClicked(this, &SCognitiveTrainingStudio::OnCaptureSceneScenario) ]
                    + SVerticalBox::Slot().AutoHeight().Padding(0, 2, 0, 8)
                    [ SNew(SButton)
                      .HAlign(HAlign_Center)
                      .Text(LOCTEXT("AskAgent", "Pedir decisão ao agente"))
                      .OnClicked(this, &SCognitiveTrainingStudio::OnAskAgent) ]

                    + SVerticalBox::Slot().AutoHeight().Padding(0, 4)
                    [ SAssignNew(TeachChoiceText, STextBlock)
                      .AutoWrapText(true)
                      .Text(LOCTEXT("NoChoice", "Aguardando cenário...")) ]

                    // Correção
                    + SVerticalBox::Slot().AutoHeight().Padding(0, 6)
                    [ SNew(SHorizontalBox)
                      + SHorizontalBox::Slot().FillWidth(0.5f).Padding(0, 0, 4, 0)
                        [ SNew(SButton)
                          .HAlign(HAlign_Center)
                          .IsEnabled_Lambda([this]{ return bHasChoice; })
                          .Text(LOCTEXT("Correct", "✔ Correto"))
                          .OnClicked_Lambda([this]{ return OnFeedback(true); }) ]
                      + SHorizontalBox::Slot().FillWidth(0.5f).Padding(4, 0, 0, 0)
                        [ SNew(SButton)
                          .HAlign(HAlign_Center)
                          .IsEnabled_Lambda([this]{ return bHasChoice; })
                          .Text(LOCTEXT("Wrong", "✘ Errado (usa sugestões abaixo)"))
                          .OnClicked_Lambda([this]{ return OnFeedback(false); }) ] ]

                    + SVerticalBox::Slot().AutoHeight()
                    [ Row(LOCTEXT("Suggestions", "Reações sugeridas\n(marque as corretas)"),
                          BuildReactionChecklist(SuggestionChecks)) ]
                ]
            ]
        ]
    ];
}

// ─────────────────────────────────────────────────────────────────────────────
UCognitiveInferenceSubsystem* SCognitiveTrainingStudio::GetSubsystem() const
{
    // O subsystem vive no GameInstance do PIE. Sem PIE = sem conexão.
    if (!GEditor) return nullptr;
    if (const FWorldContext* Ctx = GEditor->GetPIEWorldContext())
        if (UWorld* World = Ctx->World())
            if (UGameInstance* GI = World->GetGameInstance())
                return GI->GetSubsystem<UCognitiveInferenceSubsystem>();
    return nullptr;
}

FText SCognitiveTrainingStudio::GetConnectionStatusText() const
{
    UCognitiveInferenceSubsystem* Sub = GetSubsystem();
    if (!Sub)
        return LOCTEXT("NoPIE",
            "⚠ Inicie o PIE (Play) com o servidor Python rodando para usar Treino & Ensino.");
    return Sub->IsReady()
        ? LOCTEXT("Ready", "● Conectado ao servidor Python.")
        : LOCTEXT("NotReady", "○ PIE ativo, mas não conectado ao Python ainda.");
}

// ─────────────────────────────────────────────────────────────────────────────
FReply SCognitiveTrainingStudio::OnPickAnimFromContentBrowser()
{
    FContentBrowserModule& CBM =
        FModuleManager::LoadModuleChecked<FContentBrowserModule>("ContentBrowser");
    TArray<FAssetData> Selected;
    CBM.Get().GetSelectedAssets(Selected);

    for (const FAssetData& A : Selected)
    {
        if (A.GetClass() && A.GetClass()->IsChildOf(UAnimSequence::StaticClass()))
        {
            TrainAnimPath->SetText(FText::FromString(A.GetObjectPathString()));
            return FReply::Handled();
        }
    }
    TrainAnimPath->SetText(FText::GetEmpty());
    TrainAnimPath->SetHintText(LOCTEXT("NoAnimSel",
        "Selecione uma AnimSequence no Content Browser primeiro"));
    return FReply::Handled();
}

FReply SCognitiveTrainingStudio::OnRegisterTraining()
{
    UCognitiveInferenceSubsystem* Sub = GetSubsystem();
    const FString Type     = TrainType->GetText().ToString().TrimStartAndEnd();
    const FString Reaction = TrainReactionSel.IsValid() ? *TrainReactionSel : FString();
    const FString AnimPath = TrainAnimPath->GetText().ToString().TrimStartAndEnd();

    if (Type.IsEmpty() || Reaction.IsEmpty())
    {
        TrainLogAccum = TEXT("✘ Preencha tipo de treino e reação.\n") + TrainLogAccum;
        TrainLog->SetText(FText::FromString(TrainLogAccum));
        return FReply::Handled();
    }
    if (!Sub)
    {
        TrainLogAccum = TEXT("✘ Sem PIE/conexão — treino NÃO registrado.\n") + TrainLogAccum;
        TrainLog->SetText(FText::FromString(TrainLogAccum));
        return FReply::Handled();
    }

    FCognitiveTrainingEntry Entry;
    Entry.TrainingType  = Type;
    Entry.ReactionName  = Reaction;
    Entry.AnimationPath = AnimPath;
    Entry.Notes         = TrainNotes->GetText().ToString();

    const bool bSent = Sub->SendTrainingRegister(Entry);
    TrainLogAccum = FString::Printf(TEXT("%s [%s] %s  (%s)\n"),
        bSent ? TEXT("✔") : TEXT("✘"), *Type, *Reaction,
        AnimPath.IsEmpty() ? TEXT("sem animação") : *AnimPath) + TrainLogAccum;
    TrainLog->SetText(FText::FromString(TrainLogAccum));
    return FReply::Handled();
}

// ─────────────────────────────────────────────────────────────────────────────
// Fase 2a: monta o cenário a partir das percepções REAIS do agente no PIE.
// Fonte da verdade: UCognitiveWorldPerceptionComponent::PerceivedEntities do
// primeiro agente encontrado (o NPC com o componente de percepção).
FReply SCognitiveTrainingStudio::OnCaptureSceneScenario()
{
    UWorld* World = nullptr;
    if (GEditor)
        if (const FWorldContext* Ctx = GEditor->GetPIEWorldContext())
            World = Ctx->World();
    if (!World)
    {
        TeachChoiceText->SetText(LOCTEXT("CapNoPIE",
            "✘ Captura requer PIE rodando (o agente percebe em tempo real)."));
        return FReply::Handled();
    }

    // Acha o agente: primeiro ator com WorldPerception no PIE.
    UCognitiveWorldPerceptionComponent* Perception = nullptr;
    AActor* Agent = nullptr;
    for (TActorIterator<AActor> It(World); It; ++It)
    {
        if (UCognitiveWorldPerceptionComponent* P =
                It->FindComponentByClass<UCognitiveWorldPerceptionComponent>())
        {
            Perception = P; Agent = *It; break;
        }
    }
    if (!Perception)
    {
        TeachChoiceText->SetText(LOCTEXT("CapNoAgent",
            "✘ Nenhum ator com Cognitive World Perception encontrado no PIE."));
        return FReply::Handled();
    }

    // Zera e agrega
    EnemyCount = EnemyFacing = AllyCount = ObjectCount = DangerCount = 0;
    float DistSum = 0.f; int32 DistN = 0;
    const FVector AgentLoc = Agent->GetActorLocation();

    for (const FCognitivePerceivedEntity& E : Perception->PerceivedEntities)
    {
        DistSum += E.Distance; ++DistN;

        // Perigo: categoria Hazard OU ameaça alta não-personagem
        const bool bDanger =
            E.Category == ECognitiveEntityCategory::Hazard ||
            (E.ThreatWeight >= 0.7f &&
             E.Category != ECognitiveEntityCategory::Character);

        if (E.Category == ECognitiveEntityCategory::Character)
        {
            if (E.Disposition == ECognitiveDisposition::Enemy)
            {
                ++EnemyCount;
                // "virado para mim": forward do percebido aponta para o agente
                if (const AActor* A = E.Actor.Get())
                {
                    const FVector ToAgent =
                        (AgentLoc - A->GetActorLocation()).GetSafeNormal();
                    if (FVector::DotProduct(A->GetActorForwardVector(), ToAgent) > 0.5f)
                        ++EnemyFacing;
                }
            }
            else if (E.Disposition == ECognitiveDisposition::Ally ||
                     E.Disposition == ECognitiveDisposition::Friend)
            {
                ++AllyCount;
            }
        }
        else if (bDanger)              ++DangerCount;
        else if (E.Category != ECognitiveEntityCategory::Ignore &&
                 E.Category != ECognitiveEntityCategory::Unknown)
                                        ++ObjectCount;
    }
    AvgDistanceM = DistN > 0 ? (DistSum / DistN) / 100.f : 0.f;  // cm → m

    // Descrição gerada (editável depois)
    FString Desc = FString::Printf(
        TEXT("Capturado do PIE (agente: %s): %d inimigo(s), %d encarando; "
             "%d aliado(s); %d objeto(s); %d perigo(s); distância média %.1fm."),
        *Agent->GetName(), EnemyCount, EnemyFacing,
        AllyCount, ObjectCount, DangerCount, AvgDistanceM);
    TeachDescription->SetText(FText::FromString(Desc));

    TeachChoiceText->SetText(FText::FromString(FString::Printf(
        TEXT("📷 Cenário capturado de %d entidade(s) percebida(s). "
             "Revise e peça a decisão."), Perception->PerceivedEntities.Num())));
    return FReply::Handled();
}

// ─────────────────────────────────────────────────────────────────────────────
FReply SCognitiveTrainingStudio::OnAskAgent()
{
    UCognitiveInferenceSubsystem* Sub = GetSubsystem();
    if (!Sub)
    {
        TeachChoiceText->SetText(LOCTEXT("AskNoPIE", "✘ Sem PIE/conexão."));
        return FReply::Handled();
    }

    FCognitiveTeachingScenario S;
    S.TrainingType = TeachType->GetText().ToString().TrimStartAndEnd();
    S.Description  = TeachDescription->GetText().ToString();

    auto AddEntity = [&S, this](const TCHAR* Kind, int32 Count, int32 Facing)
    {
        if (Count <= 0) return;
        FCognitiveScenarioEntity E;
        E.Kind = Kind; E.Count = Count; E.FacingMe = Facing; E.DistanceM = AvgDistanceM;
        S.Entities.Add(MoveTemp(E));
    };
    AddEntity(TEXT("enemy"),  EnemyCount, EnemyFacing);
    AddEntity(TEXT("ally"),   AllyCount, 0);
    AddEntity(TEXT("object"), ObjectCount, 0);
    AddEntity(TEXT("danger"), DangerCount, 0);

    S.CandidateReactions = CheckedReactions(CandidateChecks);

    bHasChoice = false;
    PendingScenarioId = Sub->SendTeachingScenario(S);
    TeachChoiceText->SetText(PendingScenarioId != 0
        ? LOCTEXT("Waiting", "⏳ Cenário enviado — aguardando decisão do agente...")
        : LOCTEXT("SendFail", "✘ Falha ao enviar (não conectado)."));
    return FReply::Handled();
}

EActiveTimerReturnType SCognitiveTrainingStudio::PollTeachingChoice(double, float)
{
    if (PendingScenarioId != 0)
    {
        if (UCognitiveInferenceSubsystem* Sub = GetSubsystem())
        {
            FCognitiveTeachingChoice Choice;
            while (Sub->TryGetTeachingChoice(Choice))
            {
                if (Choice.ScenarioId != PendingScenarioId) continue; // cenário antigo
                LastChoice = Choice;
                bHasChoice = true;
                PendingScenarioId = 0;
                TeachChoiceText->SetText(FText::FromString(FString::Printf(
                    TEXT("🤖 Agente escolheu: \"%s\"  (confiança %.0f%%)%s%s"),
                    *Choice.ChosenReaction, Choice.Confidence * 100.f,
                    Choice.Rationale.IsEmpty() ? TEXT("") : TEXT("\nJustificativa: "),
                    *Choice.Rationale)));
                break;
            }
        }
    }
    return EActiveTimerReturnType::Continue;
}

FReply SCognitiveTrainingStudio::OnFeedback(bool bCorrect)
{
    UCognitiveInferenceSubsystem* Sub = GetSubsystem();
    if (!Sub || !bHasChoice) return FReply::Handled();

    FCognitiveTeachingFeedback F;
    F.ScenarioId     = LastChoice.ScenarioId;
    F.bCorrect       = bCorrect;
    F.ChosenReaction = LastChoice.ChosenReaction;
    if (!bCorrect)
        F.SuggestedReactions = CheckedReactions(SuggestionChecks);

    Sub->SendTeachingFeedback(F);
    bHasChoice = false;
    TeachChoiceText->SetText(bCorrect
        ? LOCTEXT("SentOK", "✔ Feedback enviado: escolha CORRETA. Monte o próximo cenário.")
        : LOCTEXT("SentFix", "✔ Feedback enviado: escolha ERRADA + sugestões. Monte o próximo cenário."));
    return FReply::Handled();
}

#undef LOCTEXT_NAMESPACE
