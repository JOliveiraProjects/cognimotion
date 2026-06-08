#pragma once

#include "CoreMinimal.h"
#include "Widgets/SCompoundWidget.h"
#include "Widgets/Views/SListView.h"
#include "Widgets/Input/SComboBox.h"
#include "CognitiveMotionTypes.h"
#include "Styling/SlateTypes.h"

class USkeleton;
class UAnimBlueprint;
class AActor;

// ─── Wizard Step ──────────────────────────────────────────────────────────────
enum class ECognitiveWizardStep : uint8
{
    SelectSkeleton = 0,
    ConfigureDatabases,
    SetupAnimBlueprint,
    ConfigureNPC,
    Validate,
    Complete
};

// ─── Validation Row (shown in step 5) ─────────────────────────────────────────
struct FCognitiveValidationRow
{
    FString  ActorName;
    bool     bPassed = false;
    FString  Summary;
};

// ─── Bone check row (shown visually in step 1) ────────────────────────────────
struct FCognitiveBoneCheckRow
{
    FString BoneName;
    bool    bPresent = false;   // verde se presente no skeleton selecionado
};

// ─── Main Wizard Slate Widget ─────────────────────────────────────────────────
class COGNITIVEMOTIONINTELLIGENCEEDITOR_API SCognitiveSetupWizard : public SCompoundWidget
{
public:
    SLATE_BEGIN_ARGS(SCognitiveSetupWizard) {}
    SLATE_END_ARGS()

    void Construct(const FArguments& InArgs);

private:
    // ── State ──────────────────────────────────────────────────────────────────
    ECognitiveWizardStep   CurrentStep     = ECognitiveWizardStep::SelectSkeleton;
    USkeleton*             TargetSkeleton  = nullptr;
    FString                SkeletonPath;
    FString                AnimBPOutputPath     = TEXT("/Game/CognitiveMotion");
    FString                PythonHost           = TEXT("127.0.0.1");
    FString                CreatedNPCBlueprintPath;
    // BB-06 FIX: porta default alinhada com UCognitiveMotionLearnerComponent::PythonPort (9000).
    // O valor anterior 9876 causava confusão — o wizard configurava 9876 mas o componente
    // conectava em 9000 se o usuário não editasse manualmente ambos.
    int32                  PythonPort       = 9000;

    // Step 3 result
    UAnimBlueprint*        GeneratedAnimBP  = nullptr;

    // Validation results
    TArray<TSharedPtr<FCognitiveValidationRow>> ValidationRows;

    // Bone check list (step 1 — validação visual contra bones esperados)
    TArray<TSharedPtr<FCognitiveBoneCheckRow>> BoneCheckRows;
    TSharedPtr<SListView<TSharedPtr<FCognitiveBoneCheckRow>>> BoneCheckListView;
    void RebuildBoneCheckList();
    TSharedRef<ITableRow> GenerateBoneCheckRow(
        TSharedPtr<FCognitiveBoneCheckRow> Item,
        const TSharedRef<STableViewBase>& OwnerTable);

    // Log
    TArray<TSharedPtr<FString>> LogLines;
    TSharedPtr<SListView<TSharedPtr<FString>>> LogListView;

    // ── Layout builders ────────────────────────────────────────────────────────
    TSharedRef<SWidget> BuildStepContent();
    TSharedRef<SWidget> BuildStep_SelectSkeleton();
    TSharedRef<SWidget> BuildStep_ConfigureDatabases();
    TSharedRef<SWidget> BuildStep_SetupAnimBlueprint();
    TSharedRef<SWidget> BuildStep_ConfigureNPC();
    TSharedRef<SWidget> BuildStep_Validate();
    TSharedRef<SWidget> BuildStep_Complete();
    TSharedRef<SWidget> BuildNavigationRow();
    TSharedRef<SWidget> BuildProgressBar();
    TSharedRef<SWidget> BuildLogPanel();

    // Validation list row
    TSharedRef<ITableRow> GenerateValidationRow(
        TSharedPtr<FCognitiveValidationRow> Item,
        const TSharedRef<STableViewBase>& OwnerTable);

    TSharedRef<ITableRow> GenerateLogRow(
        TSharedPtr<FString> Item,
        const TSharedRef<STableViewBase>& OwnerTable);

    // ── Button handlers ────────────────────────────────────────────────────────
    FReply OnNextClicked();
    FReply OnBackClicked();
    FReply OnPickSkeletonClicked();
    FReply OnGenerateAnimBPClicked();
    FReply OnApplyToActorClicked();
    FReply OnCreateNPCBlueprintClicked();
    FReply OnRunValidationClicked();
    FReply OnFinishClicked();

    // ── Helpers ────────────────────────────────────────────────────────────────
    bool CanAdvance() const;
    TOptional<float> GetStepProgress() const;
    FText GetStepTitle() const;
    FText GetNextButtonLabel() const;
    EVisibility GetBackVisibility() const;

    void AddLog(const FString& Message);
    void RebuildContent();

    bool DoGenerateAnimBP();
    bool DoValidateScene();
    bool DoApplyToSelectedActor();

    // Root content switcher slot
    TSharedPtr<SVerticalBox> ContentArea;
};
