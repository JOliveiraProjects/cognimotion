#include "CognitiveMotionSetupWizard.h"

#include "Animation/AnimBlueprint.h"
#include "Factories/AnimBlueprintFactory.h"
#include "AssetToolsModule.h"
#include "IAssetTools.h"
#include "AssetRegistry/AssetRegistryModule.h"
#include "Animation/Skeleton.h"
#include "CognitiveAnimInstance.h"
#include "CognitiveMotionLearnerComponent.h"
#include "CognitivePoseRecorderComponent.h"
#include "CognitiveNPCBoneDriver.h"
#include "CognitiveLeaderObserverComponent.h"
#include "CognitiveNPCStateMachine.h"
#include "CognitiveSkeletonManagerComponent.h"
#include "CognitiveWorldPerceptionComponent.h"
#include "PropertyCustomizationHelpers.h"
#include "Kismet2/KismetEditorUtilities.h"
#include "Kismet2/BlueprintEditorUtils.h"          // FBlueprintEditorUtils::MarkBlueprintAsModified
#include "Engine/SimpleConstructionScript.h"       // USimpleConstructionScript
#include "Engine/SCS_Node.h"                       // USCS_Node
#include "GameFramework/Character.h"
#include "Components/SkeletalMeshComponent.h"

#include "EditorStyleSet.h"
#include "Widgets/Layout/SScrollBox.h"
#include "Widgets/Layout/SSeparator.h"
#include "Widgets/Layout/SBox.h"
#include "Widgets/Layout/SBorder.h"
#include "Widgets/Text/STextBlock.h"
#include "Widgets/Input/SButton.h"
#include "Widgets/Input/SEditableTextBox.h"
#include "Widgets/Input/SNumericEntryBox.h"
#include "Widgets/Notifications/SProgressBar.h"
#include "Widgets/SBoxPanel.h"
#include "Widgets/Views/SListView.h"
#include "Widgets/Views/STableRow.h"
#include "Widgets/Views/STableViewBase.h"
#include "Styling/AppStyle.h"

#include "ContentBrowserModule.h"
#include "IContentBrowserSingleton.h"
#include "Engine/World.h"
#include "EngineUtils.h"
#include "GameFramework/Character.h"
#include "Components/SkeletalMeshComponent.h"
#include "Editor.h"
#include "Selection.h"

#define LOCTEXT_NAMESPACE "CognitiveSetupWizard"

// ─────────────────────────────────────────────────────────────────────────────
// Construct
// ─────────────────────────────────────────────────────────────────────────────
void SCognitiveSetupWizard::Construct(const FArguments& InArgs)
{
    CurrentStep = ECognitiveWizardStep::SelectSkeleton;
    RebuildBoneCheckList();   // popula a lista (vazia/vermelha até escolher skeleton)

    ChildSlot
    [
        SNew(SBorder)
        .BorderImage(FAppStyle::GetBrush("ToolPanel.GroupBorder"))
        .Padding(FMargin(8.f))
        [
            SNew(SVerticalBox)

            // ── Header ────────────────────────────────────────────────────────
            + SVerticalBox::Slot()
            .AutoHeight()
            .Padding(0.f, 0.f, 0.f, 8.f)
            [
                SNew(SBorder)
                .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryTop"))
                .Padding(FMargin(12.f, 8.f))
                [
                    SNew(SHorizontalBox)
                    + SHorizontalBox::Slot()
                    .FillWidth(1.f)
                    [
                        SNew(STextBlock)
                        .Text(this, &SCognitiveSetupWizard::GetStepTitle)
                        .TextStyle(FAppStyle::Get(), "LargeText")
                    ]
                    + SHorizontalBox::Slot()
                    .AutoWidth()
                    .VAlign(VAlign_Center)
                    [
                        SNew(STextBlock)
                        .Text_Lambda([this]()
                        {
                            int32 StepNum = (int32)CurrentStep + 1;
                            int32 Total   = (int32)ECognitiveWizardStep::Complete + 1;
                            return FText::Format(LOCTEXT("StepCount","Step {0} of {1}"),
                                FText::AsNumber(StepNum), FText::AsNumber(Total));
                        })
                        .ColorAndOpacity(FSlateColor::UseSubduedForeground())
                    ]
                ]
            ]

            // ── Progress bar ──────────────────────────────────────────────────
            + SVerticalBox::Slot()
            .AutoHeight()
            .Padding(0.f, 0.f, 0.f, 8.f)
            [
                BuildProgressBar()
            ]

            // ── Step content (rebuilt per step) ───────────────────────────────
            + SVerticalBox::Slot()
            .FillHeight(1.f)
            [
                SNew(SScrollBox)
                + SScrollBox::Slot()
                [
                    SAssignNew(ContentArea, SVerticalBox)
                    + SVerticalBox::Slot()
                    [
                        BuildStepContent()
                    ]
                ]
            ]

            // ── Separator ─────────────────────────────────────────────────────
            + SVerticalBox::Slot()
            .AutoHeight()
            .Padding(0.f, 8.f)
            [
                SNew(SSeparator)
            ]

            // ── Log panel ─────────────────────────────────────────────────────
            + SVerticalBox::Slot()
            .MaxHeight(120.f)
            .AutoHeight()
            .Padding(0.f, 0.f, 0.f, 8.f)
            [
                BuildLogPanel()
            ]

            // ── Navigation ────────────────────────────────────────────────────
            + SVerticalBox::Slot()
            .AutoHeight()
            [
                BuildNavigationRow()
            ]
        ]
    ];

    AddLog(TEXT("Cognitive Motion Setup Wizard initialized. Ready."));
}

// ─────────────────────────────────────────────────────────────────────────────
// Progress bar
// ─────────────────────────────────────────────────────────────────────────────
TSharedRef<SWidget> SCognitiveSetupWizard::BuildProgressBar()
{
    return SNew(SProgressBar)
        .Percent(this, &SCognitiveSetupWizard::GetStepProgress);
}

// ─────────────────────────────────────────────────────────────────────────────
// Log panel
// ─────────────────────────────────────────────────────────────────────────────
TSharedRef<SWidget> SCognitiveSetupWizard::BuildLogPanel()
{
    return SNew(SBorder)
        .BorderImage(FAppStyle::GetBrush("ToolPanel.DarkGroupBorder"))
        .Padding(FMargin(4.f))
        [
            SNew(SVerticalBox)
            + SVerticalBox::Slot()
            .AutoHeight()
            [
                SNew(STextBlock)
                .Text(LOCTEXT("LogLabel","Log"))
                .TextStyle(FAppStyle::Get(), "SmallText")
                .ColorAndOpacity(FSlateColor::UseSubduedForeground())
            ]
            + SVerticalBox::Slot()
            .MaxHeight(90.f)
            [
                SAssignNew(LogListView, SListView<TSharedPtr<FString>>)
                .ListItemsSource(&LogLines)
                .OnGenerateRow(this, &SCognitiveSetupWizard::GenerateLogRow)
                .SelectionMode(ESelectionMode::None)
            ]
        ];
}

TSharedRef<ITableRow> SCognitiveSetupWizard::GenerateLogRow(
    TSharedPtr<FString> Item,
    const TSharedRef<STableViewBase>& OwnerTable)
{
    return SNew(STableRow<TSharedPtr<FString>>, OwnerTable)
        [
            SNew(STextBlock)
            .Text(FText::FromString(*Item))
            .TextStyle(FAppStyle::Get(), "SmallText")
        ];
}

// ─────────────────────────────────────────────────────────────────────────────
// Step content dispatcher
// ─────────────────────────────────────────────────────────────────────────────
TSharedRef<SWidget> SCognitiveSetupWizard::BuildStepContent()
{
    switch (CurrentStep)
    {
    case ECognitiveWizardStep::SelectSkeleton:       return BuildStep_SelectSkeleton();
    case ECognitiveWizardStep::ConfigureDatabases:   return BuildStep_ConfigureDatabases();
    case ECognitiveWizardStep::SetupAnimBlueprint:   return BuildStep_SetupAnimBlueprint();
    case ECognitiveWizardStep::ConfigureNPC:         return BuildStep_ConfigureNPC();
    case ECognitiveWizardStep::Validate:             return BuildStep_Validate();
    case ECognitiveWizardStep::Complete:             return BuildStep_Complete();
    default:                                         return SNullWidget::NullWidget;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// STEP 1 — Select Skeleton
// ─────────────────────────────────────────────────────────────────────────────
TSharedRef<SWidget> SCognitiveSetupWizard::BuildStep_SelectSkeleton()
{
    return SNew(SVerticalBox)

    + SVerticalBox::Slot().AutoHeight().Padding(0, 4)
    [
        SNew(STextBlock)
        .AutoWrapText(true)
        .Text(LOCTEXT("Step1Desc",
            "Selecione o USkeleton que o NPC utiliza.\n"
            "Clique no ícone de lupa ao lado do campo para abrir o asset picker.\n"
            "O skeleton escolhido será usado para gerar o AnimBlueprint no Step 3."))
    ]

    + SVerticalBox::Slot().AutoHeight().Padding(0, 8)
    [
        SNew(SObjectPropertyEntryBox)
        .AllowedClass(USkeleton::StaticClass())
        .ObjectPath_Lambda([this]() -> FString
        {
            return SkeletonPath;
        })
        .OnObjectChanged_Lambda([this](const FAssetData& AssetData)
        {
            if (!AssetData.IsValid()) return;
            TargetSkeleton = Cast<USkeleton>(AssetData.GetAsset());
            if (TargetSkeleton)
            {
                SkeletonPath = AssetData.GetObjectPathString();
                AddLog(FString::Printf(TEXT("Skeleton selecionado: %s"),
                    *TargetSkeleton->GetName()));
            }
            else
            {
                SkeletonPath.Empty();
            }
            RebuildBoneCheckList();
            RebuildContent();
        })
        .AllowClear(true)
        .DisplayThumbnail(false)
    ]

    + SVerticalBox::Slot().AutoHeight().Padding(0, 4)
    [
        SNew(STextBlock)
        .AutoWrapText(true)
        .ColorAndOpacity_Lambda([this]() -> FSlateColor
        {
            if (!TargetSkeleton)
                return FSlateColor(FLinearColor(1.f, 0.6f, 0.f, 1.f));
            const bool bL = TargetSkeleton->GetReferenceSkeleton()
                .FindBoneIndex(FName("foot_l")) != INDEX_NONE;
            const bool bR = TargetSkeleton->GetReferenceSkeleton()
                .FindBoneIndex(FName("foot_r")) != INDEX_NONE;
            return (bL && bR)
                ? FSlateColor(FLinearColor(0.2f, 0.8f, 0.2f, 1.f))
                : FSlateColor(FLinearColor(1.f, 0.6f, 0.f, 1.f));
        })
        .Text_Lambda([this]() -> FText
        {
            if (!TargetSkeleton)
                return LOCTEXT("NoSkelWarning", "⚠ Nenhum skeleton selecionado.");
            const bool bL = TargetSkeleton->GetReferenceSkeleton()
                .FindBoneIndex(FName("foot_l")) != INDEX_NONE;
            const bool bR = TargetSkeleton->GetReferenceSkeleton()
                .FindBoneIndex(FName("foot_r")) != INDEX_NONE;
            if (bL && bR)
                return FText::Format(
                    LOCTEXT("SkelOK", "✔ {0} — foot_l e foot_r encontrados."),
                    FText::FromString(TargetSkeleton->GetName()));
            return FText::Format(
                LOCTEXT("SkelWarn", "⚠ {0} — foot_l ou foot_r não encontrados (opcional)."),
                FText::FromString(TargetSkeleton->GetName()));
        })
    ]

    // ── Validação visual de bones: cada bone esperado com ponto verde/vermelho ──
    + SVerticalBox::Slot().AutoHeight().Padding(0, 8)
    [
        SNew(STextBlock)
        .Text(LOCTEXT("BoneCheckTitle",
            "Verificação de bones (humanoide padrão):"))
    ]

    + SVerticalBox::Slot().AutoHeight().Padding(0, 4)
    [
        SNew(SBox).MaxDesiredHeight(180.f)
        [
            SAssignNew(BoneCheckListView, SListView<TSharedPtr<FCognitiveBoneCheckRow>>)
            .ListItemsSource(&BoneCheckRows)
            .OnGenerateRow(this, &SCognitiveSetupWizard::GenerateBoneCheckRow)
            .SelectionMode(ESelectionMode::None)
        ]
    ];
}

// ── Bones essenciais esperados num humanoide (Mannequin/UE5). A validação é
//    informativa: presença = verde, ausência = vermelho. Não bloqueia o wizard.
static const TArray<FString>& CognitiveExpectedBones()
{
    static const TArray<FString> Bones = {
        TEXT("root"), TEXT("pelvis"),
        TEXT("spine_01"), TEXT("spine_02"), TEXT("spine_03"),
        TEXT("neck_01"), TEXT("head"),
        TEXT("clavicle_l"), TEXT("upperarm_l"), TEXT("lowerarm_l"), TEXT("hand_l"),
        TEXT("clavicle_r"), TEXT("upperarm_r"), TEXT("lowerarm_r"), TEXT("hand_r"),
        TEXT("thigh_l"), TEXT("calf_l"), TEXT("foot_l"),
        TEXT("thigh_r"), TEXT("calf_r"), TEXT("foot_r")
    };
    return Bones;
}

void SCognitiveSetupWizard::RebuildBoneCheckList()
{
    BoneCheckRows.Reset();
    const TArray<FString>& Expected = CognitiveExpectedBones();
    for (const FString& BoneName : Expected)
    {
        TSharedPtr<FCognitiveBoneCheckRow> Row = MakeShared<FCognitiveBoneCheckRow>();
        Row->BoneName = BoneName;
        Row->bPresent = TargetSkeleton
            ? (TargetSkeleton->GetReferenceSkeleton()
                  .FindBoneIndex(FName(*BoneName)) != INDEX_NONE)
            : false;
        BoneCheckRows.Add(Row);
    }
    if (BoneCheckListView.IsValid())
    {
        BoneCheckListView->RequestListRefresh();
    }
}

TSharedRef<ITableRow> SCognitiveSetupWizard::GenerateBoneCheckRow(
    TSharedPtr<FCognitiveBoneCheckRow> Item,
    const TSharedRef<STableViewBase>& OwnerTable)
{
    const bool bOk = Item.IsValid() && Item->bPresent;
    const FLinearColor Dot = bOk
        ? FLinearColor(0.2f, 0.8f, 0.2f, 1.f)   // verde
        : FLinearColor(0.85f, 0.2f, 0.2f, 1.f); // vermelho

    return SNew(STableRow<TSharedPtr<FCognitiveBoneCheckRow>>, OwnerTable)
    [
        SNew(SHorizontalBox)
        + SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(4, 2)
        [
            SNew(STextBlock)
            .Text(FText::FromString(bOk ? TEXT("●") : TEXT("●")))
            .ColorAndOpacity(FSlateColor(Dot))
        ]
        + SHorizontalBox::Slot().FillWidth(1.f).VAlign(VAlign_Center).Padding(4, 2)
        [
            SNew(STextBlock)
            .Text(Item.IsValid()
                ? FText::FromString(Item->BoneName)
                : FText::GetEmpty())
        ]
        + SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(4, 2)
        [
            SNew(STextBlock)
            .Text(bOk
                ? LOCTEXT("BonePresent", "presente")
                : LOCTEXT("BoneMissing", "ausente"))
            .ColorAndOpacity(FSlateColor(Dot))
        ]
    ];
}

// ─────────────────────────────────────────────────────────────────────────────
// STEP 2 — Informational (Motion Matching / PoseSearch removidos)
// ─────────────────────────────────────────────────────────────────────────────
TSharedRef<SWidget> SCognitiveSetupWizard::BuildStep_ConfigureDatabases()
{
    return SNew(SVerticalBox)

    + SVerticalBox::Slot().AutoHeight().Padding(0,4)
    [
        SNew(STextBlock)
        .AutoWrapText(true)
        .Text(LOCTEXT("Step2Desc",
            "Motion Matching e PoseSearch foram removidos deste plugin.\n\n"
            "A animação do NPC é gerada diretamente pelo Python: "
            "o servidor recebe todos os bones do líder, aprende os movimentos "
            "e devolve bone transforms para o NPC.\n\n"
            "Nenhuma configuração de banco de dados é necessária.\n\n"
            "Clique Next para continuar para a geração do AnimBlueprint."))
    ]

    + SVerticalBox::Slot().AutoHeight().Padding(0,8)
    [
        SNew(SBorder)
        .BorderImage(FAppStyle::GetBrush("MessageLog.Warning"))
        .Padding(8.f)
        [
            SNew(STextBlock)
            .AutoWrapText(true)
            .Text(LOCTEXT("Step2Tip",
                "TIP: You can always come back and re-run the wizard after creating your databases. "
                "The AnimBP generation in the next step does not require databases."))
        ]
    ];
}

// ─────────────────────────────────────────────────────────────────────────────
// STEP 3 — Generate AnimBP
// ─────────────────────────────────────────────────────────────────────────────
TSharedRef<SWidget> SCognitiveSetupWizard::BuildStep_SetupAnimBlueprint()
{
    return SNew(SVerticalBox)

    + SVerticalBox::Slot().AutoHeight().Padding(0,4)
    [
        SNew(STextBlock)
        .AutoWrapText(true)
        .Text(LOCTEXT("Step3Desc",
            "Generate a new Animation Blueprint parented to UCognitiveAnimInstance. "
            "The blueprint will be created at the path below. "
            "You can add the Cognitive Motion Matching node inside it after creation."))
    ]

    + SVerticalBox::Slot().AutoHeight().Padding(0,8)
    [
        SNew(SHorizontalBox)
        + SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(0,0,8,0)
        [
            SNew(STextBlock).Text(LOCTEXT("OutputPathLabel","Output path:"))
        ]
        + SHorizontalBox::Slot().FillWidth(1.f)
        [
            SNew(SEditableTextBox)
            .Text_Lambda([this](){ return FText::FromString(AnimBPOutputPath); })
            .OnTextCommitted_Lambda([this](const FText& T, ETextCommit::Type)
            {
                AnimBPOutputPath = T.ToString();
            })
        ]
    ]

    + SVerticalBox::Slot().AutoHeight().Padding(0,4)
    [
        SNew(SHorizontalBox)
        + SHorizontalBox::Slot().AutoWidth()
        [
            SNew(SButton)
            .Text(LOCTEXT("GenerateBtn","Generate AnimBP Now"))
            .IsEnabled_Lambda([this](){ return TargetSkeleton != nullptr; })
            .OnClicked(this, &SCognitiveSetupWizard::OnGenerateAnimBPClicked)
        ]
        + SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(8,0,0,0)
        [
            SNew(STextBlock)
            .Text_Lambda([this]()
            {
                if (!GeneratedAnimBP)
                    return TargetSkeleton
                        ? LOCTEXT("GenReady","Ready to generate.")
                        : LOCTEXT("GenNeedSkel","Select a skeleton in Step 1 first.");
                return FText::Format(LOCTEXT("GenDone","✔  Created: {0}"),
                    FText::FromString(GeneratedAnimBP->GetName()));
            })
            .ColorAndOpacity_Lambda([this]()
            {
                return GeneratedAnimBP
                    ? FSlateColor(FLinearColor(0.2f,0.8f,0.2f,1))
                    : FSlateColor::UseForeground();
            })
        ]
    ];
}

// ─────────────────────────────────────────────────────────────────────────────
// STEP 4 — Configure NPC
// ─────────────────────────────────────────────────────────────────────────────
TSharedRef<SWidget> SCognitiveSetupWizard::BuildStep_ConfigureNPC()
{
    return SNew(SVerticalBox)

    + SVerticalBox::Slot().AutoHeight().Padding(0,4)
    [
        SNew(STextBlock)
        .AutoWrapText(true)
        .Text(LOCTEXT("Step4Desc",
            "Passo 1: Clique em 'Criar BP_CognitiveNPC' para criar o Character Blueprint "
            "do NPC. Ele sera criado em /Game/CognitiveMotion/ e inserido no level.\n"
            "Passo 2: Configure o IP e porta do Python.\n"
            "Passo 3: Clique em 'Aplicar Componentes' para adicionar LearnerComponent "
            "e PoseRecorderComponent ao NPC selecionado."))
    ]

    // Botao: criar Blueprint do NPC
    + SVerticalBox::Slot().AutoHeight().Padding(0,8)
    [
        SNew(SHorizontalBox)
        + SHorizontalBox::Slot().AutoWidth()
        [
            SNew(SButton)
            .Text(LOCTEXT("CreateNPCBtn","1. Criar BP_CognitiveNPC no Level"))
            .ToolTipText(LOCTEXT("CreateNPCTip",
                "Cria um Blueprint ACharacter em /Game/CognitiveMotion/BP_CognitiveNPC, "
                "insere no level atual e seleciona automaticamente."))
            .OnClicked(this, &SCognitiveSetupWizard::OnCreateNPCBlueprintClicked)
        ]
        + SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(8,0,0,0)
        [
            SNew(STextBlock)
            .Text_Lambda([this]() -> FText {
                return CreatedNPCBlueprintPath.IsEmpty()
                    ? LOCTEXT("NoBPYet","(ou selecione manualmente um Character no viewport)")
                    : FText::Format(LOCTEXT("BPDone","Criado: {0}"),
                        FText::FromString(FPaths::GetBaseFilename(CreatedNPCBlueprintPath)));
            })
        ]
    ]

    + SVerticalBox::Slot().AutoHeight().Padding(0,4)
    [
        SNew(SHorizontalBox)
        + SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(0,0,8,0)
        [ SNew(STextBlock).Text(LOCTEXT("HostLabel","2. Python Host:")) ]
        + SHorizontalBox::Slot().FillWidth(1.f)
        [
            SNew(SEditableTextBox)
            .Text_Lambda([this](){ return FText::FromString(PythonHost); })
            .OnTextCommitted_Lambda([this](const FText& T, ETextCommit::Type)
            { PythonHost = T.ToString(); })
        ]
    ]

    + SVerticalBox::Slot().AutoHeight().Padding(0,4)
    [
        SNew(SHorizontalBox)
        + SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(0,0,8,0)
        [ SNew(STextBlock).Text(LOCTEXT("PortLabel","   Python Port:")) ]
        + SHorizontalBox::Slot().AutoWidth()
        [
            SNew(SNumericEntryBox<int32>)
            .Value_Lambda([this]() -> TOptional<int32> { return PythonPort; })
            .OnValueCommitted_Lambda([this](int32 V, ETextCommit::Type){ PythonPort = V; })
            .MinValue(1024).MaxValue(65535)
        ]
    ]

    + SVerticalBox::Slot().AutoHeight().Padding(0,8)
    [
        SNew(SButton)
        .Text(LOCTEXT("ApplyBtn","3. Aplicar Componentes ao Ator Selecionado"))
        .ToolTipText(LOCTEXT("ApplyTip",
            "Adiciona CognitiveMotionLearnerComponent e CognitivePoseRecorderComponent "
            "ao ator selecionado no viewport."))
        .OnClicked(this, &SCognitiveSetupWizard::OnApplyToActorClicked)
    ]

    + SVerticalBox::Slot().AutoHeight().Padding(0,4)
    [
        SNew(STextBlock)
        .AutoWrapText(true)
        .ColorAndOpacity(FSlateColor::UseSubduedForeground())
        .Text(LOCTEXT("Step4Note",
            "Apos aplicar: abra o Blueprint e em Mesh->Anim Class selecione o AnimBP "
            "do Step 3. Adicione CognitiveNPCBoneDriver e CognitiveLeaderObserverComponent."))
    ];
}

// ─────────────────────────────────────────────────────────────────────────────
// STEP 5 — Validate
// ─────────────────────────────────────────────────────────────────────────────
TSharedRef<SWidget> SCognitiveSetupWizard::BuildStep_Validate()
{
    TSharedRef<SListView<TSharedPtr<FCognitiveValidationRow>>> ValidationList =
        SNew(SListView<TSharedPtr<FCognitiveValidationRow>>)
        .ListItemsSource(&ValidationRows)
        .OnGenerateRow(this, &SCognitiveSetupWizard::GenerateValidationRow)
        .SelectionMode(ESelectionMode::None);

    return SNew(SVerticalBox)

    + SVerticalBox::Slot().AutoHeight().Padding(0,4)
    [
        SNew(STextBlock)
        .AutoWrapText(true)
        .Text(LOCTEXT("Step5Desc",
            "Run validation to check all Cognitive NPC Actors in the currently open level. "
            "Each actor is checked for required components and a valid AnimBP."))
    ]

    + SVerticalBox::Slot().AutoHeight().Padding(0,8)
    [
        SNew(SButton)
        .Text(LOCTEXT("ValidateBtn","Run Validation Now"))
        .OnClicked(this, &SCognitiveSetupWizard::OnRunValidationClicked)
    ]

    // Column header row (manual)
    + SVerticalBox::Slot().AutoHeight().Padding(0,8,0,0)
    [
        SNew(SBorder)
        .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryTop"))
        .Padding(FMargin(4.f, 2.f))
        [
            SNew(SHorizontalBox)
            + SHorizontalBox::Slot().FillWidth(0.35f)
            [ SNew(STextBlock).Text(LOCTEXT("ColActor","Actor")).TextStyle(FAppStyle::Get(),"SmallText") ]
            + SHorizontalBox::Slot().FillWidth(0.15f)
            [ SNew(STextBlock).Text(LOCTEXT("ColStatus","Status")).TextStyle(FAppStyle::Get(),"SmallText") ]
            + SHorizontalBox::Slot().FillWidth(0.5f)
            [ SNew(STextBlock).Text(LOCTEXT("ColSummary","Summary")).TextStyle(FAppStyle::Get(),"SmallText") ]
        ]
    ]

    + SVerticalBox::Slot().MaxHeight(200.f)
    [
        SNew(SBorder)
        .BorderImage(FAppStyle::GetBrush("ToolPanel.DarkGroupBorder"))
        [
            ValidationList
        ]
    ];
}

// ─────────────────────────────────────────────────────────────────────────────
// STEP 6 — Complete
// ─────────────────────────────────────────────────────────────────────────────
TSharedRef<SWidget> SCognitiveSetupWizard::BuildStep_Complete()
{
    return SNew(SVerticalBox)

    + SVerticalBox::Slot().AutoHeight().Padding(0,16)
    [
        SNew(SBorder)
        .BorderImage(FAppStyle::GetBrush("Graph.PlayInEditor"))
        .Padding(16.f)
        [
            SNew(SVerticalBox)
            + SVerticalBox::Slot().AutoHeight()
            [
                SNew(STextBlock)
                .Text(LOCTEXT("DoneTitle","✔  Cognitive Motion Setup Complete!"))
                .TextStyle(FAppStyle::Get(), "LargeText")
                .ColorAndOpacity(FLinearColor(0.2f,0.9f,0.2f,1))
            ]
            + SVerticalBox::Slot().AutoHeight().Padding(0,8)
            [
                SNew(STextBlock)
                .AutoWrapText(true)
                .Text(LOCTEXT("DoneBody",
                    "Seu NPC está configurado para o Cognitive Motion Intelligence.\n\n"
                    "Próximos passos:\n"
                    "  1. Abra o AnimBP gerado e verifique o nó Cognitive Motion Matching.\n"
                    "  2. Adicione o CognitiveNPCBoneDriver ao NPC e defina o TargetLeader.\n"
                    "  3. Inicie o servidor Python: python main.py --host 0.0.0.0 --port 9000\n"
                    "  4. Pressione Play — o NPC conecta automaticamente e aprende com o líder."))
            ]
        ]
    ];
}

// ─────────────────────────────────────────────────────────────────────────────
// Navigation row
// ─────────────────────────────────────────────────────────────────────────────
TSharedRef<SWidget> SCognitiveSetupWizard::BuildNavigationRow()
{
    return SNew(SHorizontalBox)

    + SHorizontalBox::Slot()
    .AutoWidth()
    [
        SNew(SButton)
        .Text(LOCTEXT("BackBtn","← Back"))
        .Visibility(this, &SCognitiveSetupWizard::GetBackVisibility)
        .OnClicked(this, &SCognitiveSetupWizard::OnBackClicked)
    ]

    + SHorizontalBox::Slot().FillWidth(1.f)

    + SHorizontalBox::Slot()
    .AutoWidth()
    [
        SNew(SButton)
        .Text(this, &SCognitiveSetupWizard::GetNextButtonLabel)
        .IsEnabled(this, &SCognitiveSetupWizard::CanAdvance)
        .ButtonColorAndOpacity(FLinearColor(0.1f,0.4f,0.9f,1))
        .OnClicked(this, &SCognitiveSetupWizard::OnNextClicked)
    ];
}

// ─────────────────────────────────────────────────────────────────────────────
// Validation row widget
// ─────────────────────────────────────────────────────────────────────────────
TSharedRef<ITableRow> SCognitiveSetupWizard::GenerateValidationRow(
    TSharedPtr<FCognitiveValidationRow> Item,
    const TSharedRef<STableViewBase>& OwnerTable)
{
    FLinearColor StatusColor = Item->bPassed
        ? FLinearColor(0.2f,0.8f,0.2f,1)
        : FLinearColor(0.9f,0.2f,0.2f,1);
    FString StatusStr = Item->bPassed ? TEXT("✔ PASS") : TEXT("✘ FAIL");

    return SNew(STableRow<TSharedPtr<FCognitiveValidationRow>>, OwnerTable)
        .Padding(FMargin(4.f, 2.f))
        [
            SNew(SHorizontalBox)
            + SHorizontalBox::Slot()
            .FillWidth(0.35f)
            .Padding(2.f, 0.f)
            [
                SNew(STextBlock)
                .Text(FText::FromString(Item->ActorName))
                .AutoWrapText(true)
            ]
            + SHorizontalBox::Slot()
            .FillWidth(0.15f)
            .Padding(2.f, 0.f)
            [
                SNew(STextBlock)
                .Text(FText::FromString(StatusStr))
                .ColorAndOpacity(StatusColor)
            ]
            + SHorizontalBox::Slot()
            .FillWidth(0.5f)
            .Padding(2.f, 0.f)
            [
                SNew(STextBlock)
                .Text(FText::FromString(Item->Summary))
                .AutoWrapText(true)
            ]
        ];
}

// ─────────────────────────────────────────────────────────────────────────────
// Button Handlers
// ─────────────────────────────────────────────────────────────────────────────

FReply SCognitiveSetupWizard::OnPickSkeletonClicked()
{
    // Substituído por SObjectPropertyEntryBox em BuildStep_SelectSkeleton.
    // Mantido aqui para não quebrar referências de compilação.
    return FReply::Handled();
}

FReply SCognitiveSetupWizard::OnGenerateAnimBPClicked()
{
    if (DoGenerateAnimBP())
        RebuildContent();
    return FReply::Handled();
}

// ─────────────────────────────────────────────────────────────────────────────
// Adiciona um componente ao Blueprint via SCS (Simple Construction Script).
// Esta é a forma correta de adicionar componentes ao Blueprint Class em UE5.
// AddInstanceComponent() adiciona apenas à instância do level — não ao Blueprint.
// ─────────────────────────────────────────────────────────────────────────────
static USCS_Node* AddComponentToBlueprint(
    UBlueprint*  BP,
    TSubclassOf<UActorComponent> CompClass,
    FName        CompName)
{
    if (!BP || !BP->SimpleConstructionScript || !CompClass) return nullptr;

    // Não duplicar se já existe
    for (USCS_Node* Node : BP->SimpleConstructionScript->GetAllNodes())
    {
        if (Node->ComponentClass && Node->ComponentClass->IsChildOf(CompClass))
            return Node;
    }

    USCS_Node* NewNode = BP->SimpleConstructionScript->CreateNode(CompClass, CompName);
    BP->SimpleConstructionScript->AddNode(NewNode);
    return NewNode;
}

FReply SCognitiveSetupWizard::OnCreateNPCBlueprintClicked()
{
    if (!GEditor) return FReply::Handled();

    // ── 1. Cria o package ─────────────────────────────────────────────────────
    const FString BPPath = TEXT("/Game/CognitiveMotion/BP_CognitiveNPC");
    UPackage* Package = CreatePackage(*BPPath);
    if (!Package)
    {
        AddLog(TEXT("ERRO: Não foi possível criar o package /Game/CognitiveMotion/BP_CognitiveNPC."));
        return FReply::Handled();
    }

    // ── 2. Cria o Blueprint baseado em ACharacter ─────────────────────────────
    UBlueprint* BP = FKismetEditorUtilities::CreateBlueprint(
        ACharacter::StaticClass(), Package,
        FName("BP_CognitiveNPC"), BPTYPE_Normal,
        UBlueprint::StaticClass(), UBlueprintGeneratedClass::StaticClass(),
        FName("CognitiveNPCCreated"));

    if (!BP)
    {
        AddLog(TEXT("ERRO: CreateBlueprint falhou."));
        return FReply::Handled();
    }

    // ── 3. Adiciona TODOS os componentes ao Blueprint (via SCS) ───────────────
    AddComponentToBlueprint(BP, UCognitiveMotionLearnerComponent::StaticClass(),
        FName("CognitiveMotionLearner"));

    AddComponentToBlueprint(BP, UCognitivePoseRecorderComponent::StaticClass(),
        FName("CognitivePoseRecorder"));

    AddComponentToBlueprint(BP, UCognitiveNPCBoneDriver::StaticClass(),
        FName("CognitiveNPCBoneDriver"));

    AddComponentToBlueprint(BP, UCognitiveLeaderObserverComponent::StaticClass(),
        FName("CognitiveLeaderObserver"));

    AddComponentToBlueprint(BP, UCognitiveNPCStateMachine::StaticClass(),
        FName("CognitiveNPCStateMachine"));

    AddComponentToBlueprint(BP, UCognitiveSkeletonManagerComponent::StaticClass(),
        FName("CognitiveSkeletonManager"));

    AddComponentToBlueprint(BP, UCognitiveWorldPerceptionComponent::StaticClass(),
        FName("CognitiveWorldPerception"));

    AddLog(TEXT("Componentes adicionados: LearnerComponent, PoseRecorder, BoneDriver, "
                "LeaderObserver, StateMachine."));

    // ── 4. Seta AnimClass no SkeletalMeshComponent do Blueprint ──────────────
    if (GeneratedAnimBP && GeneratedAnimBP->GeneratedClass && BP->SimpleConstructionScript)
    {
        for (USCS_Node* Node : BP->SimpleConstructionScript->GetAllNodes())
        {
            if (!Node->ComponentClass) continue;
            if (!Node->ComponentClass->IsChildOf(USkeletalMeshComponent::StaticClass())) continue;

            USkeletalMeshComponent* MeshTemplate =
                Cast<USkeletalMeshComponent>(Node->ComponentTemplate);
            if (MeshTemplate)
            {
                MeshTemplate->SetAnimInstanceClass(GeneratedAnimBP->GeneratedClass);
                AddLog(FString::Printf(TEXT("AnimClass setado: %s"),
                    *GeneratedAnimBP->GetName()));
            }
        }
    }
    else
    {
        AddLog(TEXT("AVISO: AnimBP não gerado ainda (Step 3). "
                    "Abra o Blueprint e sete Mesh → Anim Class manualmente."));
    }

    // ── 5. Compila o Blueprint ────────────────────────────────────────────────
    FBlueprintEditorUtils::MarkBlueprintAsModified(BP);
    FKismetEditorUtilities::CompileBlueprint(BP);

    FAssetRegistryModule::AssetCreated(BP);
    Package->MarkPackageDirty();
    CreatedNPCBlueprintPath = BPPath;

    AddLog(TEXT("Blueprint compilado. Salve com Ctrl+S ou File → Save All."));

    // ── 6. Insere instância no level e seleciona ──────────────────────────────
    UWorld* World = GEditor->GetEditorWorldContext().World();
    if (World && BP->GeneratedClass)
    {
        GEditor->SelectNone(false, true);

        FActorSpawnParameters Params;
        Params.SpawnCollisionHandlingOverride =
            ESpawnActorCollisionHandlingMethod::AdjustIfPossibleButAlwaysSpawn;

        AActor* NewActor = World->SpawnActor<AActor>(
            BP->GeneratedClass, FTransform::Identity, Params);

        if (NewActor)
        {
            GEditor->SelectActor(NewActor, true, true);
            AddLog(TEXT("BP_CognitiveNPC inserido e selecionado no level."));
        }
        GEditor->RedrawAllViewports();
    }

    // Abre o Blueprint no editor para o usuário ver
    GEditor->GetEditorSubsystem<UAssetEditorSubsystem>()->OpenEditorForAsset(BP);

    RebuildContent();
    return FReply::Handled();
}

FReply SCognitiveSetupWizard::OnApplyToActorClicked()
{
    DoApplyToSelectedActor();
    return FReply::Handled();
}

FReply SCognitiveSetupWizard::OnRunValidationClicked()
{
    DoValidateScene();
    RebuildContent();
    return FReply::Handled();
}

FReply SCognitiveSetupWizard::OnNextClicked()
{
    if (CurrentStep == ECognitiveWizardStep::Complete)
        return FReply::Handled();

    int32 NextStep = (int32)CurrentStep + 1;
    CurrentStep = (ECognitiveWizardStep)NextStep;

    AddLog(FString::Printf(TEXT("Advanced to step %d"), NextStep + 1));
    RebuildContent();
    return FReply::Handled();
}

FReply SCognitiveSetupWizard::OnBackClicked()
{
    if ((int32)CurrentStep == 0) return FReply::Handled();
    CurrentStep = (ECognitiveWizardStep)((int32)CurrentStep - 1);
    AddLog(FString::Printf(TEXT("Went back to step %d"), (int32)CurrentStep + 1));
    RebuildContent();
    return FReply::Handled();
}

FReply SCognitiveSetupWizard::OnFinishClicked()
{
    AddLog(TEXT("Setup complete. Close this window when ready."));
    return FReply::Handled();
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

bool SCognitiveSetupWizard::CanAdvance() const
{
    if (CurrentStep == ECognitiveWizardStep::SelectSkeleton)
        return TargetSkeleton != nullptr;
    if (CurrentStep == ECognitiveWizardStep::Complete)
        return false;
    return true;
}

TOptional<float> SCognitiveSetupWizard::GetStepProgress() const
{
    return (float)((int32)CurrentStep + 1) /
           (float)((int32)ECognitiveWizardStep::Complete + 1);
}

FText SCognitiveSetupWizard::GetStepTitle() const
{
    switch (CurrentStep)
    {
    case ECognitiveWizardStep::SelectSkeleton:     return LOCTEXT("T1","1 — Select Skeleton");
    case ECognitiveWizardStep::ConfigureDatabases: return LOCTEXT("T2","2 — Configure Databases");
    case ECognitiveWizardStep::SetupAnimBlueprint: return LOCTEXT("T3","3 — Generate AnimBlueprint");
    case ECognitiveWizardStep::ConfigureNPC:       return LOCTEXT("T4","4 — Configure NPC Actor");
    case ECognitiveWizardStep::Validate:           return LOCTEXT("T5","5 — Validate Setup");
    case ECognitiveWizardStep::Complete:           return LOCTEXT("T6","6 — Complete");
    default:                                       return FText::GetEmpty();
    }
}

FText SCognitiveSetupWizard::GetNextButtonLabel() const
{
    if (CurrentStep == ECognitiveWizardStep::Validate)
        return LOCTEXT("NextFinish","Finish →");
    if (CurrentStep == ECognitiveWizardStep::Complete)
        return LOCTEXT("Done","Done");
    return LOCTEXT("Next","Next →");
}

EVisibility SCognitiveSetupWizard::GetBackVisibility() const
{
    return (int32)CurrentStep > 0 ? EVisibility::Visible : EVisibility::Hidden;
}

void SCognitiveSetupWizard::AddLog(const FString& Message)
{
    FString Timestamped = FString::Printf(TEXT("[%s] %s"),
        *FDateTime::Now().ToString(TEXT("%H:%M:%S")), *Message);
    LogLines.Add(MakeShared<FString>(Timestamped));
    if (LogListView.IsValid())
    {
        LogListView->RequestListRefresh();
        if (!LogLines.IsEmpty())
            LogListView->RequestScrollIntoView(LogLines.Last());
    }
    UE_LOG(LogTemp, Log, TEXT("[CMI Wizard] %s"), *Message);
}

void SCognitiveSetupWizard::RebuildContent()
{
    if (ContentArea.IsValid())
    {
        ContentArea->ClearChildren();
        ContentArea->AddSlot()
        [
            BuildStepContent()
        ];
    }
}

bool SCognitiveSetupWizard::DoGenerateAnimBP()
{
    if (!TargetSkeleton)
    {
        AddLog(TEXT("ERROR: No skeleton selected. Cannot generate AnimBP."));
        return false;
    }

    IAssetTools& AssetTools =
        FModuleManager::LoadModuleChecked<FAssetToolsModule>("AssetTools").Get();

    UAnimBlueprintFactory* Factory = NewObject<UAnimBlueprintFactory>();
    Factory->TargetSkeleton = TargetSkeleton;
    Factory->ParentClass    = UCognitiveAnimInstance::StaticClass();

    const FString AssetName = FString::Printf(
        TEXT("ABP_Cognitive_%s"), *TargetSkeleton->GetName());

    UObject* Asset = AssetTools.CreateAsset(
        AssetName, AnimBPOutputPath, UAnimBlueprint::StaticClass(), Factory);

    GeneratedAnimBP = Cast<UAnimBlueprint>(Asset);
    if (GeneratedAnimBP)
    {
        AddLog(FString::Printf(TEXT("AnimBP created: %s/%s"),
            *AnimBPOutputPath, *AssetName));
        return true;
    }

    AddLog(FString::Printf(TEXT("ERROR: Failed to create AnimBP at %s/%s. "
        "Check the output path exists and the asset doesn't already exist."),
        *AnimBPOutputPath, *AssetName));
    return false;
}

bool SCognitiveSetupWizard::DoValidateScene()
{
    UWorld* World = GEditor ? GEditor->GetEditorWorldContext().World() : nullptr;
    if (!World)
    {
        AddLog(TEXT("ERROR: No editor world found. Open a level first."));
        return false;
    }

    ValidationRows.Empty();
    int32 Passed = 0, Failed = 0;

    for (TActorIterator<AActor> It(World); It; ++It)
    {
        AActor* Actor = *It;
        if (!IsValid(Actor)) continue;
        if (!Actor->FindComponentByClass<UCognitiveMotionLearnerComponent>()) continue;

        auto Row = MakeShared<FCognitiveValidationRow>();
        Row->ActorName = Actor->GetName();

        TArray<FString> Errors;

        // Check required components
        if (!Actor->FindComponentByClass<UCognitivePoseRecorderComponent>())
            Errors.Add(TEXT("Missing UCognitivePoseRecorderComponent"));

        if (ACharacter* Char = Cast<ACharacter>(Actor))
        {
            USkeletalMeshComponent* Mesh = Char->GetMesh();
            if (!Mesh)
            {
                Errors.Add(TEXT("No Skeletal Mesh component"));
            }
            else
            {
                if (!Mesh->GetAnimClass())
                    Errors.Add(TEXT("No AnimBP assigned to Mesh"));
                else if (Mesh->GetAnimClass() &&
                         !Mesh->GetAnimClass()->IsChildOf(UCognitiveAnimInstance::StaticClass()))
                    Errors.Add(TEXT("AnimBP parent is not UCognitiveAnimInstance"));
            }
        }

        Row->bPassed = Errors.IsEmpty();
        Row->Summary = Row->bPassed
            ? TEXT("All checks passed")
            : FString::Join(Errors, TEXT("; "));

        if (Row->bPassed) ++Passed; else ++Failed;
        ValidationRows.Add(Row);
    }

    if (ValidationRows.IsEmpty())
    {
        auto EmptyRow = MakeShared<FCognitiveValidationRow>();
        EmptyRow->ActorName = TEXT("—");
        EmptyRow->bPassed   = false;
        EmptyRow->Summary   = TEXT(
            "Nenhum NPC com CognitiveMotionLearnerComponent encontrado no level. "
            "Volte ao Step 4, clique em 'Criar BP_CognitiveNPC' e depois "
            "'Aplicar Componentes ao Ator Selecionado'.");
        ValidationRows.Add(EmptyRow);
    }

    AddLog(FString::Printf(TEXT("Validation complete: %d passed, %d failed, %d total"),
        Passed, Failed, ValidationRows.Num()));
    return Failed == 0;
}

bool SCognitiveSetupWizard::DoApplyToSelectedActor()
{
    if (!GEditor) { AddLog(TEXT("ERROR: GEditor not available.")); return false; }

    USelection* Selection = GEditor->GetSelectedActors();
    if (!Selection || Selection->Num() == 0)
    {
        AddLog(TEXT("ERRO: Nenhum ator selecionado.\n"
                    "Use o botão '1. Criar BP_CognitiveNPC' acima, ou selecione "
                    "um Character no viewport e clique novamente."));
        return false;
    }

    AActor* Actor = Cast<AActor>(Selection->GetSelectedObject(0));
    if (!Actor)
    {
        AddLog(TEXT("ERRO: O objeto selecionado não é um Actor."));
        return false;
    }

    // ── Tenta obter o Blueprint do ator selecionado ───────────────────────────
    UBlueprint* BP = nullptr;
    if (UClass* ActorClass = Actor->GetClass())
    {
        BP = Cast<UBlueprint>(ActorClass->ClassGeneratedBy);
    }

    if (BP)
    {
        // Ator é instância de um Blueprint — adiciona ao Blueprint Class via SCS
        AddComponentToBlueprint(BP, UCognitiveMotionLearnerComponent::StaticClass(),
            FName("CognitiveMotionLearner"));
        AddComponentToBlueprint(BP, UCognitivePoseRecorderComponent::StaticClass(),
            FName("CognitivePoseRecorder"));
        AddComponentToBlueprint(BP, UCognitiveNPCBoneDriver::StaticClass(),
            FName("CognitiveNPCBoneDriver"));
        AddComponentToBlueprint(BP, UCognitiveLeaderObserverComponent::StaticClass(),
            FName("CognitiveLeaderObserver"));
        AddComponentToBlueprint(BP, UCognitiveNPCStateMachine::StaticClass(),
            FName("CognitiveNPCStateMachine"));

        AddComponentToBlueprint(BP, UCognitiveSkeletonManagerComponent::StaticClass(),
            FName("CognitiveSkeletonManager"));

        AddComponentToBlueprint(BP, UCognitiveWorldPerceptionComponent::StaticClass(),
            FName("CognitiveWorldPerception"));

        // Seta AnimClass no SkeletalMeshComponent do Blueprint
        if (GeneratedAnimBP && GeneratedAnimBP->GeneratedClass)
        {
            for (USCS_Node* Node : BP->SimpleConstructionScript->GetAllNodes())
            {
                if (!Node->ComponentClass) continue;
                if (!Node->ComponentClass->IsChildOf(USkeletalMeshComponent::StaticClass())) continue;
                USkeletalMeshComponent* MeshT =
                    Cast<USkeletalMeshComponent>(Node->ComponentTemplate);
                if (MeshT)
                {
                    MeshT->SetAnimInstanceClass(GeneratedAnimBP->GeneratedClass);
                    AddLog(FString::Printf(TEXT("AnimClass setado: %s"),
                        *GeneratedAnimBP->GetName()));
                }
            }
        }

        FBlueprintEditorUtils::MarkBlueprintAsModified(BP);
        FKismetEditorUtilities::CompileBlueprint(BP);
        BP->MarkPackageDirty();

        AddLog(FString::Printf(TEXT("Blueprint '%s' atualizado com todos os componentes."),
            *BP->GetName()));
    }
    else
    {
        // Ator não é Blueprint — adiciona como Instance Components (fallback)
        AddLog(TEXT("AVISO: Ator não é instância de um Blueprint. "
                    "Componentes adicionados como Instance Components (não persistem no Blueprint)."));

        auto AddIfMissing = [&](TSubclassOf<UActorComponent> Class, FName Name)
        {
            if (!Actor->FindComponentByClass(Class))
            {
                UActorComponent* Comp = NewObject<UActorComponent>(Actor, Class, Name);
                Actor->AddInstanceComponent(Comp);
                Comp->RegisterComponent();
                AddLog(FString::Printf(TEXT("Adicionado: %s"), *Name.ToString()));
            }
        };
        AddIfMissing(UCognitiveMotionLearnerComponent::StaticClass(),  FName("CognitiveMotionLearner"));
        AddIfMissing(UCognitivePoseRecorderComponent::StaticClass(),   FName("CognitivePoseRecorder"));
        AddIfMissing(UCognitiveNPCBoneDriver::StaticClass(),           FName("CognitiveNPCBoneDriver"));
        AddIfMissing(UCognitiveLeaderObserverComponent::StaticClass(), FName("CognitiveLeaderObserver"));
        AddIfMissing(UCognitiveNPCStateMachine::StaticClass(),         FName("CognitiveNPCStateMachine"));

        Actor->Modify();
    }

    GEditor->RedrawAllViewports();
    return true;
}

#undef LOCTEXT_NAMESPACE
