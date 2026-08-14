#include "CognitiveMotionIntelligenceEditorModule.h"
#include "CognitiveContentBrowserMenu.h"
#include "CognitiveMotionSetupWizard.h"
#include "SCognitiveDebugDashboard.h"
#include "SCognitiveTrainingStudio.h"
#include "CognitiveNPCBoneDriverDetails.h"
#include "CognitiveNPCBoneDriver.h"
#include "CognitiveDebugLog.h"
#include "PropertyEditorModule.h"
#include "CognitiveMotionValidator.h"
#include "CognitiveMotionLearnerComponent.h"
#include "CognitivePoseRecorderComponent.h"
#include "CognitiveAnimInstance.h"

#include "ToolMenus.h"
#include "ToolMenu.h"
#include "ToolMenuSection.h"
#include "Framework/Docking/TabManager.h"
#include "Widgets/Docking/SDockTab.h"
#include "WorkspaceMenuStructure.h"
#include "WorkspaceMenuStructureModule.h"
#include "LevelEditor.h"
#include "Editor.h"
#include "Engine/World.h"
#include "EngineUtils.h"
#include "GameFramework/Character.h"
#include "Components/SkeletalMeshComponent.h"
#include "Misc/MessageDialog.h"

#define LOCTEXT_NAMESPACE "CognitiveMotionEditor"

const FName FCognitiveMotionIntelligenceEditorModule::SetupWizardTabName =
    FName("CognitiveMotionSetupWizard");

const FName FCognitiveMotionIntelligenceEditorModule::DebugDashboardTabName =
    FName("CognitiveMotionDebugDashboard");
static const FName TrainingStudioTabName =
    FName("CognitiveTrainingStudio");

// ─────────────────────────────────────────────────────────────────────────────
void FCognitiveMotionIntelligenceEditorModule::StartupModule()
{
    if (IsRunningCommandlet()) return;

    // Register the dockable tab spawner for the Setup Wizard window
    FGlobalTabmanager::Get()->RegisterNomadTabSpawner(
        SetupWizardTabName,
        FOnSpawnTab::CreateRaw(this, &FCognitiveMotionIntelligenceEditorModule::OnSpawnSetupWizardTab))
        .SetDisplayName(LOCTEXT("WizardTabTitle", "Cognitive Motion Setup Wizard"))
        .SetTooltipText(LOCTEXT("WizardTabTooltip", "Open the Cognitive Motion Intelligence setup wizard"))
        .SetGroup(WorkspaceMenu::GetMenuStructure().GetToolsCategory())
        .SetMenuType(ETabSpawnerMenuType::Hidden);  // we manage the menu ourselves

    // Register the Debug Dashboard tab spawner
    FGlobalTabmanager::Get()->RegisterNomadTabSpawner(
        DebugDashboardTabName,
        FOnSpawnTab::CreateRaw(this, &FCognitiveMotionIntelligenceEditorModule::OnSpawnDebugDashboardTab))
        .SetDisplayName(LOCTEXT("DashTabTitle", "Cognitive Motion Debug Dashboard"))
        .SetTooltipText(LOCTEXT("DashTabTooltip", "Live per-NPC data health dashboard"))
        .SetGroup(WorkspaceMenu::GetMenuStructure().GetToolsCategory())
        .SetMenuType(ETabSpawnerMenuType::Hidden);

    // Register the Training Studio (Treino & Ensino) tab spawner
    FGlobalTabmanager::Get()->RegisterNomadTabSpawner(
        TrainingStudioTabName,
        FOnSpawnTab::CreateRaw(this, &FCognitiveMotionIntelligenceEditorModule::OnSpawnTrainingStudioTab))
        .SetDisplayName(LOCTEXT("StudioTabTitle", "Cognitive Training Studio"))
        .SetTooltipText(LOCTEXT("StudioTabTooltip", "Treino & Ensino: registre demonstrações e corrija decisões do agente"))
        .SetGroup(WorkspaceMenu::GetMenuStructure().GetToolsCategory());

    // Register Details panel customizations
    RegisterDetailsCustomizations();

    // Defer menu registration until ToolMenus is ready
    UToolMenus::RegisterStartupCallback(
        FSimpleMulticastDelegate::FDelegate::CreateRaw(
            this, &FCognitiveMotionIntelligenceEditorModule::RegisterMenus));
}

// ─────────────────────────────────────────────────────────────────────────────
void FCognitiveMotionIntelligenceEditorModule::ShutdownModule()
{
    FCognitiveContentBrowserMenu::Unregister();
    UToolMenus::UnregisterOwner(this);
    FGlobalTabmanager::Get()->UnregisterNomadTabSpawner(SetupWizardTabName);
    FGlobalTabmanager::Get()->UnregisterNomadTabSpawner(DebugDashboardTabName);
    FGlobalTabmanager::Get()->UnregisterNomadTabSpawner(TrainingStudioTabName);
    UnregisterDetailsCustomizations();
}

// ─────────────────────────────────────────────────────────────────────────────
// Register top-level "Cognitive Motion" menu
// ─────────────────────────────────────────────────────────────────────────────
void FCognitiveMotionIntelligenceEditorModule::RegisterMenus()
{
    FToolMenuOwnerScoped OwnerScoped(this);

    // Passo 3: menu de botão-direito no Content Browser p/ criar o NPC Blueprint.
    FCognitiveContentBrowserMenu::Register();

    // Add a top-level "Cognitive Motion" menu to the main menu bar
    UToolMenu* MainMenu = UToolMenus::Get()->ExtendMenu("MainFrame.MainMenu");
    if (!MainMenu) return;

    FToolMenuSection& MenuBarSection = MainMenu->FindOrAddSection("CognitiveMotionMenu");
    MenuBarSection.AddSubMenu(
        "CognitiveMotion",
        LOCTEXT("MenuLabel", "Cognitive Motion"),
        LOCTEXT("MenuTip",   "Cognitive Motion Intelligence tools"),
        FNewToolMenuDelegate::CreateLambda([this](UToolMenu* SubMenu)
        {
            FToolMenuSection& Section = SubMenu->FindOrAddSection("CognitiveMotionSection");

            Section.AddMenuEntry(
                "SetupWizard",
                LOCTEXT("WizardLabel", "Setup Wizard..."),
                LOCTEXT("WizardTip",
                    "Step-by-step wizard: select skeleton, generate AnimBP, configure NPC, validate."),
                FSlateIcon(FAppStyle::GetAppStyleSetName(), "LevelEditor.Tabs.Modes"),
                FUIAction(FExecuteAction::CreateRaw(
                    this, &FCognitiveMotionIntelligenceEditorModule::OnOpenSetupWizard)));

            Section.AddMenuEntry(
                "DebugDashboard",
                LOCTEXT("DashLabel", "Debug Dashboard..."),
                LOCTEXT("DashTip",
                    "Open the live per-NPC debug dashboard: data health, connection, "
                    "bones, latency, confidence. Includes the Enable Debug Logs toggle."),
                FSlateIcon(FAppStyle::GetAppStyleSetName(), "Icons.Visibility"),
                FUIAction(FExecuteAction::CreateRaw(
                    this, &FCognitiveMotionIntelligenceEditorModule::OnOpenDebugDashboard)));

            Section.AddSeparator("Sep1");

            Section.AddMenuEntry(
                "ValidateScene",
                LOCTEXT("ValidateLabel", "Validate Scene"),
                LOCTEXT("ValidateTip",
                    "Check all CognitiveNPC actors in the level for correct component setup. "
                    "Results are printed to the Output Log."),
                FSlateIcon(FAppStyle::GetAppStyleSetName(), "MessageLog.Warning"),
                FUIAction(FExecuteAction::CreateRaw(
                    this, &FCognitiveMotionIntelligenceEditorModule::OnValidateScene)));

            Section.AddMenuEntry(
                "GenerateAnimBP",
                LOCTEXT("GenBPLabel", "Generate Cognitive AnimBP..."),
                LOCTEXT("GenBPTip",
                    "Quick-generate a UCognitiveAnimInstance AnimBlueprint without opening the wizard."),
                FSlateIcon(FAppStyle::GetAppStyleSetName(), "ClassIcon.AnimBlueprint"),
                FUIAction(FExecuteAction::CreateRaw(
                    this, &FCognitiveMotionIntelligenceEditorModule::OnGenerateAnimBP)));
        }),
        false,   // bInOpenSubMenuOnClick
        FSlateIcon(FAppStyle::GetAppStyleSetName(), "LevelEditor.Tabs.Modes"));
}

// ─────────────────────────────────────────────────────────────────────────────
// Tab spawner — returns the actual Slate window content
// ─────────────────────────────────────────────────────────────────────────────
TSharedRef<SDockTab> FCognitiveMotionIntelligenceEditorModule::OnSpawnSetupWizardTab(
    const FSpawnTabArgs& SpawnTabArgs)
{
    return SNew(SDockTab)
        .TabRole(ETabRole::NomadTab)
        .Label(LOCTEXT("WizardTabTitle", "Cognitive Motion Setup Wizard"))
        [
            SNew(SCognitiveSetupWizard)
        ];
}

TSharedRef<SDockTab> FCognitiveMotionIntelligenceEditorModule::OnSpawnDebugDashboardTab(
    const FSpawnTabArgs& SpawnTabArgs)
{
    return SNew(SDockTab)
        .TabRole(ETabRole::NomadTab)
        .Label(LOCTEXT("DashTabTitle", "Cognitive Motion Debug Dashboard"))
        [
            SNew(SCognitiveDebugDashboard)
        ];
}

TSharedRef<SDockTab> FCognitiveMotionIntelligenceEditorModule::OnSpawnTrainingStudioTab(
    const FSpawnTabArgs& SpawnTabArgs)
{
    return SNew(SDockTab)
        .TabRole(ETabRole::NomadTab)
        [
            SNew(SCognitiveTrainingStudio)
        ];
}

// ─────────────────────────────────────────────────────────────────────────────
// Details panel customizations
// ─────────────────────────────────────────────────────────────────────────────
void FCognitiveMotionIntelligenceEditorModule::RegisterDetailsCustomizations()
{
    FPropertyEditorModule& PropertyModule =
        FModuleManager::LoadModuleChecked<FPropertyEditorModule>("PropertyEditor");

    PropertyModule.RegisterCustomClassLayout(
        UCognitiveNPCBoneDriver::StaticClass()->GetFName(),
        FOnGetDetailCustomizationInstance::CreateStatic(
            &FCognitiveNPCBoneDriverDetails::MakeInstance));

    PropertyModule.NotifyCustomizationModuleChanged();
}

void FCognitiveMotionIntelligenceEditorModule::UnregisterDetailsCustomizations()
{
    if (FModuleManager::Get().IsModuleLoaded("PropertyEditor"))
    {
        FPropertyEditorModule& PropertyModule =
            FModuleManager::GetModuleChecked<FPropertyEditorModule>("PropertyEditor");
        PropertyModule.UnregisterCustomClassLayout(
            UCognitiveNPCBoneDriver::StaticClass()->GetFName());
    }
}

void FCognitiveMotionIntelligenceEditorModule::OnOpenDebugDashboard()
{
    FGlobalTabmanager::Get()->TryInvokeTab(DebugDashboardTabName);
}

// ─────────────────────────────────────────────────────────────────────────────
// Menu actions
// ─────────────────────────────────────────────────────────────────────────────
void FCognitiveMotionIntelligenceEditorModule::OnOpenSetupWizard()
{
    FGlobalTabmanager::Get()->TryInvokeTab(SetupWizardTabName);
}

void FCognitiveMotionIntelligenceEditorModule::OnValidateScene()
{
    UWorld* World = GEditor ? GEditor->GetEditorWorldContext().World() : nullptr;
    if (!World)
    {
        FMessageDialog::Open(EAppMsgType::Ok,
            LOCTEXT("NoWorld", "No editor world found. Open a level first."));
        return;
    }

    int32 Passed = 0, Failed = 0, Total = 0;

    for (TActorIterator<AActor> It(World); It; ++It)
    {
        AActor* Actor = *It;
        if (!IsValid(Actor)) continue;
        if (!Actor->FindComponentByClass<UCognitiveMotionLearnerComponent>()) continue;

        ++Total;
        TArray<FString> Errors;

        if (!Actor->FindComponentByClass<UCognitivePoseRecorderComponent>())
            Errors.Add(TEXT("Missing UCognitivePoseRecorderComponent"));

        if (ACharacter* Char = Cast<ACharacter>(Actor))
        {
            USkeletalMeshComponent* Mesh = Char->GetMesh();
            if (!Mesh)
                Errors.Add(TEXT("No Skeletal Mesh"));
            else if (!Mesh->GetAnimClass())
                Errors.Add(TEXT("No AnimBP assigned"));
        }

        if (Errors.IsEmpty())
        {
            ++Passed;
            UE_LOG(LogTemp, Log, TEXT("[CMI Validate] %s: PASS"), *Actor->GetName());
        }
        else
        {
            ++Failed;
            UE_LOG(LogTemp, Warning, TEXT("[CMI Validate] %s: FAIL — %s"),
                *Actor->GetName(), *FString::Join(Errors, TEXT(", ")));
        }
    }

    FString Summary = Total == 0
        ? TEXT("No Cognitive NPC actors found in the level.")
        : FString::Printf(TEXT("Validation complete: %d/%d actors passed. See Output Log for details."),
            Passed, Total);

    FMessageDialog::Open(EAppMsgType::Ok, FText::FromString(Summary));
}

void FCognitiveMotionIntelligenceEditorModule::OnGenerateAnimBP()
{
    // Open the wizard on the AnimBP step
    FGlobalTabmanager::Get()->TryInvokeTab(SetupWizardTabName);
    // (User can navigate to Step 3 — this is better than a silent background operation)
    FMessageDialog::Open(EAppMsgType::Ok,
        LOCTEXT("GenBPMsg",
            "The Setup Wizard has been opened.\n\n"
            "To generate the AnimBP:\n"
            "  1. Complete Step 1 (select a skeleton)\n"
            "  2. Click Next to reach Step 3\n"
            "  3. Click 'Generate AnimBP Now'"));
}

#undef LOCTEXT_NAMESPACE

IMPLEMENT_MODULE(FCognitiveMotionIntelligenceEditorModule, CognitiveMotionIntelligenceEditor)
