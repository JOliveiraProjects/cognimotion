#include "CognitiveContentBrowserMenu.h"

// ── Componentes do plugin (adicionados ao Blueprint) ──────────────────────────
#include "CognitiveMotionLearnerComponent.h"
#include "CognitivePoseRecorderComponent.h"
#include "CognitiveNPCBoneDriver.h"
#include "CognitiveLeaderObserverComponent.h"
#include "CognitiveNPCStateMachine.h"
#include "CognitiveSkeletonManagerComponent.h"
#include "CognitiveWorldPerceptionComponent.h"

// ── Criação de Blueprint / SCS ────────────────────────────────────────────────
#include "Kismet2/KismetEditorUtilities.h"          // FKismetEditorUtilities::CreateBlueprint
#include "Kismet2/BlueprintEditorUtils.h"            // FBlueprintEditorUtils
#include "Engine/SimpleConstructionScript.h"         // USimpleConstructionScript
#include "Engine/SCS_Node.h"                         // USCS_Node
#include "Engine/Blueprint.h"                        // UBlueprint
#include "Engine/BlueprintGeneratedClass.h"          // UBlueprintGeneratedClass
#include "GameFramework/Character.h"                 // ACharacter
#include "UObject/Package.h"                         // CreatePackage, UPackage
#include "UObject/UObjectGlobals.h"                  // FindPackage, LoadObject
#include "Textures/SlateIcon.h"                       // FSlateIcon
#include "Framework/Commands/UIAction.h"              // FUIAction, FExecuteAction

// ── Menu / Content Browser ────────────────────────────────────────────────────
#include "ToolMenus.h"                               // UToolMenus
#include "ToolMenuOwner.h"                            // FToolMenuOwner(FName)
#include "ToolMenu.h"                                // UToolMenu
#include "ToolMenuSection.h"                          // FToolMenuSection
#include "ContentBrowserModule.h"                     // FContentBrowserModule
#include "IContentBrowserSingleton.h"                 // IContentBrowserSingleton

// ── Asset registry / notificação ──────────────────────────────────────────────
#include "AssetRegistry/AssetRegistryModule.h"        // FAssetRegistryModule
#include "Modules/ModuleManager.h"
#include "Misc/MessageDialog.h"
#include "Styling/AppStyle.h"
#include "Framework/Notifications/NotificationManager.h"
#include "Widgets/Notifications/SNotificationList.h"

#define LOCTEXT_NAMESPACE "CognitiveContentBrowserMenu"

static const FName CMI_ContentMenuOwner = FName("CognitiveContentBrowserMenu");

// Reaproveita o mesmo padrão de adição de componente via SCS usado no wizard.
static USCS_Node* CMI_AddComponent(UBlueprint* BP, UClass* CompClass, FName CompName)
{
    if (!BP || !BP->SimpleConstructionScript || !CompClass) return nullptr;

    // Evita duplicar se já existe um nó com esse nome.
    for (USCS_Node* Node : BP->SimpleConstructionScript->GetAllNodes())
    {
        if (Node && Node->GetVariableName() == CompName) return Node;
    }

    USCS_Node* NewNode = BP->SimpleConstructionScript->CreateNode(CompClass, CompName);
    if (NewNode)
    {
        BP->SimpleConstructionScript->AddNode(NewNode);
    }
    return NewNode;
}

// ─────────────────────────────────────────────────────────────────────────────
void FCognitiveContentBrowserMenu::Register()
{
    if (!UToolMenus::Get()) return;

    FToolMenuOwnerScoped OwnerScoped(CMI_ContentMenuOwner);

    // Menu de contexto da área de "adicionar novo" do Content Browser.
    UToolMenu* Menu = UToolMenus::Get()->ExtendMenu("ContentBrowser.AddNewContextMenu");
    if (!Menu) return;

    FToolMenuSection& Section = Menu->FindOrAddSection(
        "CognitiveMotion",
        LOCTEXT("CMISectionLabel", "Cognitive Motion"));

    Section.AddMenuEntry(
        "CreateCognitiveNPC",
        LOCTEXT("CreateNPCLabel", "Cognitive NPC Blueprint"),
        LOCTEXT("CreateNPCTip",
            "Cria um Blueprint herdando de Character já com todos os componentes "
            "do Cognitive Motion (Learner, PoseRecorder, BoneDriver, LeaderObserver, "
            "StateMachine, SkeletonManager)."),
        FSlateIcon(FAppStyle::GetAppStyleSetName(), "ClassIcon.Character"),
        FUIAction(FExecuteAction::CreateLambda([]()
        {
            const FString Folder = ResolveTargetFolder();
            UBlueprint* BP = CreateCognitiveNPCBlueprint(Folder);
            if (BP)
            {
                FNotificationInfo Info(LOCTEXT("CreatedNPC",
                    "Cognitive NPC Blueprint criado com todos os componentes."));
                Info.ExpireDuration = 4.0f;
                FSlateNotificationManager::Get().AddNotification(Info);
            }
            else
            {
                FMessageDialog::Open(EAppMsgType::Ok,
                    LOCTEXT("CreateNPCFail",
                        "Não foi possível criar o Blueprint. Veja o Output Log."));
            }
        })));
}

// ─────────────────────────────────────────────────────────────────────────────
void FCognitiveContentBrowserMenu::Unregister()
{
    if (UObjectInitialized() && UToolMenus::Get())
    {
        UToolMenus::Get()->UnregisterOwnerByName(CMI_ContentMenuOwner);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
FString FCognitiveContentBrowserMenu::ResolveTargetFolder()
{
    // Tenta usar a pasta atualmente selecionada no Content Browser; se falhar,
    // cai num caminho padrão do plugin.
    FString Result = TEXT("/Game/CognitiveMotion");

    if (FModuleManager::Get().IsModuleLoaded("ContentBrowser"))
    {
        FContentBrowserModule& CBModule =
            FModuleManager::GetModuleChecked<FContentBrowserModule>("ContentBrowser");
        IContentBrowserSingleton& CB = CBModule.Get();

        TArray<FString> SelectedPaths;
        CB.GetSelectedPathViewFolders(SelectedPaths);
        if (SelectedPaths.Num() > 0 && !SelectedPaths[0].IsEmpty())
        {
            Result = SelectedPaths[0];
        }
    }
    return Result;
}

// ─────────────────────────────────────────────────────────────────────────────
UBlueprint* FCognitiveContentBrowserMenu::CreateCognitiveNPCBlueprint(const FString& TargetPath)
{
    // Garante um nome único no caminho destino.
    FString BaseName = TEXT("BP_CognitiveNPC");
    FString PackagePath = FString::Printf(TEXT("%s/%s"), *TargetPath, *BaseName);

    // Se já existir, adiciona sufixo numérico.
    int32 Suffix = 1;
    while (FindPackage(nullptr, *PackagePath) != nullptr ||
           LoadObject<UBlueprint>(nullptr, *(PackagePath + TEXT(".") + BaseName)) != nullptr)
    {
        BaseName = FString::Printf(TEXT("BP_CognitiveNPC_%d"), Suffix++);
        PackagePath = FString::Printf(TEXT("%s/%s"), *TargetPath, *BaseName);
        if (Suffix > 1000) break;  // guarda contra loop infinito
    }

    UPackage* Package = CreatePackage(*PackagePath);
    if (!Package)
    {
        UE_LOG(LogTemp, Error,
            TEXT("[CMI] CreatePackage falhou para %s"), *PackagePath);
        return nullptr;
    }

    UBlueprint* BP = FKismetEditorUtilities::CreateBlueprint(
        ACharacter::StaticClass(), Package,
        FName(*BaseName), BPTYPE_Normal,
        UBlueprint::StaticClass(), UBlueprintGeneratedClass::StaticClass(),
        FName("CognitiveContentBrowser"));

    if (!BP)
    {
        UE_LOG(LogTemp, Error, TEXT("[CMI] CreateBlueprint falhou."));
        return nullptr;
    }

    // Adiciona todos os componentes do plugin (mesma composição do wizard).
    CMI_AddComponent(BP, UCognitiveMotionLearnerComponent::StaticClass(),
        FName("CognitiveMotionLearner"));
    CMI_AddComponent(BP, UCognitivePoseRecorderComponent::StaticClass(),
        FName("CognitivePoseRecorder"));
    CMI_AddComponent(BP, UCognitiveNPCBoneDriver::StaticClass(),
        FName("CognitiveNPCBoneDriver"));
    CMI_AddComponent(BP, UCognitiveLeaderObserverComponent::StaticClass(),
        FName("CognitiveLeaderObserver"));
    CMI_AddComponent(BP, UCognitiveNPCStateMachine::StaticClass(),
        FName("CognitiveNPCStateMachine"));
    CMI_AddComponent(BP, UCognitiveSkeletonManagerComponent::StaticClass(),
        FName("CognitiveSkeletonManager"));
    CMI_AddComponent(BP, UCognitiveWorldPerceptionComponent::StaticClass(),
        FName("CognitiveWorldPerception"));

    FBlueprintEditorUtils::MarkBlueprintAsStructurallyModified(BP);
    FKismetEditorUtilities::CompileBlueprint(BP);

    // Registra o novo asset e marca o package como sujo (para salvar).
    FAssetRegistryModule::AssetCreated(BP);
    Package->MarkPackageDirty();

    UE_LOG(LogTemp, Log,
        TEXT("[CMI] Cognitive NPC Blueprint criado: %s (7 componentes)"), *PackagePath);
    return BP;
}

#undef LOCTEXT_NAMESPACE
