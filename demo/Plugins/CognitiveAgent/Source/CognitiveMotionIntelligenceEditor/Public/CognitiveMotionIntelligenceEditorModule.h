#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleManager.h"
#include "Widgets/SWidget.h"
#include "Framework/Docking/TabManager.h"

class FCognitiveSetupWizardTab;

class FCognitiveMotionIntelligenceEditorModule : public IModuleInterface
{
public:
    static const FName SetupWizardTabName;
    static const FName DebugDashboardTabName;

    virtual void StartupModule() override;
    virtual void ShutdownModule() override;

private:
    void RegisterMenus();
    void RegisterDetailsCustomizations();
    void UnregisterDetailsCustomizations();

    void OnOpenSetupWizard();
    void OnOpenDebugDashboard();
    void OnValidateScene();
    void OnGenerateAnimBP();

    TSharedRef<SDockTab> OnSpawnSetupWizardTab(const FSpawnTabArgs& SpawnTabArgs);
    TSharedRef<SDockTab> OnSpawnDebugDashboardTab(const FSpawnTabArgs& SpawnTabArgs);

    TSharedPtr<FTabManager::FLayout> PersistentLayout;
};
