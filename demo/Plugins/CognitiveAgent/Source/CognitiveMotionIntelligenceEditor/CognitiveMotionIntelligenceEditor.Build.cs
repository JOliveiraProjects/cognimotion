using UnrealBuildTool;

public class CognitiveMotionIntelligenceEditor : ModuleRules
{
    public CognitiveMotionIntelligenceEditor(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(new string[]
        {
            "Core",
            "CoreUObject",
            "Engine",
            "InputCore",
            "CognitiveMotionIntelligence",
            "AnimGraph",
            "AnimGraphRuntime",
            "BlueprintGraph",
            "UnrealEd",
            "AssetTools",
            "Kismet",
            "KismetCompiler",
        });

        PrivateDependencyModuleNames.AddRange(new string[]
        {
            "Slate",
            "SlateCore",
            "EditorFramework",
            "ToolMenus",
            "LevelEditor",
            "EditorStyle",
            "ContentBrowser",
            "ContentBrowserData",
            "WorkspaceMenuStructure",
            "Sockets",
            "UMG",
            "UMGEditor",
            "Blutility",
            "PropertyEditor",  // SObjectPropertyEntryBox para skeleton picker
        });
    }
}
