using System;
using System.IO;
using UnrealBuildTool;

public class CognitiveMotionIntelligence : ModuleRules
{
    public CognitiveMotionIntelligence(ReadOnlyTargetRules Target) : base(Target)
    {
        // A LibTorch NÃO é mais linkada neste módulo. A inferência roda num
        // processo separado (cmi_worker.exe) que comunica por named pipe. Isso
        // elimina o conflito de heap (0xC0000374) entre o Mimalloc do Unreal e
        // o alocador da LibTorch, que derrubava o editor no primeiro forward.
        // Com a torch fora do processo do UE, este módulo volta a ser um módulo
        // UE comum (PCH compartilhado OK, sem RTTI/exceptions especiais).
        PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(new string[] {
            "Core", "CoreUObject", "Engine", "InputCore",
            "AnimGraphRuntime", "GameplayTags",
            "Sockets", "Networking"
        });
        PrivateDependencyModuleNames.AddRange(new string[] {
            "AnimationCore", "SkeletalMeshDescription",
            "Projects",   // IPluginManager — localizar o modelo e o worker
            "NavigationSystem", "AIModule"   // NavMesh + SimpleMoveToLocation
        });
    }
}
