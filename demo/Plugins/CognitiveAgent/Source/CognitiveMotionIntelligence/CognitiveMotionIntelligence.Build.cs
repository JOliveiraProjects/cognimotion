using System;
using System.IO;
using UnrealBuildTool;

public class CognitiveMotionIntelligence : ModuleRules
{
    public CognitiveMotionIntelligence(ReadOnlyTargetRules Target) : base(Target)
    {
        // IMPORTANTE: NÃO usar PCH compartilhado. O PCH compartilhado (com RTTI)
        // arrasta os headers da LibTorch para outros módulos (ex.: o módulo de
        // jogo CMI), que não têm as macros C10_USE_GFLAGS=0 — causando o erro
        // 'Cannot open include file: gflags/gflags.h' ao linkar esses módulos.
        // Com PCH próprio, a LibTorch fica isolada neste módulo.
        PCHUsage = ModuleRules.PCHUsageMode.NoSharedPCHs;
        PrivatePCHHeaderFile = "Private/CognitiveMotionIntelligencePCH.h";
        PublicDependencyModuleNames.AddRange(new string[] {
            "Core", "CoreUObject", "Engine", "InputCore",
            "AnimGraphRuntime", "GameplayTags",
            "Sockets", "Networking"
        });
        PrivateDependencyModuleNames.AddRange(new string[] {
            "AnimationCore", "SkeletalMeshDescription",
            "Projects"   // IPluginManager — localizar o modelo em Content/Models
        });

        // LibTorch — inferência nativa (sem rede, sem Python). Se a lib não
        // estiver instalada, o módulo compila com WITH_LIBTORCH=0 e o plugin
        // usa o caminho TCP/Python como fallback.
        PrivateDependencyModuleNames.Add("LibTorch");

        // Este módulo INCLUI os headers da LibTorch (torch/script.h) quando ela
        // está presente. RTTI e exceptions são necessários. As macros gflags/glog
        // NÃO são definidas aqui de propósito: o c10/util/Flags.h usa #ifdef
        // (testa existência, não valor), então defini-las como 0 RELIGA o include
        // de <gflags/gflags.h> e quebra o build. Ausência = desligado.
        bUseRTTI = true;
        bEnableExceptions = true;
        PublicDefinitions.Add("NOMINMAX");
        PublicDefinitions.Add("_CRT_SECURE_NO_WARNINGS=1");
    }
}
