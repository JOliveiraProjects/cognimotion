using System;
using System.IO;
using UnrealBuildTool;

// ─────────────────────────────────────────────────────────────────────────────
// LibTorch — módulo ThirdParty
//
// Vincula a LibTorch (PyTorch C++) ao plugin para inferência nativa do modelo
// TorchScript, SEM rede e SEM Python em runtime.
//
// INSTALAÇÃO (uma vez):
//   1. Baixe a LibTorch para Windows que CASE com a versão do treino.
//      Treino usa torch 2.12 + CUDA 13.0 → baixe "LibTorch 2.12 CUDA 12.x/13.x
//      Release" em https://pytorch.org/get-started/locally/
//      (use a versão Release, não Debug — o UE compila em Development/Shipping).
//   2. Extraia para:  <Plugin>/Source/ThirdParty/LibTorch/
//      Estrutura esperada:
//        ThirdParty/LibTorch/include/        (headers: torch/, c10/, ATen/...)
//        ThirdParty/LibTorch/lib/            (.lib e .dll)
//   3. Recompile o plugin.
//
// Se a pasta não existir, o módulo compila vazio e o plugin roda em modo
// "sem LibTorch" (inferência nativa desabilitada — usa o caminho TCP/Python
// como fallback, se presente).
// ─────────────────────────────────────────────────────────────────────────────
public class LibTorch : ModuleRules
{
    public LibTorch(ReadOnlyTargetRules Target) : base(Target)
    {
        Type = ModuleType.External;

        string LibTorchRoot = Path.Combine(ModuleDirectory, "LibTorch");
        string IncludePath  = Path.Combine(LibTorchRoot, "include");
        string LibPath      = Path.Combine(LibTorchRoot, "lib");

        bool bHasLibTorch = Directory.Exists(IncludePath) && Directory.Exists(LibPath);

        if (!bHasLibTorch)
        {
            // LibTorch não instalada — define macro p/ o código C++ se adaptar.
            PublicDefinitions.Add("WITH_LIBTORCH=0");
            Console.WriteLine("[CognitiveAgent] LibTorch NÃO encontrada em " + LibTorchRoot
                + " — inferência nativa desabilitada (WITH_LIBTORCH=0).");
            return;
        }

        PublicDefinitions.Add("WITH_LIBTORCH=1");

        // Headers da LibTorch + torch/csrc/api/include (frontend C++)
        PublicSystemIncludePaths.Add(IncludePath);
        PublicSystemIncludePaths.Add(Path.Combine(IncludePath, "torch", "csrc", "api", "include"));

        if (Target.Platform == UnrealTargetPlatform.Win64)
        {
            // Vincula todas as .lib presentes (torch, torch_cpu, c10, torch_cuda, c10_cuda...)
            foreach (string Lib in Directory.GetFiles(LibPath, "*.lib"))
            {
                PublicAdditionalLibraries.Add(Lib);
            }

            // As DLLs são copiadas para junto do executável (staging). NÃO usar
            // delay-load: a c10.dll exporta SÍMBOLOS DE DADOS (não só funções),
            // e o linker do MSVC não consegue fazer delay-load nesse caso
            // (erro LNK1194). Sem delay-load, o Windows carrega as DLLs no
            // startup — que é o comportamento correto aqui, pois elas estão
            // ao lado do binário.
            foreach (string Dll in Directory.GetFiles(LibPath, "*.dll"))
            {
                string DllName = Path.GetFileName(Dll);
                RuntimeDependencies.Add(Path.Combine("$(BinaryOutputDir)", DllName), Dll);
            }

            // CRÍTICO: resolve o conflito "inconsistent dll linkage" entre LibTorch
            // e UE relatado nos fóruns. Estas definições alinham o modo de linkagem.
            PublicDefinitions.Add("NOMINMAX");                 // evita conflito min/max do Windows.h
            PublicDefinitions.Add("_CRT_SECURE_NO_WARNINGS=1");

            // ATENÇÃO — CAUSA RAIZ DO ERRO 'gflags/gflags.h not found':
            // O c10/util/Flags.h testa estas macros com #ifdef (existência),
            // NÃO com #if (valor). Portanto definir C10_USE_GFLAGS=0 AINDA
            // dispara o #include <gflags/gflags.h> (a macro existe!).
            // A forma correta de DESLIGAR gflags/glog é NÃO definir as macros.
            // Por isso elas foram REMOVIDAS daqui. NÃO as adicione de volta com
            // valor (=0) — isso religa o include e quebra o build.
            //   ❌ ERRADO: PublicDefinitions.Add("C10_USE_GFLAGS=0");
            //   ✅ CERTO:  (não definir nada — ausência = desligado)

            // LibTorch usa RTTI e exceptions — UE desliga por padrão.
            bUseRTTI    = true;
            bEnableExceptions = true;
        }
    }
}
