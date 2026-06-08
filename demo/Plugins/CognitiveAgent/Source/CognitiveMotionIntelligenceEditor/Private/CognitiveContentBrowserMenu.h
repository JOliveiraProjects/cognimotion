#pragma once

#include "CoreMinimal.h"

/**
 * Extensão de menu do Content Browser para o Cognitive Motion.
 *
 * Adiciona um submenu "Cognitive Motion" ao clicar com o botão direito numa
 * pasta/área vazia do Content Browser, com a ação:
 *   - "Create Cognitive NPC Blueprint" — cria um Blueprint herdando de ACharacter
 *     já com TODOS os componentes do plugin (Learner, PoseRecorder, BoneDriver,
 *     LeaderObserver, StateMachine, SkeletonManager) adicionados via SCS.
 *
 * É registrada no StartupModule do editor e removida no ShutdownModule.
 */
class FCognitiveContentBrowserMenu
{
public:
    // Registra a extensão de menu no Content Browser. Idempotente.
    static void Register();

    // Remove a extensão (chamado no shutdown do módulo).
    static void Unregister();

private:
    // Cria o BP_CognitiveNPC completo no caminho destino informado.
    // Retorna o objeto criado (ou nullptr em falha) e loga no Output Log.
    static class UBlueprint* CreateCognitiveNPCBlueprint(const FString& TargetPath);

    // Resolve a pasta destino a partir do contexto do menu do Content Browser.
    static FString ResolveTargetFolder();
};
