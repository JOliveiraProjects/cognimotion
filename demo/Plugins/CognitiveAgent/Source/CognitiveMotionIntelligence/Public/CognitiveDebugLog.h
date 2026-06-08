#pragma once

#include "CoreMinimal.h"
#include "Logging/LogMacros.h"

// ─────────────────────────────────────────────────────────────────────────────
// Categoria de log dedicada do Cognitive Motion Intelligence.
// Use CMI_LOG(...) em vez de UE_LOG(LogTemp, ...) para que o Debug Dashboard
// possa ligar/desligar toda a verbosidade do plugin com um único toggle.
// ─────────────────────────────────────────────────────────────────────────────
COGNITIVEMOTIONINTELLIGENCE_API DECLARE_LOG_CATEGORY_EXTERN(LogCognitiveMotion, Log, All);

/**
 * FCognitiveDebugState
 *
 * Estado global de debug do plugin, compartilhado entre o runtime
 * (BoneDriver, LearnerComponent, LeaderObserver, InferenceSubsystem) e o
 * módulo Editor (Debug Dashboard). Thread-safe via std::atomic.
 *
 * O Dashboard liga/desliga via SetDebugEnabled(); o runtime consulta com
 * IsDebugEnabled() antes de emitir logs verbosos.
 */
class COGNITIVEMOTIONINTELLIGENCE_API FCognitiveDebugState
{
public:
    static bool IsDebugEnabled()
    {
        return bDebugEnabled;
    }

    static void SetDebugEnabled(bool bEnabled)
    {
        bDebugEnabled = bEnabled;
    }

private:
    static TAtomic<bool> bDebugEnabled;
};

// Macro de conveniência: só formata/emite o log se o debug estiver ligado.
// Evita custo de formatação de string quando o debug está desativado.
#define CMI_DBG(Format, ...) \
    do { \
        if (FCognitiveDebugState::IsDebugEnabled()) \
        { \
            UE_LOG(LogCognitiveMotion, Log, TEXT(Format), ##__VA_ARGS__); \
        } \
    } while (0)
