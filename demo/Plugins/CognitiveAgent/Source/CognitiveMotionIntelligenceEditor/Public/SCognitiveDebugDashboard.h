#pragma once

#include "CoreMinimal.h"
#include "Widgets/SCompoundWidget.h"
#include "Containers/Ticker.h"

class SVerticalBox;
class SScrollBox;
class UCognitiveNPCBoneDriver;

/**
 * SCognitiveDebugDashboard
 *
 * Painel Slate (estilo nativo Unreal Engine) que mostra, em tempo real, a saúde
 * de dados de cada NPC Cognitive na cena: estado de observação, modo de
 * comportamento, conexão com o Python, bones aplicados, requests/responses,
 * latência e confiança. Se houver múltiplos NPCs, cada um aparece em seu próprio
 * cartão.
 *
 * Inclui um toggle "Enable Debug Logs" que liga/desliga FCognitiveDebugState,
 * controlando toda a verbosidade do plugin (categoria LogCognitiveMotion).
 *
 * Atualiza automaticamente via FTSTicker (intervalo configurável).
 */
class SCognitiveDebugDashboard : public SCompoundWidget
{
public:
    SLATE_BEGIN_ARGS(SCognitiveDebugDashboard) {}
    SLATE_END_ARGS()

    void Construct(const FArguments& InArgs);
    virtual ~SCognitiveDebugDashboard() override;

private:
    // Refresh
    bool        OnTick(float DeltaTime);
    void        RebuildNPCList();
    void        CollectDrivers(TArray<TWeakObjectPtr<UCognitiveNPCBoneDriver>>& Out) const;

    // Card builder por NPC
    TSharedRef<class SWidget> BuildNPCCard(UCognitiveNPCBoneDriver* Driver) const;

    // Seção de inferência neural (.pt): ação prevista, confiança, estado latente.
    TSharedRef<class SWidget> BuildNeuralSection(
        bool bHasNative, bool bLoaded, int32 ActionIdx, float Confidence,
        float HiddenNorm, float StochNorm, float InferenceMs) const;

    // Toggle de debug logs
    ECheckBoxState  IsDebugEnabled() const;
    void            OnDebugToggleChanged(ECheckBoxState NewState);

    // Header status (quantos NPCs, mundo ativo)
    FText           GetHeaderSummary() const;

    // Helpers de cor (verde/amarelo/vermelho) por saúde do dado
    static FSlateColor HealthColor(bool bGood, bool bWarn = false);

private:
    TSharedPtr<SVerticalBox>  NPCListBox;
    FTSTicker::FDelegateHandle TickHandle;

    // Cache para evitar rebuild a cada frame quando nada muda
    int32  LastDriverCount = -1;
    double LastRefreshTime = 0.0;
    float  RefreshInterval = 0.5f;  // segundos
};
