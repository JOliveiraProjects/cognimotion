#include "SCognitiveDebugDashboard.h"

#include "CognitiveDebugLog.h"
#include "CognitiveNPCBoneDriver.h"
#include "CognitiveMotionLearnerComponent.h"
#include "CognitiveNativeInferenceComponent.h"
#include "CognitiveLeaderObserverComponent.h"
#include "CognitiveBehaviorTypes.h"

#include "Widgets/SBoxPanel.h"
#include "Widgets/Layout/SBorder.h"
#include "Widgets/Layout/SScrollBox.h"
#include "Widgets/Layout/SSeparator.h"
#include "Widgets/Notifications/SProgressBar.h"
#include "Widgets/Layout/SWrapBox.h"
#include "Widgets/Text/STextBlock.h"
#include "Widgets/Input/SCheckBox.h"
#include "Styling/AppStyle.h"
#include "Editor.h"
#include "Engine/World.h"
#include "EngineUtils.h"
#include "GameFramework/Actor.h"

#define LOCTEXT_NAMESPACE "CognitiveDebugDashboard"

// ─────────────────────────────────────────────────────────────────────────────
namespace
{
    // Nome legível da ação discreta do Python (semântica canônica)
    FString ActionName(int32 Action)
    {
        switch (Action)
        {
            case 0: return TEXT("0 · idle");
            case 1: return TEXT("1 · forward");
            case 2: return TEXT("2 · backward");
            case 3: return TEXT("3 · left");
            case 4: return TEXT("4 · right");
            case 5: return TEXT("5 · run");
            case 6: return TEXT("6 · jump");
            case 7: return TEXT("7 · crouch");
            case 8: return TEXT("8 · stop");
            default: return FString::Printf(TEXT("%d"), Action);
        }
    }

    // Linha rótulo→valor padronizada (estilo Details panel da UE)
    TSharedRef<SWidget> MakeStatRow(const FText& Label, const FText& Value,
                                    const FSlateColor& ValueColor = FSlateColor::UseForeground())
    {
        return SNew(SHorizontalBox)
            + SHorizontalBox::Slot()
            .FillWidth(0.45f)
            .VAlign(VAlign_Center)
            [
                SNew(STextBlock)
                .Text(Label)
                .ColorAndOpacity(FSlateColor::UseSubduedForeground())
                .Font(FAppStyle::GetFontStyle("PropertyWindow.NormalFont"))
            ]
            + SHorizontalBox::Slot()
            .FillWidth(0.55f)
            .VAlign(VAlign_Center)
            [
                SNew(STextBlock)
                .Text(Value)
                .ColorAndOpacity(ValueColor)
                .Font(FAppStyle::GetFontStyle("PropertyWindow.BoldFont"))
            ];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
void SCognitiveDebugDashboard::Construct(const FArguments& InArgs)
{
    ChildSlot
    [
        SNew(SBorder)
        .BorderImage(FAppStyle::GetBrush("Brushes.Panel"))
        .Padding(8.f)
        [
            SNew(SVerticalBox)

            // ── Toolbar ──────────────────────────────────────────────────────
            + SVerticalBox::Slot()
            .AutoHeight()
            [
                SNew(SBorder)
                .BorderImage(FAppStyle::GetBrush("Brushes.Header"))
                .Padding(FMargin(10.f, 8.f))
                [
                    SNew(SHorizontalBox)

                    + SHorizontalBox::Slot()
                    .AutoWidth()
                    .VAlign(VAlign_Center)
                    [
                        SNew(STextBlock)
                        .Text(LOCTEXT("Title", "Cognitive Motion — NPC Debug Dashboard"))
                        .Font(FAppStyle::GetFontStyle("HeadingExtraSmall"))
                    ]

                    + SHorizontalBox::Slot()
                    .FillWidth(1.f)
                    .VAlign(VAlign_Center)
                    .Padding(16.f, 0.f, 0.f, 0.f)
                    [
                        SNew(STextBlock)
                        .Text(this, &SCognitiveDebugDashboard::GetHeaderSummary)
                        .ColorAndOpacity(FSlateColor::UseSubduedForeground())
                    ]

                    // Toggle de debug logs
                    + SHorizontalBox::Slot()
                    .AutoWidth()
                    .VAlign(VAlign_Center)
                    [
                        SNew(SCheckBox)
                        .Style(FAppStyle::Get(), "ToggleButtonCheckbox")
                        .IsChecked(this, &SCognitiveDebugDashboard::IsDebugEnabled)
                        .OnCheckStateChanged(this, &SCognitiveDebugDashboard::OnDebugToggleChanged)
                        .Padding(FMargin(10.f, 4.f))
                        [
                            SNew(STextBlock)
                            .Text(LOCTEXT("ToggleDebug", "Enable Debug Logs"))
                        ]
                    ]
                ]
            ]

            + SVerticalBox::Slot()
            .AutoHeight()
            .Padding(0.f, 6.f)
            [
                SNew(SSeparator)
            ]

            // ── Lista de NPCs (scroll) ───────────────────────────────────────
            + SVerticalBox::Slot()
            .FillHeight(1.f)
            [
                SNew(SScrollBox)
                + SScrollBox::Slot()
                [
                    SAssignNew(NPCListBox, SVerticalBox)
                ]
            ]
        ]
    ];

    RebuildNPCList();

    // Ticker de refresh
    TickHandle = FTSTicker::GetCoreTicker().AddTicker(
        FTickerDelegate::CreateRaw(this, &SCognitiveDebugDashboard::OnTick),
        RefreshInterval);
}

// ─────────────────────────────────────────────────────────────────────────────
SCognitiveDebugDashboard::~SCognitiveDebugDashboard()
{
    if (TickHandle.IsValid())
    {
        FTSTicker::GetCoreTicker().RemoveTicker(TickHandle);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
bool SCognitiveDebugDashboard::OnTick(float /*DeltaTime*/)
{
    RebuildNPCList();
    return true;  // continua tickando
}

// ─────────────────────────────────────────────────────────────────────────────
void SCognitiveDebugDashboard::CollectDrivers(
    TArray<TWeakObjectPtr<UCognitiveNPCBoneDriver>>& Out) const
{
    Out.Reset();

    UWorld* World = nullptr;

    // Prioriza o mundo PIE (jogo rodando) sobre o mundo do editor
    if (GEditor)
    {
        for (const FWorldContext& Ctx : GEditor->GetWorldContexts())
        {
            if (Ctx.WorldType == EWorldType::PIE && Ctx.World())
            {
                World = Ctx.World();
                break;
            }
        }
        if (!World)
        {
            World = GEditor->GetEditorWorldContext().World();
        }
    }

    if (!World) return;

    for (TActorIterator<AActor> It(World); It; ++It)
    {
        AActor* Actor = *It;
        if (!IsValid(Actor)) continue;

        if (UCognitiveNPCBoneDriver* Driver =
                Actor->FindComponentByClass<UCognitiveNPCBoneDriver>())
        {
            Out.Add(Driver);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
void SCognitiveDebugDashboard::RebuildNPCList()
{
    if (!NPCListBox.IsValid()) return;

    TArray<TWeakObjectPtr<UCognitiveNPCBoneDriver>> Drivers;
    CollectDrivers(Drivers);

    NPCListBox->ClearChildren();

    if (Drivers.Num() == 0)
    {
        NPCListBox->AddSlot()
        .AutoHeight()
        .Padding(12.f)
        [
            SNew(STextBlock)
            .Text(LOCTEXT("NoNPCs",
                "Nenhum NPC Cognitive na cena. Entre em Play (PIE) ou adicione um "
                "ator com o componente Cognitive NPC Bone Driver."))
            .ColorAndOpacity(FSlateColor::UseSubduedForeground())
            .AutoWrapText(true)
        ];
        LastDriverCount = 0;
        return;
    }

    for (const TWeakObjectPtr<UCognitiveNPCBoneDriver>& WeakDriver : Drivers)
    {
        if (UCognitiveNPCBoneDriver* Driver = WeakDriver.Get())
        {
            NPCListBox->AddSlot()
            .AutoHeight()
            .Padding(0.f, 0.f, 0.f, 8.f)
            [
                BuildNPCCard(Driver)
            ];
        }
    }

    LastDriverCount = Drivers.Num();
}

// ─────────────────────────────────────────────────────────────────────────────
TSharedRef<SWidget> SCognitiveDebugDashboard::BuildNPCCard(
    UCognitiveNPCBoneDriver* Driver) const
{
    AActor* Owner = Driver->GetOwner();
    const FString NPCName = Owner ? Owner->GetActorNameOrLabel() : TEXT("(desconhecido)");

    // ── Estado de observação ──────────────────────────────────────────────────
    const UEnum* StateEnum = StaticEnum<ECognitiveObservationState>();
    const FString StateStr = StateEnum
        ? StateEnum->GetDisplayNameTextByValue((int64)Driver->ObservationState).ToString()
        : TEXT("?");

    // ── Comportamento (mode/type) ─────────────────────────────────────────────
    const FString BehaviorStr = Driver->BehaviorContext.ToKey();

    // ── Conexão & dados ───────────────────────────────────────────────────────
    // IMPORTANTE: o LearnerComponent é o ÚNICO consumidor da RecvQueue do Python
    // (o BoneDriver não lê respostas — evita competição na fila). Portanto os
    // dados vivos de resposta (confiança, latência, ação) vêm do Learner.
    AActor* OwnerActor = Driver->GetOwner();
    UCognitiveMotionLearnerComponent* Learner = OwnerActor
        ? OwnerActor->FindComponentByClass<UCognitiveMotionLearnerComponent>()
        : nullptr;

    const bool  bConnected   = Learner ? Learner->HasValidResponse() : Driver->HasValidResponse();
    const int32 BonesApplied = Driver->BonesApplied;  // aplicados via AnimInstance
    const int32 ReqSent      = Driver->TotalRequestsSent;
    const float Latency      = Learner ? Learner->MotionQuality.LatencyMs : Driver->LastLatencyMs;
    const float Confidence   = Learner ? Learner->GetResponseConfidence() : Driver->GetLastConfidence();
    const int32 LastAction   = Learner ? Learner->LastSelectedStyle : 0;

    // Dados da inferência neural nativa (.pt), se o NPC tiver o componente.
    // Mostram o que o modelo está "pensando": ação prevista, confiança e a
    // magnitude do estado latente (h determinístico, z estocástico).
    float NativeMs = 0.f, HiddenNorm = 0.f, StochNorm = 0.f, ActConf = 0.f;
    int32 NativeAction = -1;
    bool  bHasNative = false, bNativeLoaded = false;
    if (OwnerActor)
    {
        if (UCognitiveNativeInferenceComponent* Native =
                OwnerActor->FindComponentByClass<UCognitiveNativeInferenceComponent>())
        {
            bHasNative    = true;
            bNativeLoaded = Native->IsModelLoaded();
            NativeAction  = Native->LastActionIndex;
            NativeMs      = Native->LastInferenceMs;
            HiddenNorm    = Native->LatentHiddenNorm;
            StochNorm     = Native->LatentStochasticNorm;
            ActConf       = Native->LastActionConfidence;
        }
    }

    // Saúde dos dados
    const bool  bBonesOK   = BonesApplied > 0;
    const bool  bLatencyOK = Latency > 0.f && Latency < Driver->MaxLatencyMs;
    const bool  bLatencyWarn = Latency >= Driver->MaxLatencyMs;

    const FText ConnText = bConnected
        ? LOCTEXT("Connected", "● Conectado (Python)")
        : LOCTEXT("Disconnected", "● Sem resposta válida");

    return SNew(SBorder)
        .BorderImage(FAppStyle::GetBrush("Brushes.Recessed"))
        .Padding(0.f)
        [
            SNew(SVerticalBox)

            // Cabeçalho do cartão
            + SVerticalBox::Slot()
            .AutoHeight()
            [
                SNew(SBorder)
                .BorderImage(FAppStyle::GetBrush("Brushes.Header"))
                .Padding(FMargin(10.f, 6.f))
                [
                    SNew(SHorizontalBox)
                    + SHorizontalBox::Slot()
                    .FillWidth(1.f)
                    .VAlign(VAlign_Center)
                    [
                        SNew(STextBlock)
                        .Text(FText::FromString(NPCName))
                        .Font(FAppStyle::GetFontStyle("BoldFont"))
                    ]
                    + SHorizontalBox::Slot()
                    .AutoWidth()
                    .VAlign(VAlign_Center)
                    [
                        SNew(STextBlock)
                        .Text(ConnText)
                        .ColorAndOpacity(HealthColor(bConnected))
                    ]
                ]
            ]

            // Corpo do cartão — duas colunas de stats
            + SVerticalBox::Slot()
            .AutoHeight()
            .Padding(10.f, 8.f)
            [
                SNew(SHorizontalBox)

                // Coluna 1: estado
                + SHorizontalBox::Slot()
                .FillWidth(0.5f)
                .Padding(0.f, 0.f, 8.f, 0.f)
                [
                    SNew(SVerticalBox)
                    + SVerticalBox::Slot().AutoHeight().Padding(0.f, 2.f)
                    [ MakeStatRow(LOCTEXT("ObsState", "Observation State"),
                                  FText::FromString(StateStr)) ]
                    + SVerticalBox::Slot().AutoHeight().Padding(0.f, 2.f)
                    [ MakeStatRow(LOCTEXT("Behavior", "Behavior"),
                                  FText::FromString(BehaviorStr)) ]
                    + SVerticalBox::Slot().AutoHeight().Padding(0.f, 2.f)
                    [ MakeStatRow(LOCTEXT("Endpoint", "Python Endpoint"),
                                  FText::FromString(FString::Printf(TEXT("%s:%d"),
                                      *Driver->PythonHost, Driver->PythonPort))) ]
                ]

                // Coluna 2: dados em tempo real
                + SHorizontalBox::Slot()
                .FillWidth(0.5f)
                .Padding(8.f, 0.f, 0.f, 0.f)
                [
                    SNew(SVerticalBox)
                    + SVerticalBox::Slot().AutoHeight().Padding(0.f, 2.f)
                    [ MakeStatRow(LOCTEXT("Bones", "Bones recebidos/aplicados"),
                                  FText::AsNumber(BonesApplied),
                                  HealthColor(bBonesOK)) ]
                    + SVerticalBox::Slot().AutoHeight().Padding(0.f, 2.f)
                    [ MakeStatRow(LOCTEXT("ReqSent", "Requests enviados"),
                                  FText::AsNumber(ReqSent)) ]
                    + SVerticalBox::Slot().AutoHeight().Padding(0.f, 2.f)
                    [ MakeStatRow(LOCTEXT("Latency", "Latência (ms)"),
                                  FText::AsNumber(FMath::RoundToInt(Latency)),
                                  HealthColor(bLatencyOK, bLatencyWarn)) ]
                    + SVerticalBox::Slot().AutoHeight().Padding(0.f, 2.f)
                    [ MakeStatRow(LOCTEXT("Confidence", "Confiança"),
                                  FText::FromString(FString::Printf(TEXT("%.2f"), Confidence)),
                                  HealthColor(Confidence > 0.1f)) ]
                    + SVerticalBox::Slot().AutoHeight().Padding(0.f, 2.f)
                    [ MakeStatRow(LOCTEXT("LastAction", "Última ação (Python)"),
                                  FText::FromString(ActionName(LastAction))) ]
                ]
            ]

            // Seção: Inferência Neural (.pt) — o que o modelo está "pensando".
            // Só aparece se o NPC tem o componente Native Inference.
            + SVerticalBox::Slot()
            .AutoHeight()
            .Padding(10.f, 0.f, 10.f, 8.f)
            [
                BuildNeuralSection(bHasNative, bNativeLoaded, NativeAction,
                                   ActConf, HiddenNorm, StochNorm, NativeMs)
            ]
        ];
}

// ─────────────────────────────────────────────────────────────────────────────
// Seção de inferência neural — mostra ação prevista, barra de confiança e a
// magnitude do estado latente (h/z). Segue o design system do editor (Brushes,
// fontes e cores nativas via FAppStyle).
TSharedRef<SWidget> SCognitiveDebugDashboard::BuildNeuralSection(
    bool bHasNative, bool bLoaded, int32 ActionIdx, float Confidence,
    float HiddenNorm, float StochNorm, float InferenceMs) const
{
    if (!bHasNative)
    {
        return SNew(STextBlock)
            .Text(LOCTEXT("NoNative", "Sem componente Native Inference (modo offline indisponível)"))
            .ColorAndOpacity(FSlateColor(FLinearColor(0.6f, 0.6f, 0.6f)))
            .Font(FAppStyle::GetFontStyle("SmallFont"));
    }

    return SNew(SBorder)
        .BorderImage(FAppStyle::GetBrush("Brushes.Panel"))
        .Padding(FMargin(10.f, 8.f))
        [
            SNew(SVerticalBox)

            // Título da seção
            + SVerticalBox::Slot().AutoHeight().Padding(0.f, 0.f, 0.f, 6.f)
            [
                SNew(SHorizontalBox)
                + SHorizontalBox::Slot().FillWidth(1.f).VAlign(VAlign_Center)
                [
                    SNew(STextBlock)
                    .Text(LOCTEXT("NeuralTitle", "Inferência Neural (.pt)"))
                    .Font(FAppStyle::GetFontStyle("BoldFont"))
                ]
                + SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center)
                [
                    SNew(STextBlock)
                    .Text(bLoaded ? LOCTEXT("PtLoaded", "● Modelo carregado")
                                  : LOCTEXT("PtNotLoaded", "● .pt não carregado"))
                    .ColorAndOpacity(HealthColor(bLoaded))
                    .Font(FAppStyle::GetFontStyle("SmallFont"))
                ]
            ]

            // Ação prevista
            + SVerticalBox::Slot().AutoHeight().Padding(0.f, 2.f)
            [ MakeStatRow(LOCTEXT("PredAction", "Ação prevista"),
                          FText::FromString(ActionName(ActionIdx))) ]

            // Barra de confiança (visual)
            + SVerticalBox::Slot().AutoHeight().Padding(0.f, 4.f)
            [
                SNew(SHorizontalBox)
                + SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center)
                .Padding(0.f, 0.f, 8.f, 0.f)
                [
                    SNew(STextBlock)
                    .Text(LOCTEXT("ConfBar", "Confiança"))
                    .MinDesiredWidth(120.f)
                ]
                + SHorizontalBox::Slot().FillWidth(1.f).VAlign(VAlign_Center)
                [
                    SNew(SProgressBar)
                    .Percent(FMath::Clamp(Confidence, 0.f, 1.f))
                    .FillColorAndOpacity(HealthColor(Confidence > 0.5f, Confidence > 0.25f))
                ]
                + SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center)
                .Padding(8.f, 0.f, 0.f, 0.f)
                [
                    SNew(STextBlock)
                    .Text(FText::FromString(FString::Printf(TEXT("%.0f%%"), Confidence * 100.f)))
                ]
            ]

            // Estado latente (normas h e z) — leitura compacta do "pensamento"
            + SVerticalBox::Slot().AutoHeight().Padding(0.f, 2.f)
            [ MakeStatRow(LOCTEXT("LatentH", "Estado latente |h| (determinístico)"),
                          FText::FromString(FString::Printf(TEXT("%.2f"), HiddenNorm))) ]
            + SVerticalBox::Slot().AutoHeight().Padding(0.f, 2.f)
            [ MakeStatRow(LOCTEXT("LatentZ", "Estado latente |z| (estocástico)"),
                          FText::FromString(FString::Printf(TEXT("%.2f"), StochNorm))) ]
            + SVerticalBox::Slot().AutoHeight().Padding(0.f, 2.f)
            [ MakeStatRow(LOCTEXT("NativeMs", "Tempo de inferência (ms)"),
                          FText::FromString(FString::Printf(TEXT("%.2f"), InferenceMs)),
                          HealthColor(InferenceMs < 16.f, InferenceMs < 33.f)) ]
        ];
}

// ─────────────────────────────────────────────────────────────────────────────
ECheckBoxState SCognitiveDebugDashboard::IsDebugEnabled() const
{
    return FCognitiveDebugState::IsDebugEnabled()
        ? ECheckBoxState::Checked
        : ECheckBoxState::Unchecked;
}

void SCognitiveDebugDashboard::OnDebugToggleChanged(ECheckBoxState NewState)
{
    const bool bEnabled = (NewState == ECheckBoxState::Checked);
    FCognitiveDebugState::SetDebugEnabled(bEnabled);

    UE_LOG(LogCognitiveMotion, Log, TEXT("[Dashboard] Debug logs %s"),
        bEnabled ? TEXT("ATIVADOS") : TEXT("DESATIVADOS"));
}

// ─────────────────────────────────────────────────────────────────────────────
FText SCognitiveDebugDashboard::GetHeaderSummary() const
{
    TArray<TWeakObjectPtr<UCognitiveNPCBoneDriver>> Drivers;
    CollectDrivers(Drivers);

    int32 Live = 0;
    for (const auto& D : Drivers)
        if (UCognitiveNPCBoneDriver* Drv = D.Get())
        {
            AActor* Own = Drv->GetOwner();
            UCognitiveMotionLearnerComponent* L = Own
                ? Own->FindComponentByClass<UCognitiveMotionLearnerComponent>() : nullptr;
            const bool bOk = L ? L->HasValidResponse() : Drv->HasValidResponse();
            if (bOk) ++Live;
        }

    return FText::Format(
        LOCTEXT("HeaderFmt", "{0} NPC(s) na cena · {1} com resposta ativa"),
        FText::AsNumber(Drivers.Num()),
        FText::AsNumber(Live));
}

// ─────────────────────────────────────────────────────────────────────────────
FSlateColor SCognitiveDebugDashboard::HealthColor(bool bGood, bool bWarn)
{
    if (bWarn)  return FSlateColor(FLinearColor(0.95f, 0.65f, 0.10f)); // amarelo
    if (bGood)  return FSlateColor(FLinearColor(0.20f, 0.80f, 0.35f)); // verde
    return            FSlateColor(FLinearColor(0.85f, 0.25f, 0.25f));  // vermelho
}

#undef LOCTEXT_NAMESPACE
