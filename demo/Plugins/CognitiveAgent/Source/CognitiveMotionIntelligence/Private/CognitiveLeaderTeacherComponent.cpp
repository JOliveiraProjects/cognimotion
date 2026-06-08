#include "CognitiveLeaderTeacherComponent.h"

#include "CognitiveInferenceSubsystem.h"
#include "CognitiveMotionProtocol.h"
#include "CognitiveEntityTagComponent.h"
#include "CognitiveWorldPerceptionTypes.h"
#include "CognitiveDebugLog.h"

#include "GameFramework/Actor.h"
#include "Engine/GameInstance.h"
#include "Engine/World.h"
#include "EngineUtils.h"               // TActorIterator

// ─────────────────────────────────────────────────────────────────────────────
UCognitiveLeaderTeacherComponent::UCognitiveLeaderTeacherComponent()
{
    PrimaryComponentTick.bCanEverTick = true;
    BuildDefaultVocabulary();
}

// ─────────────────────────────────────────────────────────────────────────────
void UCognitiveLeaderTeacherComponent::BuildDefaultVocabulary()
{
    // Só preenche se o usuário não customizou no editor.
    if (ActionVocabulary.Num() > 0) return;

    // Mapeia cada verbo para o índice de ação atual do executor Python
    // (0=idle 1=forward 2=backward 3=left 4=right 5=run 6=jump 7=crouch 8=stop).
    // Verbos sem ação física ainda (crawl/vault/pickup/flee/hide/attack/defend)
    // recebem índice -1 = "ensinado mas ainda sem efetuador". Quando você
    // adicionar a ação no Python e a animação, troque o índice aqui.
    auto Add = [this](ECognitiveActionVerb V, int32 Idx, ECognitiveEntityCategory Tgt)
    {
        FCognitiveTaughtAction A;
        A.Verb = V; A.ActionIndex = Idx; A.TargetCategory = Tgt;
        ActionVocabulary.Add(A);
    };

    Add(ECognitiveActionVerb::Idle,   0, ECognitiveEntityCategory::Unknown);
    Add(ECognitiveActionVerb::Walk,   1, ECognitiveEntityCategory::Unknown);
    Add(ECognitiveActionVerb::Run,    5, ECognitiveEntityCategory::Unknown);
    Add(ECognitiveActionVerb::Jump,   6, ECognitiveEntityCategory::Unknown);
    Add(ECognitiveActionVerb::Crouch, 7, ECognitiveEntityCategory::Unknown);
    Add(ECognitiveActionVerb::Crawl,  -1, ECognitiveEntityCategory::Unknown);
    Add(ECognitiveActionVerb::Vault,  -1, ECognitiveEntityCategory::Cover);
    Add(ECognitiveActionVerb::PickUp, -1, ECognitiveEntityCategory::Pickup);
    Add(ECognitiveActionVerb::Flee,   -1, ECognitiveEntityCategory::Hazard);
    Add(ECognitiveActionVerb::Hide,   -1, ECognitiveEntityCategory::Cover);
    Add(ECognitiveActionVerb::Attack, -1, ECognitiveEntityCategory::Character);
    Add(ECognitiveActionVerb::Defend, -1, ECognitiveEntityCategory::Character);
}

// ─────────────────────────────────────────────────────────────────────────────
int32 UCognitiveLeaderTeacherComponent::GetActionIndexForVerb(ECognitiveActionVerb Verb) const
{
    for (const FCognitiveTaughtAction& A : ActionVocabulary)
        if (A.Verb == Verb) return A.ActionIndex;
    return -1;
}

// ─────────────────────────────────────────────────────────────────────────────
void UCognitiveLeaderTeacherComponent::BeginPlay()
{
    Super::BeginPlay();

    if (UGameInstance* GI = GetWorld() ? GetWorld()->GetGameInstance() : nullptr)
    {
        if (UCognitiveInferenceSubsystem* Sub =
            GI->GetSubsystem<UCognitiveInferenceSubsystem>())
        {
            InferenceSubsystem = Sub;
        }
    }

    // Garante que o próprio líder tenha um Entity Tag (Character por padrão),
    // para que NPCs o percebam como o modelo a seguir.
    if (AActor* Owner = GetOwner())
    {
        if (!Owner->FindComponentByClass<UCognitiveEntityTagComponent>())
        {
            UCognitiveEntityTagComponent* Tag =
                NewObject<UCognitiveEntityTagComponent>(Owner);
            if (Tag)
            {
                Tag->Category    = LeaderSelfCategory;
                Tag->Disposition = ECognitiveDisposition::Friend;
                Tag->RegisterComponent();
                Owner->AddInstanceComponent(Tag);
            }
        }
    }

    if (bAutoTagScene)
    {
        AutoTagScene();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
void UCognitiveLeaderTeacherComponent::AutoTagScene()
{
    UWorld* World = GetWorld();
    if (!World) return;

    int32 Tagged = 0;

    for (TActorIterator<AActor> It(World); It; ++It)
    {
        AActor* Actor = *It;
        if (!IsValid(Actor) || Actor == GetOwner()) continue;

        // Já tem tag? respeita a configuração manual e não sobrescreve.
        if (Actor->FindComponentByClass<UCognitiveEntityTagComponent>()) continue;

        // Heurística por nome de classe. Conservadora: só taggeia o que
        // reconhece; o resto fica sem tag (Ignore implícito), evitando ruído.
        const FString ClassName = Actor->GetClass()->GetName();
        ECognitiveEntityCategory Cat = ECognitiveEntityCategory::Unknown;
        ECognitiveVehicleType    Veh = ECognitiveVehicleType::None;
        float Threat = 0.f;
        bool  bPickup = false;

        auto Has = [&ClassName](const TCHAR* Sub)
        {
            return ClassName.Contains(Sub, ESearchCase::IgnoreCase);
        };

        if (Has(TEXT("Character")) || Has(TEXT("Pawn")) || Has(TEXT("NPC")) ||
            Has(TEXT("Enemy")) || Has(TEXT("Bot")))
        {
            Cat = ECognitiveEntityCategory::Character;
            if (Has(TEXT("Enemy")) || Has(TEXT("Bot"))) Threat = 0.6f;
        }
        else if (Has(TEXT("Weapon")) || Has(TEXT("Gun")) || Has(TEXT("Rifle")) ||
                 Has(TEXT("Pistol")) || Has(TEXT("Sword")))
        {
            Cat = ECognitiveEntityCategory::Weapon; bPickup = true;
        }
        else if (Has(TEXT("Money")) || Has(TEXT("Coin")) || Has(TEXT("Cash")) ||
                 Has(TEXT("Item")) || Has(TEXT("Pickup")) || Has(TEXT("Ammo")))
        {
            Cat = ECognitiveEntityCategory::Pickup; bPickup = true;
        }
        else if (Has(TEXT("Car")) || Has(TEXT("Vehicle")) || Has(TEXT("Truck")))
        {
            Cat = ECognitiveEntityCategory::Vehicle; Veh = ECognitiveVehicleType::Car;
        }
        else if (Has(TEXT("Moto")) || Has(TEXT("Bike")))
        {
            Cat = ECognitiveEntityCategory::Vehicle; Veh = ECognitiveVehicleType::Motorcycle;
        }
        else if (Has(TEXT("Traffic")) || Has(TEXT("Semaph")) || Has(TEXT("Signal")))
        {
            Cat = ECognitiveEntityCategory::TrafficLight;
        }
        else if (Has(TEXT("Hazard")) || Has(TEXT("Fire")) || Has(TEXT("Trap")) ||
                 Has(TEXT("Spike")) || Has(TEXT("Lava")))
        {
            Cat = ECognitiveEntityCategory::Hazard; Threat = 0.8f;
        }
        else if (Has(TEXT("Cover")) || Has(TEXT("Wall")) || Has(TEXT("Crate")) ||
                 Has(TEXT("Barrier")))
        {
            Cat = ECognitiveEntityCategory::Cover;
        }

        if (Cat == ECognitiveEntityCategory::Unknown) continue;  // não reconhecido

        UCognitiveEntityTagComponent* Tag =
            NewObject<UCognitiveEntityTagComponent>(Actor);
        if (!Tag) continue;
        Tag->Category     = Cat;
        Tag->VehicleType  = Veh;
        Tag->ThreatWeight = Threat;
        Tag->bCanPickUp   = bPickup;
        if (Cat == ECognitiveEntityCategory::Character)
            Tag->Disposition = Threat > 0.f ? ECognitiveDisposition::Enemy
                                            : ECognitiveDisposition::Neutral;
        Tag->RegisterComponent();
        Actor->AddInstanceComponent(Tag);
        ++Tagged;
    }

    CMI_DBG("[LeaderTeacher] Auto-tag concluído: %d atores classificados na cena. "
            "(Ajuste manualmente os que precisarem de categoria específica.)", Tagged);
}

// ─────────────────────────────────────────────────────────────────────────────
void UCognitiveLeaderTeacherComponent::TickComponent(
    float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

    if (!InferenceSubsystem.IsValid()) return;

    const float Interval = 1.f / FMath::Max(TeachRateHz, 0.2f);
    SendAccumulator += DeltaTime;
    if (SendAccumulator >= Interval)
    {
        SendAccumulator = 0.f;
        SendVocabulary();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
void UCognitiveLeaderTeacherComponent::SendVocabulary()
{
    if (!InferenceSubsystem.IsValid()) return;

    CognitiveMotionProtocol::FTeachPayload Payload;
    Payload.LeaderNPCId    = 0;
    Payload.CurrentVerb    = static_cast<int32>(CurrentVerb);
    Payload.LeaderCategory = static_cast<int32>(LeaderSelfCategory);

    for (const FCognitiveTaughtAction& A : ActionVocabulary)
    {
        CognitiveMotionProtocol::FTaughtActionWire W;
        W.Verb           = static_cast<int32>(A.Verb);
        W.ActionIndex    = A.ActionIndex;
        W.TargetCategory = static_cast<int32>(A.TargetCategory);
        W.Label          = VerbToString(A.Verb);
        Payload.Vocabulary.Add(W);
    }

    const TArray<uint8> Data = CognitiveMotionProtocol::SerializeTeach(Payload);
    if (Data.Num() > 0)
    {
        InferenceSubsystem->SendRawMessage(Data);
        if (!bVocabularySent)
        {
            bVocabularySent = true;
            CMI_DBG("[LeaderTeacher] Vocabulário enviado ao Python: %d verbos. "
                    "Verbo atual=%s", ActionVocabulary.Num(),
                    *VerbToString(CurrentVerb));
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
FString UCognitiveLeaderTeacherComponent::VerbToString(ECognitiveActionVerb Verb)
{
    switch (Verb)
    {
        case ECognitiveActionVerb::Idle:   return TEXT("idle");
        case ECognitiveActionVerb::Walk:   return TEXT("walk");
        case ECognitiveActionVerb::Run:    return TEXT("run");
        case ECognitiveActionVerb::Jump:   return TEXT("jump");
        case ECognitiveActionVerb::Crouch: return TEXT("crouch");
        case ECognitiveActionVerb::Crawl:  return TEXT("crawl");
        case ECognitiveActionVerb::Vault:  return TEXT("vault");
        case ECognitiveActionVerb::PickUp: return TEXT("pickup");
        case ECognitiveActionVerb::Flee:   return TEXT("flee");
        case ECognitiveActionVerb::Hide:   return TEXT("hide");
        case ECognitiveActionVerb::Attack: return TEXT("attack");
        case ECognitiveActionVerb::Defend: return TEXT("defend");
        default:                           return TEXT("unknown");
    }
}
