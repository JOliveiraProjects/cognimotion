#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "CognitiveWorldPerceptionTypes.h"
#include "CognitiveLeaderTeacherComponent.generated.h"

class UCognitiveInferenceSubsystem;

// ─────────────────────────────────────────────────────────────────────────────
// ECognitiveActionVerb
// O catálogo de VERBOS que o líder pode demonstrar e ensinar aos NPCs.
// Cada verbo recebe um significado (label) que é enviado ao Python para
// fundamentar (ground) o que cada movimento/decisão É.
// ─────────────────────────────────────────────────────────────────────────────
UENUM(BlueprintType)
// ─────────────────────────────────────────────────────────────────────────────
// ECognitiveTeachEmotion
// Emoção que o líder rotula durante a demonstração (SetCurrentEmotion). Os
// valores DEVEM bater com EMOTION_NAMES no Python (binary_protocol.py) e com
// EMOTIONS em demonstration_learning.py.
// ─────────────────────────────────────────────────────────────────────────────
UENUM(BlueprintType)
enum class ECognitiveTeachEmotion : uint8
{
    Calm       = 0  UMETA(DisplayName = "Calm (calmo)"),
    Happy      = 1  UMETA(DisplayName = "Happy (feliz)"),
    Alert      = 2  UMETA(DisplayName = "Alert (alerta)"),
    Fear       = 3  UMETA(DisplayName = "Fear (medo)"),
    Anger      = 4  UMETA(DisplayName = "Anger (raiva)"),
    Panic      = 5  UMETA(DisplayName = "Panic (pânico)"),
    Confident  = 6  UMETA(DisplayName = "Confident (confiante)"),
    Suspicious = 7  UMETA(DisplayName = "Suspicious (desconfiado)"),
};

// ─────────────────────────────────────────────────────────────────────────────
UENUM(BlueprintType)
enum class ECognitiveActionVerb : uint8
{
    Idle     = 0  UMETA(DisplayName = "Idle (parado)"),
    Walk     = 1  UMETA(DisplayName = "Walk (andar)"),
    Run      = 2  UMETA(DisplayName = "Run (correr)"),
    Jump     = 3  UMETA(DisplayName = "Jump (pular)"),
    Crouch   = 4  UMETA(DisplayName = "Crouch (agachar)"),
    Crawl    = 5  UMETA(DisplayName = "Crawl (rastejar)"),
    Vault    = 6  UMETA(DisplayName = "Vault (saltar obstáculo)"),
    PickUp   = 7  UMETA(DisplayName = "Pick Up (pegar objeto)"),
    Flee     = 8  UMETA(DisplayName = "Flee (fugir)"),
    Hide     = 9  UMETA(DisplayName = "Hide (esconder)"),
    Attack   = 10 UMETA(DisplayName = "Attack (atacar)"),
    Defend   = 11 UMETA(DisplayName = "Defend (defender)"),
};

// ─────────────────────────────────────────────────────────────────────────────
// FCognitiveTaughtAction
// Uma demonstração rotulada: "este movimento que estou fazendo agora É <Verb>".
// O líder publica isto e os NPCs recebem o par (verbo, intenção) para aprender
// o significado, não só copiar a pose cega.
// ─────────────────────────────────────────────────────────────────────────────
USTRUCT(BlueprintType)
struct COGNITIVEMOTIONINTELLIGENCE_API FCognitiveTaughtAction
{
    GENERATED_BODY()

    // Verbo sendo demonstrado neste instante.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Teach")
    ECognitiveActionVerb Verb = ECognitiveActionVerb::Idle;

    // Índice da ação no action_map do executor Python (0..8 hoje).
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Teach")
    int32 ActionIndex = 0;

    // Categoria de entidade-alvo que torna este verbo relevante (ex.: Flee↔Hazard,
    // PickUp↔Weapon/Pickup, Attack↔Character inimigo). Unknown = sem alvo.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Teach")
    ECognitiveEntityCategory TargetCategory = ECognitiveEntityCategory::Unknown;
};

// ─────────────────────────────────────────────────────────────────────────────
// UCognitiveLeaderTeacherComponent
//
// Coloque ESTE componente NO LÍDER (o player ou o ator que os NPCs imitam).
//
// Faz três coisas:
//   1. Carrega um VOCABULÁRIO de ações (verbo → índice de ação → significado),
//      enviado ao Python uma vez na conexão. Assim o servidor sabe o que cada
//      ação SIGNIFICA (correr, pular, rastejar, fugir, atacar...), em vez de
//      tratar a ação como um número opaco.
//   2. Publica em runtime QUAL verbo o líder está demonstrando agora
//      (CurrentVerb). Isso rotula a demonstração — o NPC aprende "isto é fugir",
//      não só "reproduza estes ossos".
//   3. Opcionalmente AUTO-TAGGEIA a cena: percorre os atores e adiciona
//      UCognitiveEntityTagComponent por heurística de classe (player, veículos,
//      itens). Resolve o problema do log "nenhum tem Cognitive Entity Tag" sem
//      você ter que taggear 24 atores na mão.
//
// O significado dos verbos é texto puro — não cria animação. Verbos sem ação
// física correspondente (crawl/vault/pickup) só passam a EXISTIR quando você
// adicionar a ação no action_map do Python e a animação no AnimGraph.
// ─────────────────────────────────────────────────────────────────────────────
UCLASS(ClassGroup=(Cognitive), meta=(BlueprintSpawnableComponent),
       DisplayName="Cognitive Leader Teacher")
class COGNITIVEMOTIONINTELLIGENCE_API UCognitiveLeaderTeacherComponent
    : public UActorComponent
{
    GENERATED_BODY()

public:
    UCognitiveLeaderTeacherComponent();

    // ── Vocabulário ───────────────────────────────────────────────────────────
    // Catálogo de ações que este líder ensina. Preenchido com um default
    // razoável no construtor; edite no editor para customizar.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Teach")
    TArray<FCognitiveTaughtAction> ActionVocabulary;

    // Verbo que o líder está demonstrando AGORA. Atualize por Blueprint/C++
    // durante a gravação (ex.: ao apertar Shift para correr → SetCurrentVerb(Run)).
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Teach")
    ECognitiveActionVerb CurrentVerb = ECognitiveActionVerb::Idle;

    // ── Demonstração de EMOÇÃO/AÇÃO (aprendizado por demonstração) ──────────────
    // O líder rotula, durante a cena, o que está SENTINDO e a AÇÃO que toma.
    // O NPC aprende a associação percepção→emoção→ação e generaliza.
    // Defina por Blueprint: SetCurrentEmotion(Fear), SetCurrentTeachAction(5).
    // bDemonstrating=false → não envia rótulo (o NPC usa o que já aprendeu).
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Teach|Demo")
    bool bDemonstrating = false;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Teach|Demo")
    ECognitiveTeachEmotion CurrentEmotion = ECognitiveTeachEmotion::Calm;

    // Índice da ação demonstrada (0..8: idle/fwd/back/left/right/run/jump/crouch/stop).
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Teach|Demo",
              meta=(ClampMin="0", ClampMax="8"))
    int32 CurrentTeachAction = 0;

    // Categoria semântica que o PRÓPRIO líder representa para os NPCs (em geral
    // Character/Friend — ele é o modelo a imitar). Usado pelo auto-tag do líder.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Teach")
    ECognitiveEntityCategory LeaderSelfCategory = ECognitiveEntityCategory::Character;

    // ── Auto-tagging da cena ────────────────────────────────────────────────────
    // Se true, no BeginPlay o líder percorre a cena e adiciona Entity Tag aos
    // atores que ainda não têm, classificando por heurística de nome de classe.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Teach|AutoTag")
    bool bAutoTagScene = true;

    // Frequência (Hz) com que o vocabulário/verbo atual é reenviado ao Python.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Teach",
              meta=(ClampMin="0.2", ClampMax="10.0"))
    float TeachRateHz = 1.0f;

    // ── Blueprint API ───────────────────────────────────────────────────────────
    UFUNCTION(BlueprintCallable, Category="Cognitive|Teach")
    void SetCurrentVerb(ECognitiveActionVerb NewVerb) { CurrentVerb = NewVerb; }

    // Rotula a emoção que o líder está sentindo/demonstrando agora. Ativa o
    // modo de demonstração (passa a enviar rótulos ao Python para aprendizado).
    UFUNCTION(BlueprintCallable, Category="Cognitive|Teach|Demo")
    void SetCurrentEmotion(ECognitiveTeachEmotion NewEmotion)
    {
        CurrentEmotion = NewEmotion;
        bDemonstrating = true;
    }

    // Rotula a ação que o líder está tomando agora (0..8).
    UFUNCTION(BlueprintCallable, Category="Cognitive|Teach|Demo")
    void SetCurrentTeachAction(int32 ActionIndex)
    {
        CurrentTeachAction = ActionIndex;
        bDemonstrating = true;
    }

    // Encerra a demonstração: o NPC passa a usar o que já aprendeu.
    UFUNCTION(BlueprintCallable, Category="Cognitive|Teach|Demo")
    void StopDemonstrating() { bDemonstrating = false; }

    UFUNCTION(BlueprintCallable, Category="Cognitive|Teach")
    int32 GetActionIndexForVerb(ECognitiveActionVerb Verb) const;

    // ── UActorComponent ─────────────────────────────────────────────────────────
    virtual void BeginPlay() override;
    virtual void TickComponent(float DeltaTime, ELevelTick TickType,
                                FActorComponentTickFunction* ThisTickFunction) override;

private:
    void BuildDefaultVocabulary();
    void AutoTagScene();
    void SendVocabulary();        // envia catálogo + verbo atual ao Python
    static FString VerbToString(ECognitiveActionVerb Verb);

    TWeakObjectPtr<UCognitiveInferenceSubsystem> InferenceSubsystem;
    float SendAccumulator = 0.f;
    bool  bVocabularySent = false;
};
