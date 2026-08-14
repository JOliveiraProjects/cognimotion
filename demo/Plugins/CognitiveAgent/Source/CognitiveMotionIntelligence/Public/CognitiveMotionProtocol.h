#pragma once

#include "CoreMinimal.h"
#include "CognitiveMotionTypes.h"

namespace CognitiveMotionProtocol
{
    static constexpr uint32 MagicHeader     = 0x434D4900;
    static constexpr uint8  ProtocolVersion = 1;

    // BC-04 FIX: tipos 0x05/0x06/0x07 agora são valores do enum (não constexpr soltos),
    // permitindo switch() exhaustive e eliminando duplicação com o lado Python.
    enum class EMessageType : uint8
    {
        MotionRequest     = 0x01,
        MotionResponse    = 0x02,
        PoseFrame         = 0x03,
        LeaderSequence    = 0x05,   // movido de static constexpr
        AutonomousRequest = 0x06,   // movido de static constexpr
        MotionAction      = 0x07,   // movido de static constexpr
        Perception        = 0x08,   // NPC → Python: entidades percebidas
        Teach             = 0x09,   // Líder → Python: vocabulário de ações
        TrainingRegister  = 0x0A,   // Editor → Python: registra treino (tipo+reação+anim)
        TeachingScenario  = 0x0B,   // Editor → Python: cenário de ensino (pede decisão)
        TeachingChoice    = 0x0C,   // Python → Editor: reação escolhida pelo agente
        TeachingFeedback  = 0x0D,   // Editor → Python: correção (certo/errado+sugestões)
        Handshake         = 0x10,
        HandshakeAck      = 0x11,
        Ping              = 0x20,
        Pong              = 0x21,
        Error             = 0xFF,
    };

    struct FPacketHeader
    {
        uint32 Magic;
        uint8  Version;
        uint8  MessageType;
        uint16 Flags;
        uint32 PayloadSize;
        uint32 Checksum;
        int64  SequenceId;
    };

    static constexpr int32 HeaderSize = sizeof(FPacketHeader);

    // ── Serialização ──────────────────────────────────────────────────────────
    TArray<uint8> SerializeRequest(const FCognitiveMotionRequest& Request);
    bool          DeserializeResponse(const TArray<uint8>& Data, FCognitiveMotionResponse& OutResponse);
    TArray<uint8> SerializePoseFrame(const FCognitivePoseFrame& Frame);
    bool          DeserializePoseFrame(const TArray<uint8>& Data, FCognitivePoseFrame& OutFrame);
    TArray<uint8> BuildHandshake(int32 ObsDim, int32 EmbeddingDim);
    bool          ValidateHandshakeAck(const TArray<uint8>& Data);
    uint32        ComputeChecksum(const uint8* Data, int32 Size);

    // ── Write helpers (endianness: int32 BE, float/double/int64 LE nativo) ────
    void WriteFloat(TArray<uint8>& Out, float Value);
    void WriteDouble(TArray<uint8>& Out, double Value);
    void WriteInt32(TArray<uint8>& Out, int32 Value);
    void WriteInt64(TArray<uint8>& Out, int64 Value);
    void WriteVector(TArray<uint8>& Out, const FVector& V);
    void WriteQuat(TArray<uint8>& Out, const FQuat& Q);
    void WriteTransform(TArray<uint8>& Out, const FTransform& T);
    void WriteFloatArray(TArray<uint8>& Out, const TArray<float>& Arr);
    // BA-01 FIX: WriteString agora usa FTCHARToUTF8::Length() (bytes UTF-8 reais)
    // em vez de S.Len() (número de chars TCHAR). Veja implementação no .cpp.
    void WriteString(TArray<uint8>& Out, const FString& S);

    // ── Read helpers ──────────────────────────────────────────────────────────
    float         ReadFloat(const uint8* Data, int32& Offset);
    double        ReadDouble(const uint8* Data, int32& Offset);
    int32         ReadInt32(const uint8* Data, int32& Offset);
    int64         ReadInt64(const uint8* Data, int32& Offset);
    FVector       ReadVector(const uint8* Data, int32& Offset);
    FQuat         ReadQuat(const uint8* Data, int32& Offset);
    FTransform    ReadTransform(const uint8* Data, int32& Offset);
    TArray<float> ReadFloatArray(const uint8* Data, int32& Offset);
    FString       ReadString(const uint8* Data, int32& Offset, int32 BufSize = -1);

    // ── Leader Sequence ───────────────────────────────────────────────────────
    // BC-03 context: líder e seguidor agora usam a mesma lista de bones nomeados.

    struct FLeaderSequencePayload
    {
        TArray<FCognitivePoseFrame> Frames;
        int32   SequenceId      = 0;
        double  StartTimestamp  = 0.0;
        double  EndTimestamp    = 0.0;
        int64   LeaderNPCId     = 0;
        int64   FollowerNPCId   = 0;
    };

    TArray<uint8> SerializeLeaderSequence(const FLeaderSequencePayload& Payload);

    // ── Perception (NPC → Python) ─────────────────────────────────────────────
    // Uma entidade percebida, achatada para o wire.
    struct FPerceivedEntityWire
    {
        int32  Category    = 0;   // ECognitiveEntityCategory
        int32  Disposition = 0;   // ECognitiveDisposition
        int32  Role        = 0;   // ECognitiveSocialRole
        int32  Reaction    = 0;   // ECognitiveReaction (sugerida)
        int32  VehicleType = 0;   // ECognitiveVehicleType
        int32  TrafficState= 0;   // ECognitiveTrafficState
        float  Distance    = 0.f;
        float  DirX = 0.f, DirY = 0.f, DirZ = 0.f;  // direção relativa (local)
        float  ThreatWeight= 0.f;
    };

    struct FPerceptionPayload
    {
        int64 NPCId = 0;
        TArray<FPerceivedEntityWire> Entities;
    };

    TArray<uint8> SerializePerception(const FPerceptionPayload& Payload);

    // ── Teach (Líder → Python) ────────────────────────────────────────────────
    struct FTaughtActionWire
    {
        int32   Verb        = 0;   // ECognitiveActionVerb
        int32   ActionIndex = 0;
        int32   TargetCategory = 0;
        FString Label;             // texto do significado ("Run (correr)")
    };

    struct FTeachPayload
    {
        int64 LeaderNPCId = 0;
        int32 CurrentVerb = 0;     // verbo demonstrado agora
        int32 LeaderCategory = 0;
        TArray<FTaughtActionWire> Vocabulary;
        int32 CurrentEmotion = -1; // emoção rotulada agora (-1 = sem rótulo)
        int32 CurrentAction  = -1; // ação rotulada agora (-1 = sem rótulo)
    };

    TArray<uint8> SerializeTeach(const FTeachPayload& Payload);

    // ── Treino & Ensino (Editor ⇄ Python) ─────────────────────────────────────
    // TrainingRegister: registra uma demonstração rotulada no catálogo Python.
    struct FTrainingRegisterWire
    {
        FString TrainingType;    // ex: "combate"
        FString ReactionName;    // ex: "agachar com arma"
        FString AnimationPath;   // ex: /Game/Anims/X.X
        FString Notes;
    };
    TArray<uint8> SerializeTrainingRegister(const FTrainingRegisterWire& W);

    // TeachingScenario: cenário + reações candidatas; Python responde TeachingChoice.
    struct FScenarioEntityWire
    {
        FString Kind;            // "enemy", "ally", "object", "danger", ...
        int32   Count    = 0;
        int32   FacingMe = 0;
        float   DistanceM = 0.f;
    };
    struct FTeachingScenarioWire
    {
        int64   ScenarioId = 0;
        FString TrainingType;
        FString Description;
        TArray<FScenarioEntityWire> Entities;
        TArray<FString> CandidateReactions;  // vazio = todas do tipo
    };
    TArray<uint8> SerializeTeachingScenario(const FTeachingScenarioWire& W);

    // TeachingChoice (Python → UE): escolha do agente para um cenário.
    struct FTeachingChoiceWire
    {
        int64   ScenarioId = 0;
        FString ChosenReaction;
        float   Confidence = 0.f;
        FString Rationale;
    };
    bool DeserializeTeachingChoice(const TArray<uint8>& Data, FTeachingChoiceWire& Out);

    // TeachingFeedback: correção do professor sobre a escolha.
    struct FTeachingFeedbackWire
    {
        int64   ScenarioId = 0;
        int32   bCorrect   = 0;
        FString ChosenReaction;
        TArray<FString> SuggestedReactions;
        FString Comment;
    };
    TArray<uint8> SerializeTeachingFeedback(const FTeachingFeedbackWire& W);
}

