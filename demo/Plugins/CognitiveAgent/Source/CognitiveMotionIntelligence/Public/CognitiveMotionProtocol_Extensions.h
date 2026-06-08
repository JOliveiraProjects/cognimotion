// CognitiveMotionProtocol_Extensions.h
// ========================================
// Extensões ao CognitiveMotionProtocol existente.
//
// INSTRUÇÃO DE INTEGRAÇÃO:
//   1. Adicione os conteúdos desta seção ao FINAL de CognitiveMotionProtocol.h
//      (dentro do namespace CognitiveMotionProtocol, antes do fechamento })
//   2. Implemente as funções em CognitiveMotionProtocol.cpp
//   3. Adicione NPCId e TargetNPCId à FCognitiveMotionRequest em CognitiveMotionTypes.h
//
// NÃO MODIFIQUE os tipos e funções existentes — apenas adicione ao final.

// ──────────────────────────────────────────────────────────────────────────────
// A. Adicionar ao namespace CognitiveMotionProtocol (CognitiveMotionProtocol.h)
// ──────────────────────────────────────────────────────────────────────────────

/*
// Novos tipos de mensagem
static constexpr uint8 MSG_LEADER_SEQUENCE    = 0x05;
static constexpr uint8 MSG_AUTONOMOUS_REQUEST = 0x06;
static constexpr uint8 MSG_MOTION_ACTION      = 0x07;

// Payload da sequência de poses do líder
// (definido também em CognitiveLeaderObserverComponent.h como USTRUCT)
struct FLeaderSequencePayload
{
    TArray<FCognitivePoseFrame> Frames;
    int32  SequenceId      = 0;
    double StartTimestamp  = 0.0;
    double EndTimestamp    = 0.0;
    int64  LeaderNPCId     = 0;
    int64  FollowerNPCId   = 0;
};

// Serialização da sequência do líder → TArray<uint8>
TArray<uint8> SerializeLeaderSequence(const FLeaderSequencePayload& Payload);

// Desserialização da sequência do líder (chamada no lado Python)
bool DeserializeLeaderSequence(const TArray<uint8>& Data, FLeaderSequencePayload& Out);

// Payload de resposta com ação discreta (retornado pelo servidor Python)
struct FMotionActionPayload
{
    int64  SequenceId  = 0;
    int64  NPCId       = 0;
    int32  ActionIdx   = 0;
    FVector Direction  = FVector::ZeroVector;
    float  Speed       = 0.0f;
    float  Confidence  = 0.0f;
    float  LatencyMs   = 0.0f;
    bool   bValid      = false;
};

// Desserialização do MSG_MOTION_ACTION
bool DeserializeMotionAction(const TArray<uint8>& Data, FMotionActionPayload& Out);
*/

// ──────────────────────────────────────────────────────────────────────────────
// B. Adicionar a FCognitiveMotionRequest (CognitiveMotionTypes.h)
// ──────────────────────────────────────────────────────────────────────────────

/*
// Dentro da USTRUCT FCognitiveMotionRequest, adicionar APÓS os campos existentes:

    // Identificação do NPC que envia a requisição (0 = não identificado)
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cognitive|Identity")
    int64 NPCId = 0;

    // NPCId do líder a ser imitado (0 = nenhum / modo autônomo puro)
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cognitive|Identity")
    int64 TargetNPCId = 0;

    // Ação executada pelo NPC no último step (para aprendizado por reforço)
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cognitive|RL")
    int32 ActionTaken = 0;

    // Recompensa recebida no último step
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cognitive|RL")
    float Reward = 0.0f;
*/

// ──────────────────────────────────────────────────────────────────────────────
// C. Implementação em CognitiveMotionProtocol.cpp
// ──────────────────────────────────────────────────────────────────────────────

/*
// Adicionar ao FINAL de CognitiveMotionProtocol.cpp:

TArray<uint8> SerializeLeaderSequence(const FLeaderSequencePayload& Payload)
{
    TArray<uint8> P;

    // Header NPCIds e metadados
    WriteInt64(P, Payload.LeaderNPCId);
    WriteInt64(P, Payload.FollowerNPCId);
    WriteInt32(P, Payload.SequenceId);
    WriteDouble(P, Payload.StartTimestamp);
    WriteDouble(P, Payload.EndTimestamp);
    WriteInt32(P, Payload.Frames.Num());

    // Serializa cada frame usando a função existente SerializePoseFrame
    for (const FCognitivePoseFrame& Frame : Payload.Frames)
    {
        TArray<uint8> FrameData = SerializePoseFrame(Frame);
        WriteInt32(P, FrameData.Num());
        P.Append(FrameData);
    }

    return BuildFrame(EMessageType::LeaderSequence, Payload.SequenceId, P);
}

bool DeserializeLeaderSequence(const TArray<uint8>& Data, FLeaderSequencePayload& Out)
{
    if (Data.Num() < HeaderSize)
    {
        return false;
    }

    int32 O = HeaderSize;
    const uint8* D = Data.GetData();

    Out.LeaderNPCId    = ReadInt64(D, O);
    Out.FollowerNPCId  = ReadInt64(D, O);
    Out.SequenceId     = ReadInt32(D, O);
    Out.StartTimestamp = ReadDouble(D, O);
    Out.EndTimestamp   = ReadDouble(D, O);

    const int32 NumFrames = ReadInt32(D, O);
    Out.Frames.Reserve(NumFrames);

    for (int32 i = 0; i < NumFrames; ++i)
    {
        const int32 FrameSize = ReadInt32(D, O);
        if (O + FrameSize > Data.Num())
        {
            break;
        }
        TArray<uint8> FrameData(D + O, FrameSize);
        O += FrameSize;

        FCognitivePoseFrame Frame;
        if (DeserializePoseFrame(FrameData, Frame))
        {
            Out.Frames.Add(Frame);
        }
    }

    return Out.Frames.Num() > 0;
}

bool DeserializeMotionAction(const TArray<uint8>& Data, FMotionActionPayload& Out)
{
    if (Data.Num() < HeaderSize + 8 + 8 + 4 + 12 + 4 + 4 + 4 + 4)
    {
        return false;
    }

    int32 O = HeaderSize;
    const uint8* D = Data.GetData();

    Out.SequenceId = ReadInt64(D, O);
    Out.NPCId      = ReadInt64(D, O);
    Out.ActionIdx  = ReadInt32(D, O);

    FVector Dir;
    Dir.X         = ReadFloat(D, O);
    Dir.Y         = ReadFloat(D, O);
    Dir.Z         = ReadFloat(D, O);
    Out.Direction = Dir;
    Out.Speed      = ReadFloat(D, O);
    Out.Confidence = ReadFloat(D, O);
    Out.LatencyMs  = ReadFloat(D, O);
    Out.bValid     = (ReadInt32(D, O) != 0);

    return true;
}

// Adicionar ao enum EMessageType (se existir):
// LeaderSequence    = 0x05,
// AutonomousRequest = 0x06,
// MotionAction      = 0x07,
*/

// ──────────────────────────────────────────────────────────────────────────────
// D. Adicionar SendRawMessage ao UCognitiveInferenceSubsystem
// ──────────────────────────────────────────────────────────────────────────────

/*
// Em CognitiveInferenceSubsystem.h, adicionar:

    // Envia bytes arbitrários via TCP (para extensões de protocolo como MSG_LEADER_SEQUENCE)
    UFUNCTION(BlueprintCallable, Category = "Cognitive|Protocol")
    void SendRawMessage(const TArray<uint8>& Data);

// Em CognitiveInferenceSubsystem.cpp, implementar:

void UCognitiveInferenceSubsystem::SendRawMessage(const TArray<uint8>& Data)
{
    if (!Socket || !bConnected || Data.Num() == 0)
    {
        return;
    }

    int32 BytesSent = 0;
    Socket->Send(Data.GetData(), Data.Num(), BytesSent);

    if (BytesSent != Data.Num())
    {
        UE_LOG(LogTemp, Warning,
               TEXT("[CognitiveInferenceSubsystem] SendRawMessage: enviados %d/%d bytes"),
               BytesSent, Data.Num());
    }
}
*/
