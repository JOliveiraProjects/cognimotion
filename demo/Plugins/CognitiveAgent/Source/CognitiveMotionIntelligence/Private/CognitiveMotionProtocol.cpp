#include "CognitiveMotionProtocol.h"

namespace CognitiveMotionProtocol
{

void WriteFloat(TArray<uint8>& Out, float Value)
{
    uint8 Bytes[4];
    FMemory::Memcpy(Bytes, &Value, 4);
    Out.Append(Bytes, 4);
}

void WriteDouble(TArray<uint8>& Out, double Value)
{
    uint8 Bytes[8];
    FMemory::Memcpy(Bytes, &Value, 8);
    Out.Append(Bytes, 8);
}

void WriteInt32(TArray<uint8>& Out, int32 Value)
{
    const int32 BE = BYTESWAP_ORDER32(Value);
    uint8 Bytes[4];
    FMemory::Memcpy(Bytes, &BE, 4);
    Out.Append(Bytes, 4);
}

void WriteInt64(TArray<uint8>& Out, int64 Value)
{
    uint8 Bytes[8];
    FMemory::Memcpy(Bytes, &Value, 8);
    Out.Append(Bytes, 8);
}

void WriteVector(TArray<uint8>& Out, const FVector& V)
{
    WriteFloat(Out, (float)V.X);
    WriteFloat(Out, (float)V.Y);
    WriteFloat(Out, (float)V.Z);
}

void WriteQuat(TArray<uint8>& Out, const FQuat& Q)
{
    WriteFloat(Out, (float)Q.X);
    WriteFloat(Out, (float)Q.Y);
    WriteFloat(Out, (float)Q.Z);
    WriteFloat(Out, (float)Q.W);
}

void WriteTransform(TArray<uint8>& Out, const FTransform& T)
{
    WriteVector(Out, T.GetLocation());
    WriteQuat(Out, T.GetRotation());
    WriteVector(Out, T.GetScale3D());
}

void WriteFloatArray(TArray<uint8>& Out, const TArray<float>& Arr)
{
    WriteInt32(Out, Arr.Num());
    for (float V : Arr)
        WriteFloat(Out, V);
}

void WriteString(TArray<uint8>& Out, const FString& S)
{
    // BA-01 FIX: S.Len() retorna o número de caracteres TCHAR (UTF-16 no Windows),
    // NÃO o número de bytes UTF-8. Para strings com caracteres multibyte (acentos,
    // Unicode) isso causava truncamento silencioso porque menos bytes eram copiados
    // do que os produzidos por TCHAR_TO_UTF8.
    // Correção: usar FTCHARToUTF8 e ler o comprimento em bytes do conversor.
    FTCHARToUTF8 UTF8Converter(*S);
    const int32 ByteLen = UTF8Converter.Length();  // bytes UTF-8, correto
    WriteInt32(Out, ByteLen);
    Out.Append(reinterpret_cast<const uint8*>(UTF8Converter.Get()), ByteLen);
}

float ReadFloat(const uint8* D, int32& O)
{
    float V;
    FMemory::Memcpy(&V, D + O, 4);
    O += 4;
    return V;
}

double ReadDouble(const uint8* D, int32& O)
{
    double V;
    FMemory::Memcpy(&V, D + O, 8);
    O += 8;
    return V;
}

int32 ReadInt32(const uint8* D, int32& O)
{
    int32 V;
    FMemory::Memcpy(&V, D + O, 4);
    O += 4;
    return BYTESWAP_ORDER32(V);
}

int64 ReadInt64(const uint8* D, int32& O)
{
    int64 V;
    FMemory::Memcpy(&V, D + O, 8);
    O += 8;
    return V;
}

FVector ReadVector(const uint8* D, int32& O)
{
    const float X = ReadFloat(D, O);
    const float Y = ReadFloat(D, O);
    const float Z = ReadFloat(D, O);
    return FVector(X, Y, Z);
}

FQuat ReadQuat(const uint8* D, int32& O)
{
    const float X = ReadFloat(D, O);
    const float Y = ReadFloat(D, O);
    const float Z = ReadFloat(D, O);
    const float W = ReadFloat(D, O);
    return FQuat(X, Y, Z, W);
}

FTransform ReadTransform(const uint8* D, int32& O)
{
    const FVector Loc = ReadVector(D, O);
    const FQuat Rot   = ReadQuat(D, O);
    const FVector Scl = ReadVector(D, O);
    return FTransform(Rot, Loc, Scl);
}

TArray<float> ReadFloatArray(const uint8* D, int32& O)
{
    const int32 N = ReadInt32(D, O);
    TArray<float> Out;
    // Proteção contra N malformado/gigante: limita a um teto razoável. Sem isso,
    // um pacote corrompido com N = 0x7FFFFFFF causaria leitura fora dos limites.
    if (N <= 0 || N > 65536) return Out;
    Out.Reserve(N);
    for (int32 i = 0; i < N; ++i)
        Out.Add(ReadFloat(D, O));
    return Out;
}

// Versão segura: valida que há bytes suficientes antes de ler N floats.
TArray<float> ReadFloatArraySafe(const uint8* D, int32& O, int32 BufSize)
{
    const int32 N = ReadInt32(D, O);
    TArray<float> Out;
    if (N <= 0 || N > 65536) return Out;
    if (O + N * 4 > BufSize) return Out;  // não há bytes suficientes
    Out.Reserve(N);
    for (int32 i = 0; i < N; ++i)
        Out.Add(ReadFloat(D, O));
    return Out;
}

FString ReadString(const uint8* D, int32& O)
{
    const int32 Len = ReadInt32(D, O);
    if (Len <= 0) return FString();
    // Cria string a partir de bytes UTF-8 com comprimento explícito.
    // FString(UTF8_TO_TCHAR(...)) usava construtor null-terminated — se o buffer
    // contivesse null-bytes antes da posição Len, a string seria truncada.
    // FUTF8ToTCHAR com comprimento evita essa dependência.
    FUTF8ToTCHAR Converter(reinterpret_cast<const ANSICHAR*>(D + O), Len);
    FString Out(Converter.Length(), Converter.Get());
    O += Len;
    return Out;
}

uint32 ComputeChecksum(const uint8* Data, int32 Size)
{
    uint32 CRC = 0xFFFFFFFF;
    for (int32 i = 0; i < Size; ++i)
    {
        CRC ^= Data[i];
        for (int32 j = 0; j < 8; ++j)
            CRC = (CRC >> 1) ^ (0xEDB88320 & -(int32)(CRC & 1));
    }
    return CRC ^ 0xFFFFFFFF;
}

static TArray<uint8> BuildFrame(EMessageType MsgType, int64 SeqId, const TArray<uint8>& Payload)
{
    TArray<uint8> Out;
    Out.Reserve(HeaderSize + Payload.Num());

    FPacketHeader Hdr;
    Hdr.Magic       = MagicHeader;
    Hdr.Version     = ProtocolVersion;
    Hdr.MessageType = (uint8)MsgType;
    Hdr.Flags       = 0;
    Hdr.PayloadSize = Payload.Num();
    Hdr.SequenceId  = SeqId;
    Hdr.Checksum    = Payload.Num() > 0
        ? ComputeChecksum(Payload.GetData(), Payload.Num())
        : 0;

    Out.Append((uint8*)&Hdr, HeaderSize);
    Out.Append(Payload);
    return Out;
}

static void SerializeTrajectorySample(TArray<uint8>& Out, const FCognitiveTrajectorySample& S)
{
    WriteVector(Out, S.Position);
    WriteVector(Out, S.LinearVelocity);
    WriteVector(Out, S.AngularVelocity);
    WriteQuat(Out, S.Facing);
    WriteFloat(Out, S.TimeInSeconds);
    WriteFloat(Out, S.Speed);
}

static void SerializeTrajectory(TArray<uint8>& Out, const FCognitiveTrajectory& T)
{
    WriteInt32(Out, T.Samples.Num());
    for (const FCognitiveTrajectorySample& S : T.Samples)
        SerializeTrajectorySample(Out, S);
}

TArray<uint8> SerializePoseFrame(const FCognitivePoseFrame& Frame)
{
    TArray<uint8> P;
    P.Reserve(4096);

    WriteDouble(P, Frame.Timestamp);
    WriteInt32(P, Frame.FrameIndex);
    WriteVector(P, Frame.LinearVelocity);
    WriteVector(P, Frame.AngularVelocity);
    WriteVector(P, Frame.RootLocation);
    WriteQuat(P, Frame.RootRotation);
    WriteInt32(P, Frame.BoneTransforms.Num());
    for (const FTransform& T : Frame.BoneTransforms)
        WriteTransform(P, T);
    SerializeTrajectory(P, Frame.PastTrajectory);
    SerializeTrajectory(P, Frame.FutureTrajectory);
    WriteInt32(P, Frame.CurveValues.Num());
    for (const auto& KV : Frame.CurveValues)
    {
        WriteString(P, KV.Key.ToString());
        WriteFloat(P, KV.Value);
    }
    WriteInt32(P, Frame.Tags.Num());
    for (const FName& Tag : Frame.Tags)
        WriteString(P, Tag.ToString());
    WriteInt32(P, (int32)Frame.MovementMode);
    WriteInt32(P, (int32)Frame.MotionStyle);

    return BuildFrame(EMessageType::PoseFrame, Frame.FrameIndex, P);
}

bool DeserializePoseFrame(const TArray<uint8>& Data, FCognitivePoseFrame& Out)
{
    if (Data.Num() < HeaderSize) return false;

    const FPacketHeader* Hdr = reinterpret_cast<const FPacketHeader*>(Data.GetData());
    if (Hdr->Magic != MagicHeader) return false;
    if ((EMessageType)Hdr->MessageType != EMessageType::PoseFrame) return false;

    int32 O = HeaderSize;
    const uint8* D = Data.GetData();

    Out.Timestamp        = ReadDouble(D, O);
    Out.FrameIndex       = ReadInt32(D, O);
    Out.LinearVelocity   = ReadVector(D, O);
    Out.AngularVelocity  = ReadVector(D, O);
    Out.RootLocation     = ReadVector(D, O);
    Out.RootRotation     = ReadQuat(D, O);

    const int32 NBones = ReadInt32(D, O);
    if (NBones > 0 && NBones <= 4096 && O + NBones * 40 <= Data.Num())
    {
        Out.BoneTransforms.SetNum(NBones);
        for (int32 i = 0; i < NBones; ++i)
            Out.BoneTransforms[i] = ReadTransform(D, O);
    }

    auto ReadTraj = [&](FCognitiveTrajectory& T)
    {
        const int32 N = ReadInt32(D, O);
        if (N <= 0 || N > 4096 || O + N * 60 > Data.Num()) return;
        T.Samples.SetNum(N);
        for (int32 i = 0; i < N; ++i)
        {
            T.Samples[i].Position      = ReadVector(D, O);
            T.Samples[i].LinearVelocity = ReadVector(D, O);
            T.Samples[i].AngularVelocity = ReadVector(D, O);
            T.Samples[i].Facing        = ReadQuat(D, O);
            T.Samples[i].TimeInSeconds = ReadFloat(D, O);
            T.Samples[i].Speed         = ReadFloat(D, O);
        }
    };
    ReadTraj(Out.PastTrajectory);
    ReadTraj(Out.FutureTrajectory);

    const int32 NCurves = ReadInt32(D, O);
    for (int32 i = 0; i < NCurves; ++i)
    {
        const FName Key(*ReadString(D, O));
        const float Val = ReadFloat(D, O);
        Out.CurveValues.Add(Key, Val);
    }

    const int32 NTags = ReadInt32(D, O);
    Out.Tags.SetNum(NTags);
    for (int32 i = 0; i < NTags; ++i)
        Out.Tags[i] = FName(*ReadString(D, O));

    Out.MovementMode = (ECognitiveMovementMode)ReadInt32(D, O);
    Out.MotionStyle  = (ECognitiveMotionStyle)ReadInt32(D, O);
    return true;
}

TArray<uint8> SerializeRequest(const FCognitiveMotionRequest& Req)
{
    TArray<uint8> P;
    P.Reserve(8192);

    WriteInt64(P, Req.SequenceId);
    WriteInt32(P, (int32)Req.RequestedStyle);
    WriteFloat(P, Req.MaxLatencyMs);

    const TArray<uint8> PoseData = SerializePoseFrame(Req.CurrentPose);
    WriteInt32(P, PoseData.Num());
    P.Append(PoseData);

    SerializeTrajectory(P, Req.DesiredTrajectory);

    WriteFloat(P, Req.Blackboard.Health);
    WriteFloat(P, Req.Blackboard.Stamina);
    WriteFloat(P, Req.Blackboard.Alertness);
    WriteFloat(P, Req.Blackboard.FearLevel);
    WriteFloat(P, Req.Blackboard.AggressionLevel);
    WriteFloat(P, Req.Blackboard.ThreatLevel);
    WriteInt32(P, (int32)Req.Blackboard.CurrentState);
    WriteInt32(P, (int32)Req.Blackboard.EmotionalState);

    return BuildFrame(EMessageType::MotionRequest, Req.SequenceId, P);
}

bool DeserializeResponse(const TArray<uint8>& Data, FCognitiveMotionResponse& Out)
{
    if (Data.Num() < HeaderSize) return false;

    const FPacketHeader* Hdr = reinterpret_cast<const FPacketHeader*>(Data.GetData());
    if (Hdr->Magic != MagicHeader) return false;
    if ((EMessageType)Hdr->MessageType != EMessageType::MotionResponse) return false;

    int32 O = HeaderSize;
    const uint8* D = Data.GetData();

    Out.SequenceId     = ReadInt64(D, O);
    Out.SelectedStyle  = (ECognitiveMotionStyle)ReadInt32(D, O);
    Out.LatencyMs      = ReadFloat(D, O);
    Out.bValid         = ReadInt32(D, O) != 0;

    Out.Embedding.Confidence = ReadFloat(D, O);
    Out.Embedding.Values     = ReadFloatArraySafe(D, O, Data.Num());
    Out.Embedding.Style      = Out.SelectedStyle;
    Out.Embedding.SequenceId = Out.SequenceId;

    // Trajetória: valida N contra o buffer. Cada sample = 3 vetores(36) + quat(16)
    // + 2 floats(8) = 60 bytes. Sem essa checagem, um N malformado lê fora.
    int32 N = 0;
    if (O + 4 <= Data.Num())
    {
        N = ReadInt32(D, O);
        if (N < 0 || N > 4096 || O + N * 60 > Data.Num())
            N = 0;  // pacote inconsistente — ignora a trajetória
    }
    Out.RefinedTrajectory.Samples.SetNum(N);
    for (int32 i = 0; i < N; ++i)
    {
        Out.RefinedTrajectory.Samples[i].Position        = ReadVector(D, O);
        Out.RefinedTrajectory.Samples[i].LinearVelocity  = ReadVector(D, O);
        Out.RefinedTrajectory.Samples[i].AngularVelocity = ReadVector(D, O);
        Out.RefinedTrajectory.Samples[i].Facing          = ReadQuat(D, O);
        Out.RefinedTrajectory.Samples[i].TimeInSeconds   = ReadFloat(D, O);
        Out.RefinedTrajectory.Samples[i].Speed           = ReadFloat(D, O);
    }

    // Bone transforms gerados pelo Python para todos os bones do skeleton do NPC.
    // Python envia: BoneCount(4 BE) + N × (Location(12) + Rotation(16) + Scale(12)) = N × 40 bytes
    if (O + 4 <= Data.Num())
    {
        const int32 NBones = ReadInt32(D, O);
        // Teto de segurança: skeletons reais têm centenas de bones, não milhões.
        if (NBones > 0 && NBones <= 4096)
        {
            Out.BoneTransforms.SetNum(NBones);
            for (int32 i = 0; i < NBones && O + 40 <= Data.Num(); ++i)
            {
                Out.BoneTransforms[i] = ReadTransform(D, O);
            }
        }
    }

    // PhysicalState (4 BE) — opcional; clientes/servidores antigos não enviam.
    if (O + 4 <= Data.Num())
    {
        const int32 PS = ReadInt32(D, O);
        Out.PhysicalState = (ECognitivePhysicalState)FMath::Clamp(PS, 0, 9);
    }
    return true;
}

TArray<uint8> BuildHandshake(int32 ObsDim, int32 EmbeddingDim)
{
    TArray<uint8> P;
    WriteInt32(P, ObsDim);
    WriteInt32(P, EmbeddingDim);
    WriteInt32(P, ProtocolVersion);
    return BuildFrame(EMessageType::Handshake, 0, P);
}

bool ValidateHandshakeAck(const TArray<uint8>& Data)
{
    if (Data.Num() < HeaderSize) return false;
    const FPacketHeader* Hdr = reinterpret_cast<const FPacketHeader*>(Data.GetData());
    return Hdr->Magic == MagicHeader
        && (EMessageType)Hdr->MessageType == EMessageType::HandshakeAck
        && Hdr->Version == ProtocolVersion;
}

TArray<uint8> SerializeLeaderSequence(const FLeaderSequencePayload& Payload)
{
    TArray<uint8> P;
    P.Reserve(Payload.Frames.Num() * 2048 + 64);

    // Wire layout — espelhado por parse_leader_sequence no Python (CP-01 fix):
    //   LeaderNPCId    (8 LE) + FollowerNPCId (8 LE) + SequenceId  (4 BE)
    //   StartTimestamp (8 LE) + EndTimestamp  (8 LE) + NumFrames   (4 BE)
    //   [FrameSize(4 BE) + FrameData]*
    WriteInt64(P, Payload.LeaderNPCId);
    WriteInt64(P, Payload.FollowerNPCId);
    WriteInt32(P, Payload.SequenceId);
    WriteDouble(P, Payload.StartTimestamp);
    WriteDouble(P, Payload.EndTimestamp);
    WriteInt32(P, Payload.Frames.Num());

    for (const FCognitivePoseFrame& Frame : Payload.Frames)
    {
        const TArray<uint8> FrameData = SerializePoseFrame(Frame);
        WriteInt32(P, FrameData.Num());
        P.Append(FrameData);
    }

    // BC-04 FIX: usa BuildFrame (helper interno) em vez de construção manual
    // de header — elimina duplicação de lógica e garante consistência com
    // os demais tipos de mensagem.
    return BuildFrame(EMessageType::LeaderSequence,
                      static_cast<int64>(Payload.SequenceId), P);
}

// ─────────────────────────────────────────────────────────────────────────────
// Perception (0x08) — NPC envia as entidades que percebe.
// Wire: NPCId(8 LE) + NumEntities(4 BE) + [ Category(4 BE) Disposition(4 BE)
//       Reaction(4 BE) VehicleType(4 BE) TrafficState(4 BE) Distance(4 LE)
//       DirX(4 LE) DirY(4 LE) DirZ(4 LE) ThreatWeight(4 LE) ]*
// Espelhado por parse_perception no Python.
// ─────────────────────────────────────────────────────────────────────────────
TArray<uint8> SerializePerception(const FPerceptionPayload& Payload)
{
    TArray<uint8> P;
    P.Reserve(16 + Payload.Entities.Num() * 40);

    WriteInt64(P, Payload.NPCId);
    WriteInt32(P, Payload.Entities.Num());

    for (const FPerceivedEntityWire& E : Payload.Entities)
    {
        WriteInt32(P, E.Category);
        WriteInt32(P, E.Disposition);
        WriteInt32(P, E.Role);
        WriteInt32(P, E.Reaction);
        WriteInt32(P, E.VehicleType);
        WriteInt32(P, E.TrafficState);
        WriteFloat(P, E.Distance);
        WriteFloat(P, E.DirX);
        WriteFloat(P, E.DirY);
        WriteFloat(P, E.DirZ);
        WriteFloat(P, E.ThreatWeight);
    }

    return BuildFrame(EMessageType::Perception, Payload.NPCId, P);
}

// ─────────────────────────────────────────────────────────────────────────────
// Teach (0x09) — líder ensina o vocabulário de ações.
// Wire: LeaderNPCId(8 LE) + CurrentVerb(4 BE) + LeaderCategory(4 BE)
//       + NumActions(4 BE)
//       + [ Verb(4 BE) ActionIndex(4 BE) TargetCategory(4 BE) Label(string) ]*
// Espelhado por parse_teach no Python.
// ─────────────────────────────────────────────────────────────────────────────
TArray<uint8> SerializeTeach(const FTeachPayload& Payload)
{
    TArray<uint8> P;
    P.Reserve(64 + Payload.Vocabulary.Num() * 64);

    WriteInt64(P, Payload.LeaderNPCId);
    WriteInt32(P, Payload.CurrentVerb);
    WriteInt32(P, Payload.LeaderCategory);
    WriteInt32(P, Payload.Vocabulary.Num());

    for (const FTaughtActionWire& A : Payload.Vocabulary)
    {
        WriteInt32(P, A.Verb);
        WriteInt32(P, A.ActionIndex);
        WriteInt32(P, A.TargetCategory);
        WriteString(P, A.Label);
    }

    return BuildFrame(EMessageType::Teach, Payload.LeaderNPCId, P);
}


}
