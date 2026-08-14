#include "CognitiveInferenceSubsystem.h"
#include "CognitiveMotionProtocol.h"
#include "Common/TcpSocketBuilder.h"
#include "Engine/GameInstance.h"
#include "Engine/World.h"
#include "TimerManager.h"

FCognitiveWorkerThread::FCognitiveWorkerThread(
    const FString& InHost,
    int32 InPort,
    TQueue<TArray<uint8>, EQueueMode::Mpsc>* InSendQueue,
    TQueue<TArray<uint8>, EQueueMode::Mpsc>* InRecvQueue,
    TQueue<TArray<uint8>, EQueueMode::Mpsc>* InFireAndForgetQueue,
    TQueue<TArray<uint8>, EQueueMode::Mpsc>* InTeachingQueue,
    FThreadSafeCounter* InConnectionState,
    float InReconnectInterval)
    : Host(InHost)
    , Port(InPort)
    , ReconnectInterval(InReconnectInterval)
    , SendQueue(InSendQueue)
    , RecvQueue(InRecvQueue)
    , FireAndForgetQueue(InFireAndForgetQueue)
    , TeachingQueue(InTeachingQueue)
    , ConnectionState(InConnectionState)
{
    ReceiveBuffer.Reserve(65536);
}

bool FCognitiveWorkerThread::Init()
{
    return true;
}

bool FCognitiveWorkerThread::Connect()
{
    ConnectionState->Set((int32)ECognitiveInferenceState::Connecting);

    ISocketSubsystem* SS = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM);
    if (!SS) return false;

    Socket = SS->CreateSocket(NAME_Stream, TEXT("CognitiveMotion"), false);
    if (!Socket) return false;

    Socket->SetNonBlocking(false);
    Socket->SetNoDelay(true);

    TSharedRef<FInternetAddr> Addr = SS->CreateInternetAddr();
    bool bValid = false;
    Addr->SetIp(*Host, bValid);
    if (!bValid)
    {
        SS->DestroySocket(Socket);
        Socket = nullptr;
        return false;
    }
    Addr->SetPort(Port);

    if (!Socket->Connect(*Addr))
    {
        SS->DestroySocket(Socket);
        Socket = nullptr;
        return false;
    }

    return PerformHandshake();
}

bool FCognitiveWorkerThread::PerformHandshake()
{
    ConnectionState->Set((int32)ECognitiveInferenceState::Handshaking);

    const TArray<uint8> HsData = CognitiveMotionProtocol::BuildHandshake(42, 256);
    if (!SendData(HsData)) return false;

    TArray<uint8> AckData;
    if (!ReceiveFrame(AckData)) return false;
    if (!CognitiveMotionProtocol::ValidateHandshakeAck(AckData)) return false;

    bConnected.AtomicSet(true);
    ConnectionState->Set((int32)ECognitiveInferenceState::Ready);
    return true;
}

void FCognitiveWorkerThread::Disconnect()
{
    bConnected.AtomicSet(false);
    if (Socket)
    {
        Socket->Close();
        ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->DestroySocket(Socket);
        Socket = nullptr;
    }
    ConnectionState->Set((int32)ECognitiveInferenceState::Disconnected);
}

bool FCognitiveWorkerThread::SendData(const TArray<uint8>& Data)
{
    if (!Socket) return false;

    const int32 TotalSize = Data.Num();
    int32 Sent = 0;

    while (Sent < TotalSize)
    {
        int32 BytesSent = 0;
        if (!Socket->Send(Data.GetData() + Sent, TotalSize - Sent, BytesSent))
            return false;
        if (BytesSent <= 0) return false;
        Sent += BytesSent;
    }
    return true;
}

bool FCognitiveWorkerThread::ReceiveFrame(TArray<uint8>& OutData)
{
    if (!Socket) return false;

    // ReceiveTimeout: tempo máximo aguardando resposta do Python.
    // Python em CPU com DreamerV3 pode levar 200-500ms por request.
    // 5s causava disconnect quando a fila acumulava (5.9s de uptime nos logs).
    // 30s dá margem para CPU lento e picos de carga.
    static const FTimespan ReceiveTimeout = FTimespan::FromSeconds(30.0);

    const int32 HdrSize = CognitiveMotionProtocol::HeaderSize;
    TArray<uint8> HdrBuf;
    HdrBuf.SetNum(HdrSize);

    int32 Received = 0;
    while (Received < HdrSize)
    {
        if (bStopRequested) return false;

        // Aguarda dado disponível (timeout 5s) — evita bloqueio permanente
        if (!Socket->Wait(ESocketWaitConditions::WaitForRead, ReceiveTimeout))
            return false;  // timeout ou erro — desconecta e reconecta

        int32 BytesRead = 0;
        if (!Socket->Recv(HdrBuf.GetData() + Received, HdrSize - Received, BytesRead))
            return false;
        if (BytesRead <= 0) return false;
        Received += BytesRead;
    }

    const CognitiveMotionProtocol::FPacketHeader* Hdr =
        reinterpret_cast<const CognitiveMotionProtocol::FPacketHeader*>(HdrBuf.GetData());

    if (Hdr->Magic != CognitiveMotionProtocol::MagicHeader) return false;

    const int32 PayloadSize = static_cast<int32>(Hdr->PayloadSize);
    // Validação de PayloadSize: rejeita pacotes malformados antes de alocar.
    // Sem isso, um pacote com PayloadSize=0xFFFFFFFF causaria SetNum(2GB+) → OOM.
    static constexpr int32 MaxReasonablePayload = 8 * 1024 * 1024; // 8 MB
    if (PayloadSize < 0 || PayloadSize > MaxReasonablePayload) return false;
    OutData.SetNum(HdrSize + PayloadSize);
    FMemory::Memcpy(OutData.GetData(), HdrBuf.GetData(), HdrSize);

    Received = 0;
    while (Received < PayloadSize)
    {
        if (bStopRequested) return false;

        if (!Socket->Wait(ESocketWaitConditions::WaitForRead, ReceiveTimeout))
            return false;

        int32 BytesRead = 0;
        if (!Socket->Recv(OutData.GetData() + HdrSize + Received, PayloadSize - Received, BytesRead))
            return false;
        if (BytesRead <= 0) return false;
        Received += BytesRead;
    }
    return true;
}

uint32 FCognitiveWorkerThread::Run()
{
    while (!bStopRequested)
    {
        if (!bConnected)
        {
            if (!Connect())
            {
                ConnectionState->Set((int32)ECognitiveInferenceState::Error);
                FPlatformProcess::Sleep(ReconnectInterval);
                continue;
            }
        }

        // BA-02 FIX: Drena FireAndForgetQueue (LeaderSequence, Ping, etc.)
        // sem aguardar resposta — não afeta PendingRequestCount.
        TArray<uint8> FireForget;
        while (FireAndForgetQueue->Dequeue(FireForget))
        {
            if (!SendData(FireForget))
            {
                Disconnect();
                break;
            }
        }
        if (!bConnected) continue;

        TArray<uint8> Outgoing;
        if (SendQueue->Dequeue(Outgoing))
        {
            if (!SendData(Outgoing))
            {
                Disconnect();
                continue;
            }

            TArray<uint8> Incoming;
            if (!ReceiveFrame(Incoming))
            {
                Disconnect();
                continue;
            }
            // Roteia por tipo: respostas de ensino vão para a fila dedicada,
            // sem interferir no fluxo request/response do Learner (RecvQueue).
            using namespace CognitiveMotionProtocol;
            const bool bTeaching =
                Incoming.Num() >= HeaderSize &&
                reinterpret_cast<const FPacketHeader*>(Incoming.GetData())->MessageType
                    == (uint8)EMessageType::TeachingChoice;
            if (bTeaching && TeachingQueue)
                TeachingQueue->Enqueue(MoveTemp(Incoming));
            else
                RecvQueue->Enqueue(MoveTemp(Incoming));
        }
        else
        {
            FPlatformProcess::Sleep(0.001f);
        }
    }

    Disconnect();
    return 0;
}

void FCognitiveWorkerThread::Stop()
{
    bStopRequested.AtomicSet(true);
}

void FCognitiveWorkerThread::Exit()
{
    Disconnect();
}

void FCognitiveWorkerThread::ForceCloseSocket()
{
    if (Socket)
        Socket->Close();
}

void UCognitiveInferenceSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
    Super::Initialize(Collection);

    if (UWorld* World = GetGameInstance()->GetWorld())
    {
        FTimerDelegate Del;
        Del.BindUObject(this, &UCognitiveInferenceSubsystem::PollState);
        World->GetTimerManager().SetTimer(TickHandle, Del, 0.1f, true);
    }
}

void UCognitiveInferenceSubsystem::Deinitialize()
{
    if (UWorld* World = GetGameInstance() ? GetGameInstance()->GetWorld() : nullptr)
        World->GetTimerManager().ClearTimer(TickHandle);
    Disconnect();
    Super::Deinitialize();
}

void UCognitiveInferenceSubsystem::Connect(const FString& Host, int32 Port)
{
    Disconnect();

    WorkerRunnable = MakeUnique<FCognitiveWorkerThread>(
        Host, Port, &SendQueue, &RecvQueue, &FireAndForgetQueue,
        &TeachingChoiceQueue,
        &ConnectionStateCounter, ReconnectIntervalSeconds);

    WorkerThread = TUniquePtr<FRunnableThread>(
        FRunnableThread::Create(
            WorkerRunnable.Get(),
            TEXT("CognitiveMotionWorker"),
            0,
            TPri_BelowNormal));
}


void UCognitiveInferenceSubsystem::Disconnect()
{
    if (WorkerRunnable)
    {
        WorkerRunnable->Stop();
        WorkerRunnable->ForceCloseSocket();
    }
    if (WorkerThread)
    {
        WorkerThread->WaitForCompletion();
        WorkerThread.Reset();
    }
    WorkerRunnable.Reset();
    ConnectionStateCounter.Set((int32)ECognitiveInferenceState::Disconnected);
}


void UCognitiveInferenceSubsystem::SendRawMessage(const TArray<uint8>& Data)
{
    // BA-02 FIX: SendRawMessage é fire-and-forget (LeaderSequence, Ping, etc).
    // O worker thread envia o pacote mas NÃO aguarda resposta para este tipo —
    // o modelo request/response 1:1 não se aplica aqui.
    // Problema original: ao enfileirar via SendQueue sem incrementar PendingRequestCount,
    // quando o Python eventualmente enviar qualquer resposta (ou o próximo MotionResponse
    // ser processado), TryGetResponse decrementava PendingRequestCount para negativo,
    // corrompendo o flood control.
    //
    // Solução: fire-and-forget não entra na SendQueue (que implica wait response).
    // Envia diretamente via SendData na thread de chamada (game thread) usando
    // um canal dedicado ou, pragmaticamente, marca o pacote com Flags=0x0001
    // (no-reply) para que o worker não aguarde resposta.
    //
    // Implementação pragmática: enfileira com flag especial no buffer — o worker
    // verifica Flags no header e não entra no loop ReceiveFrame se Flags & 0x0001.
    // Por ora, usamos FireAndForgetQueue separado que o worker drena sem Recv.
    if (!IsReady() || Data.Num() == 0) return;
    FireAndForgetQueue.Enqueue(Data);
}

bool UCognitiveInferenceSubsystem::SendMotionRequest(const FCognitiveMotionRequest& Request)
{
    if (!IsReady()) return false;
    if (PendingRequestCount >= MaxPendingRequests) return false;

    const TArray<uint8> Data = CognitiveMotionProtocol::SerializeRequest(Request);
    SendQueue.Enqueue(Data);
    ++PendingRequestCount;
    return true;
}

bool UCognitiveInferenceSubsystem::TryGetResponse(FCognitiveMotionResponse& OutResponse)
{
    TArray<uint8> Data;
    if (!RecvQueue.Dequeue(Data)) return false;
    if (PendingRequestCount > 0) --PendingRequestCount;

    const bool bOk = CognitiveMotionProtocol::DeserializeResponse(Data, OutResponse);
    if (bOk) UpdateLatencyStats(OutResponse.LatencyMs);
    return bOk;
}

ECognitiveInferenceState UCognitiveInferenceSubsystem::GetConnectionState() const
{
    return (ECognitiveInferenceState)ConnectionStateCounter.GetValue();
}

bool UCognitiveInferenceSubsystem::IsReady() const
{
    return GetConnectionState() == ECognitiveInferenceState::Ready;
}

float UCognitiveInferenceSubsystem::GetAverageLatencyMs() const
{
    FScopeLock L(&LatencyLock);   // mutable — sem const_cast
    if (LatencyHistory.IsEmpty()) return 0.f;
    float Sum = 0.f;
    for (float V : LatencyHistory) Sum += V;
    return Sum / LatencyHistory.Num();
}

void UCognitiveInferenceSubsystem::PollState()
{
    const ECognitiveInferenceState Current = GetConnectionState();
    if (Current != LastReportedState)
    {
        LastReportedState = Current;
        OnConnectionStateChanged.Broadcast(Current);
    }
}

void UCognitiveInferenceSubsystem::UpdateLatencyStats(float LatencyMs)
{
    FScopeLock L(&LatencyLock);
    if (LatencyHistory.Num() < 64)
    {
        LatencyHistory.Add(LatencyMs);
    }
    else
    {
        LatencyHistory[LatencyRingHead] = LatencyMs;
        LatencyRingHead = (LatencyRingHead + 1) % 64;
    }
}

// ── CognitiveNPCBoneDriver API ────────────────────────────────────────────────

void UCognitiveInferenceSubsystem::Connect(
    const FString& Host, int32 Port, const TArray<FName>& InBoneNames)
{
    CachedBoneNames = InBoneNames;
    // Só conecta se Host válido — permite chamar apenas para armazenar bone names
    if (!Host.IsEmpty() && Port > 0)
        Connect(Host, Port);
}

bool UCognitiveInferenceSubsystem::SendBoneFrame(const FCognitiveBoneFrame& Frame)
{
    if (!IsReady() || !Frame.IsValid()) return false;
    // Serializa como PoseFrame simplificado e enfileira como fire-and-forget
    // O Python recebe via MSG_POSE_FRAME (0x03)
    FCognitivePoseFrame PF;
    PF.Timestamp      = Frame.Timestamp;
    PF.FrameIndex     = (int32)(Frame.SequenceId & 0x7FFFFFFF);
    PF.BoneTransforms = Frame.BoneTransforms;
    const TArray<uint8> Data = CognitiveMotionProtocol::SerializePoseFrame(PF);
    FireAndForgetQueue.Enqueue(Data);
    return true;
}

bool UCognitiveInferenceSubsystem::TryGetBoneResponse(FCognitiveBoneResponse& OutResponse)
{
    // Reutiliza a RecvQueue — converte FCognitiveMotionResponse em FCognitiveBoneResponse
    FCognitiveMotionResponse Resp;
    if (!TryGetResponse(Resp)) return false;
    OutResponse.bValid          = Resp.bValid;
    OutResponse.Confidence      = Resp.Embedding.Confidence;
    OutResponse.LatencyMs       = Resp.LatencyMs;
    OutResponse.SequenceId      = Resp.SequenceId;
    OutResponse.BoneTransforms  = Resp.BoneTransforms;
    OutResponse.bApplyRootMotion = false;
    return Resp.bValid && OutResponse.BoneTransforms.Num() > 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Treino & Ensino
// ─────────────────────────────────────────────────────────────────────────────
bool UCognitiveInferenceSubsystem::SendTrainingRegister(const FCognitiveTrainingEntry& Entry)
{
    if (!IsReady()) return false;
    using namespace CognitiveMotionProtocol;

    FTrainingRegisterWire W;
    W.TrainingType  = Entry.TrainingType;
    W.ReactionName  = Entry.ReactionName;
    W.AnimationPath = Entry.AnimationPath;
    W.Notes         = Entry.Notes;

    SendRawMessage(SerializeTrainingRegister(W));
    return true;
}

int64 UCognitiveInferenceSubsystem::SendTeachingScenario(const FCognitiveTeachingScenario& Scenario)
{
    if (!IsReady()) return 0;
    using namespace CognitiveMotionProtocol;

    FTeachingScenarioWire W;
    W.ScenarioId   = NextScenarioId++;
    W.TrainingType = Scenario.TrainingType;
    W.Description  = Scenario.Description;
    for (const FCognitiveScenarioEntity& E : Scenario.Entities)
    {
        FScenarioEntityWire EW;
        EW.Kind = E.Kind; EW.Count = E.Count;
        EW.FacingMe = E.FacingMe; EW.DistanceM = E.DistanceM;
        W.Entities.Add(MoveTemp(EW));
    }
    W.CandidateReactions = Scenario.CandidateReactions;

    // Vai pela SendQueue: o worker envia e aguarda UMA resposta (TeachingChoice),
    // que o roteamento por tipo despacha para a TeachingChoiceQueue.
    SendQueue.Enqueue(SerializeTeachingScenario(W));
    return W.ScenarioId;
}

bool UCognitiveInferenceSubsystem::TryGetTeachingChoice(FCognitiveTeachingChoice& OutChoice)
{
    using namespace CognitiveMotionProtocol;

    TArray<uint8> Data;
    if (!TeachingChoiceQueue.Dequeue(Data)) return false;

    FTeachingChoiceWire W;
    if (!DeserializeTeachingChoice(Data, W)) return false;

    OutChoice.ScenarioId     = W.ScenarioId;
    OutChoice.ChosenReaction = W.ChosenReaction;
    OutChoice.Confidence     = W.Confidence;
    OutChoice.Rationale      = W.Rationale;
    return true;
}

bool UCognitiveInferenceSubsystem::SendTeachingFeedback(const FCognitiveTeachingFeedback& Feedback)
{
    if (!IsReady()) return false;
    using namespace CognitiveMotionProtocol;

    FTeachingFeedbackWire W;
    W.ScenarioId         = Feedback.ScenarioId;
    W.bCorrect           = Feedback.bCorrect ? 1 : 0;
    W.ChosenReaction     = Feedback.ChosenReaction;
    W.SuggestedReactions = Feedback.SuggestedReactions;
    W.Comment            = Feedback.Comment;

    SendRawMessage(SerializeTeachingFeedback(W));
    return true;
}
