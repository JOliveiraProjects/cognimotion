#pragma once

#include "CoreMinimal.h"
#include "HAL/ThreadSafeBool.h"
#include "Subsystems/GameInstanceSubsystem.h"
#include "HAL/Runnable.h"
#include "HAL/RunnableThread.h"
#include "Sockets.h"
#include "CognitiveBoneTypes.h"
#include "SocketSubsystem.h"
#include "Containers/Queue.h"
#include "CognitiveMotionTypes.h"
#include "CognitiveInferenceSubsystem.generated.h"

UENUM(BlueprintType)
enum class ECognitiveInferenceState : uint8
{
    Disconnected,
    Connecting,
    Handshaking,
    Ready,
    Error
};

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FCognitiveConnectionStateDelegate, ECognitiveInferenceState, State);

class FCognitiveWorkerThread : public FRunnable
{
public:
    FCognitiveWorkerThread(
        const FString& InHost,
        int32 InPort,
        TQueue<TArray<uint8>, EQueueMode::Mpsc>* InSendQueue,
        TQueue<TArray<uint8>, EQueueMode::Mpsc>* InRecvQueue,
        TQueue<TArray<uint8>, EQueueMode::Mpsc>* InFireAndForgetQueue,  // BA-02 fix
        FThreadSafeCounter* InConnectionState,
        float InReconnectInterval);

    virtual bool   Init() override;
    virtual uint32 Run() override;
    virtual void   Stop() override;
    virtual void   Exit() override;
    void ForceCloseSocket();

private:
    bool  Connect();
    void  Disconnect();
    bool  SendData(const TArray<uint8>& Data);
    bool  ReceiveFrame(TArray<uint8>& OutData);
    bool  PerformHandshake();

    FString   Host;
    int32     Port;
    float     ReconnectInterval;

    TQueue<TArray<uint8>, EQueueMode::Mpsc>* SendQueue;
    TQueue<TArray<uint8>, EQueueMode::Mpsc>* RecvQueue;
    TQueue<TArray<uint8>, EQueueMode::Mpsc>* FireAndForgetQueue;  // BA-02 fix
    FThreadSafeCounter* ConnectionState;

    FSocket* Socket = nullptr;
    FThreadSafeBool bStopRequested{false};
    FThreadSafeBool bConnected{false};
    TArray<uint8>   ReceiveBuffer;
};

UCLASS()
class COGNITIVEMOTIONINTELLIGENCE_API UCognitiveInferenceSubsystem : public UGameInstanceSubsystem
{
    GENERATED_BODY()

public:
    virtual void Initialize(FSubsystemCollectionBase& Collection) override;
    virtual void Deinitialize() override;

    UFUNCTION(BlueprintCallable, Category = "Cognitive|Inference")
    void Connect(const FString& Host, int32 Port);

    // Overload para CognitiveNPCBoneDriver: Connect(Host, Port, BoneNames)
    void Connect(const FString& Host, int32 Port, const TArray<FName>& InBoneNames);

    UFUNCTION(BlueprintCallable, Category = "Cognitive|Inference")
    void Disconnect();


    UFUNCTION(BlueprintCallable, Category="Cognitive|Protocol")
    void SendRawMessage(const TArray<uint8>& Data);

    UFUNCTION(BlueprintCallable, Category = "Cognitive|Inference")
    bool SendMotionRequest(const FCognitiveMotionRequest& Request);

    UFUNCTION(BlueprintCallable, Category = "Cognitive|Inference")
    bool TryGetResponse(FCognitiveMotionResponse& OutResponse);

    // ── Bone frame API (usado por CognitiveNPCBoneDriver) ─────────────────────
    // Envia frame de bones do NPC ao Python via FireAndForgetQueue
    bool SendBoneFrame(const FCognitiveBoneFrame& Frame);

    // Tenta retirar resposta de bone transforms da RecvQueue
    bool TryGetBoneResponse(FCognitiveBoneResponse& OutResponse);

    UFUNCTION(BlueprintPure, Category = "Cognitive|Inference")
    ECognitiveInferenceState GetConnectionState() const;

    UFUNCTION(BlueprintPure, Category = "Cognitive|Inference")
    bool IsReady() const;

    UFUNCTION(BlueprintPure, Category = "Cognitive|Inference")
    float GetAverageLatencyMs() const;

    UPROPERTY(BlueprintAssignable, Category = "Cognitive|Inference")
    FCognitiveConnectionStateDelegate OnConnectionStateChanged;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cognitive|Inference")
    FString DefaultHost = TEXT("127.0.0.1");

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cognitive|Inference")
    int32 DefaultPort = 9000;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cognitive|Inference")
    float ReconnectIntervalSeconds = 3.f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cognitive|Inference")
    int32 MaxPendingRequests = 16;

private:
    void PollState();
    void UpdateLatencyStats(float LatencyMs);

    TUniquePtr<FCognitiveWorkerThread>  WorkerRunnable;
    TUniquePtr<FRunnableThread>         WorkerThread;

    TQueue<TArray<uint8>, EQueueMode::Mpsc> SendQueue;
    TQueue<TArray<uint8>, EQueueMode::Mpsc> RecvQueue;
    // BA-02 FIX: Fila separada para mensagens sem resposta (LeaderSequence, Ping).
    // O worker drena esta fila sem entrar no loop ReceiveFrame, evitando que
    // PendingRequestCount fique negativo.
    TQueue<TArray<uint8>, EQueueMode::Mpsc> FireAndForgetQueue;

    FThreadSafeCounter ConnectionStateCounter{(int32)ECognitiveInferenceState::Disconnected};

    ECognitiveInferenceState LastReportedState{ECognitiveInferenceState::Disconnected};

    TArray<float> LatencyHistory;
    mutable FCriticalSection LatencyLock;  // mutable: usado em GetAverageLatencyMs() const
    int32 LatencyRingHead = 0;             // ring buffer head para UpdateLatencyStats O(1)
    TArray<FName> CachedBoneNames;

    FTimerHandle   TickHandle;
    int32 PendingRequestCount{0};
};
