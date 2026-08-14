#include "CMIWorkerClient.h"
#include "CognitiveDebugLog.h"
#include "Interfaces/IPluginManager.h"
#include "Misc/Paths.h"
#include "Misc/Guid.h"
#include "HAL/PlatformProcess.h"
#include "HAL/PlatformFileManager.h"

#include <Windows/AllowWindowsPlatformTypes.h>
#include <windows.h>

FCMIWorkerClient::~FCMIWorkerClient()
{
    Stop();
}

// ─────────────────────────────────────────────────────────────────────────────
bool FCMIWorkerClient::Start(const FString& ModelPath, int32 InHidden,
                             int32 InStoch, int32 InAction, int32 InObs)
{
    Hidden = InHidden; Stoch = InStoch; Action = InAction; Obs = InObs;

    // Localiza o cmi_worker.exe ao lado dos binários do plugin.
    FString WorkerExe;
    if (TSharedPtr<IPlugin> Plugin = IPluginManager::Get().FindPlugin(TEXT("CognitiveAgent")))
    {
        WorkerExe = FPaths::Combine(Plugin->GetBaseDir(),
                                    TEXT("Binaries"), TEXT("Win64"),
                                    TEXT("cmi_worker.exe"));
    }
    if (WorkerExe.IsEmpty() || !FPaths::FileExists(WorkerExe))
    {
        CMI_DBG("[WorkerClient] cmi_worker.exe NÃO encontrado em %s — inferência isolada indisponível.",
                *WorkerExe);
        return false;
    }

    // Nome de pipe único por instância (evita colisão entre múltiplos NPCs).
    const FString PipeName = FString::Printf(TEXT("cmi_infer_%d_%s"),
        FPlatformProcess::GetCurrentProcessId(),
        *FGuid::NewGuid().ToString(EGuidFormats::Digits).Left(8));

    // Args: <pipe_name> <model_path> <hidden> <stoch> <action> <obs>
    const FString Args = FString::Printf(TEXT("\"%s\" \"%s\" %d %d %d %d"),
        *PipeName, *ModelPath, Hidden, Stoch, Action, Obs);

    // Spawna o worker (oculto, sem janela de console).
    FProcHandle Proc = FPlatformProcess::CreateProc(
        *WorkerExe, *Args,
        /*bLaunchDetached*/ false, /*bLaunchHidden*/ true, /*bLaunchReallyHidden*/ true,
        nullptr, 0, nullptr, nullptr);

    if (!Proc.IsValid())
    {
        CMI_DBG("[WorkerClient] falha ao spawnar cmi_worker.exe");
        return false;
    }
    ProcStore = new FProcHandle(Proc);

    // Conecta no named pipe do worker. Ele cria o pipe ao iniciar; tentamos
    // por até ~5s (o worker precisa carregar a LibTorch e o .pt primeiro).
    const FString FullPipe = FString::Printf(TEXT("\\\\.\\pipe\\%s"), *PipeName);
    HANDLE hPipe = INVALID_HANDLE_VALUE;
    const double Deadline = FPlatformTime::Seconds() + 5.0;
    while (FPlatformTime::Seconds() < Deadline)
    {
        hPipe = CreateFileW(*FullPipe, GENERIC_READ | GENERIC_WRITE,
                            0, nullptr, OPEN_EXISTING, 0, nullptr);
        if (hPipe != INVALID_HANDLE_VALUE) break;
        if (GetLastError() == ERROR_PIPE_BUSY)
            WaitNamedPipeW(*FullPipe, 1000);
        else
            FPlatformProcess::Sleep(0.05f);  // pipe ainda não criado — espera
    }

    if (hPipe == INVALID_HANDLE_VALUE)
    {
        CMI_DBG("[WorkerClient] não conseguiu conectar ao pipe do worker (timeout).");
        Stop();
        return false;
    }
    PipeHandle = hPipe;

    // Handshake: 1 int32 indicando se o modelo carregou no worker.
    int32 Status = 0;
    if (!ReadExact(&Status, sizeof(Status)) || Status != 1)
    {
        CMI_DBG("[WorkerClient] worker reportou falha ao carregar o modelo (status=%d).", Status);
        Stop();
        return false;
    }

    bConnected = true;
    CMI_DBG("[WorkerClient] worker conectado e modelo carregado (pipe=%s).", *PipeName);
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
void FCMIWorkerClient::Stop()
{
    if (PipeHandle)
    {
        // Fechar o pipe sinaliza EOF ao worker, que encerra sozinho.
        CloseHandle(static_cast<HANDLE>(PipeHandle));
        PipeHandle = nullptr;
    }
    bConnected = false;

    if (ProcStore)
    {
        FProcHandle* Proc = static_cast<FProcHandle*>(ProcStore);
        // Dá um instante para o worker sair via EOF; se não, mata.
        if (FPlatformProcess::IsProcRunning(*Proc))
        {
            FPlatformProcess::Sleep(0.05f);
            if (FPlatformProcess::IsProcRunning(*Proc))
                FPlatformProcess::TerminateProc(*Proc);
        }
        FPlatformProcess::CloseProc(*Proc);
        delete Proc;
        ProcStore = nullptr;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
bool FCMIWorkerClient::ReadExact(void* Dst, int32 Bytes)
{
    if (!PipeHandle) return false;
    BYTE* P = static_cast<BYTE*>(Dst);
    int32 Total = 0;
    while (Total < Bytes)
    {
        DWORD Got = 0;
        if (!ReadFile(static_cast<HANDLE>(PipeHandle), P + Total,
                      (DWORD)(Bytes - Total), &Got, nullptr) || Got == 0)
            return false;
        Total += (int32)Got;
    }
    return true;
}

bool FCMIWorkerClient::WriteExact(const void* Src, int32 Bytes)
{
    if (!PipeHandle) return false;
    const BYTE* P = static_cast<const BYTE*>(Src);
    int32 Total = 0;
    while (Total < Bytes)
    {
        DWORD Put = 0;
        if (!WriteFile(static_cast<HANDLE>(PipeHandle), P + Total,
                       (DWORD)(Bytes - Total), &Put, nullptr) || Put == 0)
            return false;
        Total += (int32)Put;
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
bool FCMIWorkerClient::Forward(const TArray<float>& H, const TArray<float>& Z,
                               const TArray<float>& InAction, const TArray<float>& InObs,
                               bool bUseObs,
                               TArray<float>& OutH, TArray<float>& OutZ,
                               int32& OutActionIdx, TArray<float>& OutPose)
{
    if (!bConnected) return false;

    // ── Envia a requisição ───────────────────────────────────────────────────
    if (!WriteExact(H.GetData(),        Hidden * sizeof(float))) { bConnected = false; return false; }
    if (!WriteExact(Z.GetData(),        Stoch  * sizeof(float))) { bConnected = false; return false; }
    if (!WriteExact(InAction.GetData(), Action * sizeof(float))) { bConnected = false; return false; }
    if (!WriteExact(InObs.GetData(),    Obs    * sizeof(float))) { bConnected = false; return false; }
    int32 UseObs = bUseObs ? 1 : 0;
    if (!WriteExact(&UseObs, sizeof(UseObs))) { bConnected = false; return false; }

    // ── Lê a resposta ────────────────────────────────────────────────────────
    int32 Ok = 0;
    if (!ReadExact(&Ok, sizeof(Ok))) { bConnected = false; return false; }
    if (!Ok) return false;  // forward falhou no worker, mas o pipe ainda vive

    OutH.SetNumUninitialized(Hidden);
    OutZ.SetNumUninitialized(Stoch);
    if (!ReadExact(OutH.GetData(), Hidden * sizeof(float))) { bConnected = false; return false; }
    if (!ReadExact(OutZ.GetData(), Stoch  * sizeof(float))) { bConnected = false; return false; }
    if (!ReadExact(&OutActionIdx, sizeof(OutActionIdx)))    { bConnected = false; return false; }

    int32 PoseN = 0;
    if (!ReadExact(&PoseN, sizeof(PoseN))) { bConnected = false; return false; }
    OutPose.SetNumUninitialized(PoseN);
    if (PoseN > 0 && !ReadExact(OutPose.GetData(), PoseN * sizeof(float)))
    {
        bConnected = false; return false;
    }
    return true;
}

#include <Windows/HideWindowsPlatformTypes.h>
