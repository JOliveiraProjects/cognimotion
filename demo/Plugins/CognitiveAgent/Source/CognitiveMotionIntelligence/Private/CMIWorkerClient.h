#pragma once

#include "CoreMinimal.h"

/**
 * FCMIWorkerClient
 *
 * Cliente do processo de inferência isolado (cmi_worker.exe). Spawna o worker,
 * conecta no named pipe dele, e roda o forward enviando (h,z,action,obs,useObs)
 * e recebendo (h',z',actionIdx,pose). A LibTorch fica TODA no worker — nada de
 * torch dentro do processo do Unreal, eliminando o conflito de heap (0xC0000374).
 *
 * Uso:
 *   FCMIWorkerClient W;
 *   W.Start(ModelPath, 512, 1024, 9, 256);   // spawna worker + conecta
 *   W.Forward(h, z, action, obs, useObs, outH, outZ, outAct, outPose);
 *   W.Stop();                                 // fecha pipe → worker encerra
 *
 * Sem dependência de LibTorch no header nem no .cpp — só Win32 + UE.
 */
class FCMIWorkerClient
{
public:
    FCMIWorkerClient() = default;
    ~FCMIWorkerClient();

    /** Spawna o worker e conecta. Retorna true se o modelo carregou no worker. */
    bool Start(const FString& ModelPath, int32 Hidden, int32 Stoch,
               int32 Action, int32 Obs);

    /** Encerra: fecha o pipe (worker sai) e mata o processo se necessário. */
    void Stop();

    bool IsRunning() const { return bConnected; }

    /**
     * Roda um forward no worker.
     * @return true em sucesso; false se a comunicação ou o forward falharam
     *         (nesse caso o chamador deve cair no fallback).
     */
    bool Forward(const TArray<float>& H, const TArray<float>& Z,
                 const TArray<float>& Action, const TArray<float>& Obs,
                 bool bUseObs,
                 TArray<float>& OutH, TArray<float>& OutZ,
                 int32& OutActionIdx, TArray<float>& OutPose);

private:
    void* PipeHandle = nullptr;       // HANDLE do named pipe (void* p/ não vazar windows.h)
    void* ProcStore = nullptr;        // FProcHandle* alocado (evita vazar HAL no header)
    bool  bConnected = false;

    int32 Hidden = 512;
    int32 Stoch  = 1024;
    int32 Action = 9;
    int32 Obs    = 256;

    bool ReadExact(void* Dst, int32 Bytes);
    bool WriteExact(const void* Src, int32 Bytes);
};
