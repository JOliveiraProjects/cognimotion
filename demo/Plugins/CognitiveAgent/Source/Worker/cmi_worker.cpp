// ─────────────────────────────────────────────────────────────────────────────
// cmi_worker.exe — Worker de inferência LibTorch ISOLADO do processo do Unreal.
//
// MOTIVO: a LibTorch (torch_cpu.dll + libiomp5md.dll) e o Unreal (Mimalloc)
// redirecionam malloc/free de formas incompatíveis. Quando a LibTorch roda
// DENTRO do UnrealEditor.exe, o primeiro forward corrompe a heap (0xC0000374).
// Rodando a LibTorch num PROCESSO SEPARADO, os dois alocadores nunca se cruzam.
//
// PROTOCOLO (named pipe \\.\pipe\cmi_infer_<PID>, binário, little-endian):
//   Handshake inicial (worker→cliente), 1 int32:
//       status: 1 = modelo carregado OK | 0 = falha (worker encerra)
//   Loop de requisição (cliente→worker), por frame:
//       float32[HIDDEN]      h        (512)
//       float32[STOCH]       z        (1024)
//       float32[ACTION]      action   (9, one-hot)
//       float32[OBS]         obs      (256)
//       int32                useObs   (0/1)
//   Resposta (worker→cliente):
//       int32                ok       (1 OK | 0 falha no forward)
//       float32[HIDDEN]      h'       (512)
//       float32[STOCH]       z'       (1024)
//       int32                actIdx   (índice da ação 0..8)
//       int32                poseN    (nº de floats de pose, normalmente 623)
//       float32[poseN]       pose
//   Encerramento: cliente fecha o pipe → worker detecta EOF e sai.
//
// ARGS: cmi_worker.exe <pipe_name> <model_path> [hidden] [stoch] [action] [obs]
//
// Build: ver CMakeLists.txt (linka a MESMA LibTorch do plugin).
// ─────────────────────────────────────────────────────────────────────────────

// ── ORDEM DOS INCLUDES É CRÍTICA ─────────────────────────────────────────────
// A LibTorch DEVE ser incluída ANTES do <windows.h>. O windows.h define macros
// (min, max, small, e mexe em tokens) que corrompem os headers C++ da torch se
// vierem antes — gerando erros como 'std::std::c10::...' e C2589/C2059 nos
// headers Float8/BFloat16. Por isso: STL → torch → windows.h (com NOMINMAX).
#include <cstdio>
#include <cstdint>
#include <cstdarg>
#include <string>
#include <vector>

// LibTorch primeiro (sem nenhuma macro do windows.h interferindo).
#include <torch/script.h>
#include <torch/torch.h>

// Só DEPOIS o windows.h, com NOMINMAX para não definir macros min/max que
// conflitam com std::min/std::max usados pela STL/torch.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

// Dimensões padrão (podem vir por argv). Devem casar com o .pt.
static int HIDDEN = 512;
static int STOCH  = 1024;
static int ACTION = 9;
static int OBS    = 256;

// ── Log simples para stderr (capturável pelo UE se redirecionado) ────────────
static void WLOG(const char* fmt, ...)
{
    char buf[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    fprintf(stderr, "[cmi_worker] %s\n", buf);
    fflush(stderr);
}

// ── Leitura/escrita completas no pipe (lidam com leituras parciais) ──────────
static bool ReadExact(HANDLE pipe, void* dst, DWORD bytes)
{
    BYTE* p = static_cast<BYTE*>(dst);
    DWORD total = 0;
    while (total < bytes)
    {
        DWORD got = 0;
        if (!ReadFile(pipe, p + total, bytes - total, &got, nullptr) || got == 0)
            return false;  // EOF ou erro → cliente desconectou
        total += got;
    }
    return true;
}

static bool WriteExact(HANDLE pipe, const void* src, DWORD bytes)
{
    const BYTE* p = static_cast<const BYTE*>(src);
    DWORD total = 0;
    while (total < bytes)
    {
        DWORD put = 0;
        if (!WriteFile(pipe, p + total, bytes - total, &put, nullptr) || put == 0)
            return false;
        total += put;
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    if (argc < 3)
    {
        WLOG("uso: cmi_worker.exe <pipe_name> <model_path> [hidden stoch action obs]");
        return 2;
    }

    const std::string pipeName  = argv[1];
    const std::string modelPath = argv[2];
    if (argc >= 7)
    {
        HIDDEN = atoi(argv[3]);
        STOCH  = atoi(argv[4]);
        ACTION = atoi(argv[5]);
        OBS    = atoi(argv[6]);
    }

    WLOG("iniciando. pipe=%s model=%s dims=[h=%d z=%d a=%d o=%d]",
        pipeName.c_str(), modelPath.c_str(), HIDDEN, STOCH, ACTION, OBS);

    // ── Força single-thread: sem pool de threads OpenMP (evita qualquer
    //    instabilidade de inicialização; 1 NPC não precisa de paralelismo). ──
    try
    {
        torch::set_num_threads(1);
        torch::set_num_interop_threads(1);
    }
    catch (...) { /* set_num_interop só pode ser chamado 1x; ignore se já setado */ }

    // ── Carrega o modelo ─────────────────────────────────────────────────────
    torch::jit::script::Module module;
    bool loadOk = false;
    try
    {
        module = torch::jit::load(modelPath, torch::kCPU);
        module.eval();
        loadOk = true;
        WLOG("modelo carregado OK");
    }
    catch (const std::exception& e)
    {
        WLOG("FALHA ao carregar modelo: %s", e.what());
        loadOk = false;
    }

    // ── Cria o named pipe e espera o cliente (UE) conectar ───────────────────
    const std::string fullPipe = "\\\\.\\pipe\\" + pipeName;
    HANDLE hPipe = CreateNamedPipeA(
        fullPipe.c_str(),
        PIPE_ACCESS_DUPLEX,
        PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
        1,                       // 1 instância (1 worker por NPC)
        1 << 20, 1 << 20,        // buffers de saída/entrada (1 MB cada)
        0, nullptr);

    if (hPipe == INVALID_HANDLE_VALUE)
    {
        WLOG("CreateNamedPipe falhou (err=%lu)", GetLastError());
        return 3;
    }

    WLOG("aguardando conexão do cliente...");
    BOOL connected = ConnectNamedPipe(hPipe, nullptr)
                     ? TRUE : (GetLastError() == ERROR_PIPE_CONNECTED);
    if (!connected)
    {
        WLOG("ConnectNamedPipe falhou (err=%lu)", GetLastError());
        CloseHandle(hPipe);
        return 4;
    }
    WLOG("cliente conectado");

    // ── Handshake: informa se o modelo carregou ──────────────────────────────
    int32_t status = loadOk ? 1 : 0;
    WriteExact(hPipe, &status, sizeof(status));
    if (!loadOk)
    {
        WLOG("encerrando (modelo não carregou)");
        CloseHandle(hPipe);
        return 5;
    }

    // ── Estado recorrente (mantido no worker entre frames) ───────────────────
    // O cliente envia h e z a cada frame (mantém a fonte da verdade no UE),
    // então aqui apenas usamos os tensores recebidos.

    // Buffers reutilizáveis
    std::vector<float> inH(HIDDEN), inZ(STOCH), inA(ACTION), inO(OBS);

    WLOG("entrando no loop de inferência");
    uint64_t frame = 0;
    while (true)
    {
        // ── Lê a requisição ──────────────────────────────────────────────────
        if (!ReadExact(hPipe, inH.data(), HIDDEN * sizeof(float))) break;
        if (!ReadExact(hPipe, inZ.data(), STOCH  * sizeof(float))) break;
        if (!ReadExact(hPipe, inA.data(), ACTION * sizeof(float))) break;
        if (!ReadExact(hPipe, inO.data(), OBS    * sizeof(float))) break;
        int32_t useObs = 0;
        if (!ReadExact(hPipe, &useObs, sizeof(useObs))) break;

        // ── Monta tensores e roda o forward ──────────────────────────────────
        int32_t ok = 0;
        std::vector<float> outH, outZ, outPose;
        int32_t actIdx = 0;

        try
        {
            torch::NoGradGuard nograd;

            torch::Tensor H = torch::from_blob(inH.data(), {1, HIDDEN}, torch::kFloat32).clone();
            torch::Tensor Z = torch::from_blob(inZ.data(), {1, STOCH},  torch::kFloat32).clone();
            torch::Tensor A = torch::from_blob(inA.data(), {1, ACTION}, torch::kFloat32).clone();
            torch::Tensor O = torch::from_blob(inO.data(), {1, OBS},    torch::kFloat32).clone();

            std::vector<torch::jit::IValue> inputs;
            inputs.reserve(5);
            inputs.push_back(H);
            inputs.push_back(Z);
            inputs.push_back(A);
            inputs.push_back(O);
            inputs.push_back(useObs != 0);

            auto out = module.forward(inputs).toTuple();

            torch::Tensor hN = out->elements()[0].toTensor().contiguous().cpu();
            torch::Tensor zN = out->elements()[1].toTensor().contiguous().cpu();
            torch::Tensor aI = out->elements()[2].toTensor().contiguous().cpu();
            torch::Tensor pose = out->elements()[3].toTensor().to(torch::kCPU).contiguous();

            outH.assign(hN.data_ptr<float>(), hN.data_ptr<float>() + hN.numel());
            outZ.assign(zN.data_ptr<float>(), zN.data_ptr<float>() + zN.numel());
            outPose.assign(pose.data_ptr<float>(), pose.data_ptr<float>() + pose.numel());

            // action index pode vir como int64 ou float dependendo do export
            if (aI.scalar_type() == torch::kLong)
                actIdx = static_cast<int32_t>(aI.item<int64_t>());
            else
                actIdx = static_cast<int32_t>(aI.item<float>());

            ok = 1;
        }
        catch (const std::exception& e)
        {
            WLOG("forward falhou no frame %llu: %s", (unsigned long long)frame, e.what());
            ok = 0;
        }

        // ── Escreve a resposta ───────────────────────────────────────────────
        if (!WriteExact(hPipe, &ok, sizeof(ok))) break;
        if (ok)
        {
            int32_t poseN = static_cast<int32_t>(outPose.size());
            if (!WriteExact(hPipe, outH.data(),   HIDDEN * sizeof(float))) break;
            if (!WriteExact(hPipe, outZ.data(),   STOCH  * sizeof(float))) break;
            if (!WriteExact(hPipe, &actIdx,       sizeof(actIdx)))         break;
            if (!WriteExact(hPipe, &poseN,        sizeof(poseN)))          break;
            if (poseN > 0 &&
                !WriteExact(hPipe, outPose.data(), poseN * sizeof(float))) break;
        }
        ++frame;
    }

    WLOG("cliente desconectou — encerrando após %llu frames", (unsigned long long)frame);
    FlushFileBuffers(hPipe);
    DisconnectNamedPipe(hPipe);
    CloseHandle(hPipe);
    return 0;
}
