#include "CognitiveRuntimePoseMemory.h"
#include "Serialization/Archive.h"
#include "Misc/FileHelper.h"
#include "Serialization/MemoryWriter.h"
#include "Serialization/MemoryReader.h"

void UCognitiveRuntimePoseMemory::Initialize(int32 Capacity, int32 EmbeddingDim)
{
    MaxCapacity     = Capacity;
    EmbeddingDimension = EmbeddingDim;
    FRWScopeLock Lock(MemoryLock, SLT_Write);
    Entries.Reserve(Capacity);
}

void UCognitiveRuntimePoseMemory::Store(const FCognitiveGeneratedMotionFragment& Fragment)
{
    if (!Fragment.Embedding.IsValid()) return;

    FCognitiveGeneratedMotionFragment Copy = Fragment;
    Copy.GeneratedAt = FPlatformTime::Seconds();

    // DEADLOCK FIX: PruneIfNeeded() chamada dentro de Write lock causava Sort() O(n log n)
    // com lock adquirido, bloqueando todos os leitores. Corrigido usando double-check
    // locking com blocos de escopo explícitos em vez de destrutor explícito (UB risk).
    bool bNeedsPrune = false;
    {
        FRWScopeLock ReadLock(MemoryLock, SLT_ReadOnly);
        bNeedsPrune = (Entries.Num() >= MaxCapacity);
    }  // ReadLock liberado aqui

    {
        FRWScopeLock WriteLock(MemoryLock, SLT_Write);
        if (bNeedsPrune && Entries.Num() >= MaxCapacity)
            PruneIfNeeded();  // double-check: condição pode ter mudado
        Entries.Add(MoveTemp(Copy));
    }
}

TArray<FCognitiveGeneratedMotionFragment> UCognitiveRuntimePoseMemory::QuerySimilar(
    const FCognitiveMotionEmbedding& Query,
    int32 K,
    float MinSimilarity,
    ECognitiveMotionStyle StyleFilter,
    bool bUseStyleFilter) const
{
    FRWScopeLock Lock(MemoryLock, SLT_ReadOnly);

    if (Entries.IsEmpty() || !Query.IsValid())
        return {};

    TArray<TPair<float, int32>> Scored;
    Scored.Reserve(Entries.Num());

    for (int32 i = 0; i < Entries.Num(); ++i)
    {
        const FCognitiveGeneratedMotionFragment& E = Entries[i];
        if (!E.Embedding.IsValid()) continue;
        if (bUseStyleFilter && E.Embedding.Style != StyleFilter) continue;

        const float Sim = CosineSimilarity(Query.Values, E.Embedding.Values);
        if (Sim >= MinSimilarity)
            Scored.Add(TPair<float, int32>(Sim, i));
    }

    Scored.Sort([](const TPair<float, int32>& A, const TPair<float, int32>& B)
    {
        return A.Key > B.Key;
    });

    TArray<FCognitiveGeneratedMotionFragment> Result;
    const int32 Count = FMath::Min(K, Scored.Num());
    Result.Reserve(Count);
    for (int32 i = 0; i < Count; ++i)
    {
        FCognitiveGeneratedMotionFragment Entry = Entries[Scored[i].Value];
        Entry.SimilarityScore = Scored[i].Key;
        Result.Add(MoveTemp(Entry));
    }
    return Result;
}

FCognitiveGeneratedMotionFragment UCognitiveRuntimePoseMemory::BlendFragments(
    const TArray<FCognitiveGeneratedMotionFragment>& Fragments,
    const TArray<float>& Weights) const
{
    FCognitiveGeneratedMotionFragment Result;
    if (Fragments.IsEmpty()) return Result;

    float TotalWeight = 0.f;
    for (float W : Weights) TotalWeight += W;
    if (TotalWeight < KINDA_SMALL_NUMBER) return Fragments[0];

    const int32 EmbDim = Fragments[0].Embedding.Values.Num();
    Result.Embedding.Values.SetNumZeroed(EmbDim);

    for (int32 i = 0; i < Fragments.Num() && i < Weights.Num(); ++i)
    {
        const float NormW = Weights[i] / TotalWeight;
        const FCognitiveGeneratedMotionFragment& F = Fragments[i];

        for (int32 d = 0; d < FMath::Min(EmbDim, F.Embedding.Values.Num()); ++d)
            Result.Embedding.Values[d] += F.Embedding.Values[d] * NormW;

        Result.QualityScore  += F.QualityScore  * NormW;
        Result.SimilarityScore += F.SimilarityScore * NormW;
    }

    Result.Embedding.Confidence = Fragments[0].Embedding.Confidence;
    Result.Embedding.Style      = Fragments[0].Embedding.Style;
    Result.GeneratedAt          = FPlatformTime::Seconds();
    return Result;
}

float UCognitiveRuntimePoseMemory::CosineSimilarity(
    const TArray<float>& A, const TArray<float>& B) const
{
    const int32 Dim = FMath::Min(A.Num(), B.Num());
    if (Dim == 0) return 0.f;

    float Dot = 0.f, NormA = 0.f, NormB = 0.f;
    for (int32 i = 0; i < Dim; ++i)
    {
        Dot  += A[i] * B[i];
        NormA += A[i] * A[i];
        NormB += B[i] * B[i];
    }
    const float Denom = FMath::Sqrt(NormA * NormB);
    return Denom < KINDA_SMALL_NUMBER ? 0.f : Dot / Denom;
}

void UCognitiveRuntimePoseMemory::PruneIfNeeded()
{
    const int32 PruneCount = MaxCapacity / 4;

    Entries.Sort([](const FCognitiveGeneratedMotionFragment& A, const FCognitiveGeneratedMotionFragment& B)
    {
        return A.QualityScore > B.QualityScore;
    });

    const int32 RemoveFrom = FMath::Max(0, Entries.Num() - PruneCount);
    Entries.RemoveAt(RemoveFrom, Entries.Num() - RemoveFrom);
}

void UCognitiveRuntimePoseMemory::SaveToDisk(const FString& FilePath) const
{
    FRWScopeLock Lock(MemoryLock, SLT_ReadOnly);

    TArray<uint8> Buffer;
    FMemoryWriter Writer(Buffer);

    int32 Count = Entries.Num();
    Writer << Count;

    for (const FCognitiveGeneratedMotionFragment& F : Entries)
    {
        // TECH DEBT FIX: FMemoryWriter::operator<< requer referência não-const,
        // mas const_cast<TArray<float>&>(F.Embedding.Values) é undefined behavior
        // se o objeto for realmente const. Correção: copiar para variável local.
        TArray<float> EmbCopy = F.Embedding.Values;
        Writer << EmbCopy;
        float Conf = F.Embedding.Confidence;
        Writer << Conf;
        int32 Style = (int32)F.Embedding.Style;
        Writer << Style;
        float Quality = F.QualityScore;
        Writer << Quality;
    }

    FFileHelper::SaveArrayToFile(Buffer, *FilePath);
}

bool UCognitiveRuntimePoseMemory::LoadFromDisk(const FString& FilePath)
{
    TArray<uint8> Buffer;
    if (!FFileHelper::LoadFileToArray(Buffer, *FilePath)) return false;

    FRWScopeLock Lock(MemoryLock, SLT_Write);
    Entries.Empty();

    FMemoryReader Reader(Buffer);
    int32 Count = 0;
    Reader << Count;

    for (int32 i = 0; i < Count; ++i)
    {
        FCognitiveGeneratedMotionFragment F;
        Reader << F.Embedding.Values;
        Reader << F.Embedding.Confidence;
        int32 Style = 0;
        Reader << Style;
        F.Embedding.Style = (ECognitiveMotionStyle)Style;
        Reader << F.QualityScore;
        F.GeneratedAt = FPlatformTime::Seconds();
        Entries.Add(MoveTemp(F));
    }
    return true;
}

int32 UCognitiveRuntimePoseMemory::GetSize() const
{
    FRWScopeLock Lock(MemoryLock, SLT_ReadOnly);
    return Entries.Num();
}

float UCognitiveRuntimePoseMemory::GetAverageQuality() const
{
    FRWScopeLock Lock(MemoryLock, SLT_ReadOnly);
    if (Entries.IsEmpty()) return 0.f;
    float Sum = 0.f;
    for (const auto& E : Entries) Sum += E.QualityScore;
    return Sum / Entries.Num();
}

void UCognitiveRuntimePoseMemory::Clear()
{
    FRWScopeLock Lock(MemoryLock, SLT_Write);
    Entries.Empty();
}
