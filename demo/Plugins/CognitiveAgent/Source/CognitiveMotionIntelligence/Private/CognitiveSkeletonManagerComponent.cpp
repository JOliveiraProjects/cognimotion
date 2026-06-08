#include "CognitiveSkeletonManagerComponent.h"
#include "CognitiveDebugLog.h"
#include "Components/SkeletalMeshComponent.h"
#include "GameFramework/Character.h"
#include "GameFramework/Actor.h"
#include "Containers/StringConv.h"   // FTCHARToUTF8

UCognitiveSkeletonManagerComponent::UCognitiveSkeletonManagerComponent()
{
    PrimaryComponentTick.bCanEverTick = false;
}

void UCognitiveSkeletonManagerComponent::BeginPlay()
{
    Super::BeginPlay();
    if (bValidateOnBeginPlay)
    {
        const FCognitiveSkeletonValidation V = ValidateSkeleton();
        if (V.bIsCompatible)
        {
            CMI_DBG("[SkeletonManager] %s", *V.Summary);
        }
        else
        {
            CMI_DBG("[SkeletonManager] ATENÇÃO: %s", *V.Summary);
        }
    }
}

USkeletalMeshComponent* UCognitiveSkeletonManagerComponent::ResolveMesh() const
{
    // Mesmo critério do BoneDriver: Character->GetMesh(), senão primeiro
    // SkeletalMeshComponent do ator.
    if (const ACharacter* Char = Cast<ACharacter>(GetOwner()))
    {
        if (USkeletalMeshComponent* M = Char->GetMesh())
            return M;
    }
    if (const AActor* Owner = GetOwner())
    {
        return Owner->FindComponentByClass<USkeletalMeshComponent>();
    }
    return nullptr;
}

void UCognitiveSkeletonManagerComponent::ReadBoneNames(
    USkeletalMeshComponent* Mesh, TArray<FName>& OutNames) const
{
    OutNames.Reset();
    if (!Mesh) return;
    const int32 N = Mesh->GetNumBones();
    OutNames.Reserve(N);
    for (int32 i = 0; i < N; ++i)
        OutNames.Add(Mesh->GetBoneName(i));
}

FString UCognitiveSkeletonManagerComponent::HashBoneNames(const TArray<FName>& Names)
{
    // FNV-1a de 64 bits sobre os nomes unidos por '|'. Determinístico e idêntico
    // ao lado Python (skeleton_profile.py usa o MESMO FNV-1a). Retorna 12 hex.
    uint64 Hash = 1469598103934665603ULL;       // offset basis
    const uint64 Prime = 1099511628211ULL;
    auto FeedByte = [&](uint8 B) { Hash ^= B; Hash *= Prime; };

    for (int32 i = 0; i < Names.Num(); ++i)
    {
        if (i > 0) FeedByte((uint8)'|');
        const FString S = Names[i].ToString();
        // Alimenta os bytes UTF-8 minúsculos (estável entre plataformas).
        FTCHARToUTF8 Utf8(*S.ToLower());
        const ANSICHAR* Data = (const ANSICHAR*)Utf8.Get();
        const int32 Len = Utf8.Length();
        for (int32 b = 0; b < Len; ++b) FeedByte((uint8)Data[b]);
    }
    return FString::Printf(TEXT("%012llx"), Hash & 0xFFFFFFFFFFFFULL);
}

FString UCognitiveSkeletonManagerComponent::GetSkeletonTypeName() const
{
    switch (SkeletonType)
    {
        case ECognitiveSkeletonType::Default:   return TEXT("Default");
        case ECognitiveSkeletonType::Quadruped: return TEXT("Quadruped");
        case ECognitiveSkeletonType::Creature:  return TEXT("Creature");
        default:                                return TEXT("Custom");
    }
}

bool UCognitiveSkeletonManagerComponent::IsSkeletonTypeAvailable() const
{
    // Só o humanoide Default está ativo nesta versão.
    return SkeletonType == ECognitiveSkeletonType::Default;
}

FString UCognitiveSkeletonManagerComponent::GetSkeletonSignature() const
{
    USkeletalMeshComponent* Mesh = ResolveMesh();
    TArray<FName> Names;
    ReadBoneNames(Mesh, Names);
    const int32 NumBones = Names.Num();
    // Formato: "<Tipo>:<num_bones>:<hash>" — idêntico ao Python.
    return FString::Printf(TEXT("%s:%d:%s"),
        *GetSkeletonTypeName(), NumBones, *HashBoneNames(Names));
}

FCognitiveSkeletonValidation UCognitiveSkeletonManagerComponent::ValidateSkeleton()
{
    FCognitiveSkeletonValidation Result;

    USkeletalMeshComponent* Mesh = ResolveMesh();
    if (!Mesh)
    {
        Result.bIsCompatible = false;
        Result.Summary = TEXT("Nenhum SkeletalMesh encontrado no ator. "
                              "Adicione um mesh ao Character.");
        LastValidation = Result;
        return Result;
    }

    TArray<FName> Names;
    ReadBoneNames(Mesh, Names);
    Result.NumBones = Names.Num();

    // Para o tipo Default: compatível se a contagem bate com o esperado.
    // (A validação por-bone fica simples aqui; a UI visual do Passo 3 mostrará
    //  cada bone com cor. Por ora marcamos todos como Ok e contamos.)
    const bool bCountOk = (Result.NumBones == ExpectedBoneCount);

    Result.Bones.Reserve(Names.Num());
    for (const FName& BoneName : Names)
    {
        FCognitiveBoneCheck Check;
        Check.BoneName = BoneName;
        Check.Status   = ECognitiveBoneStatus::Ok;
        Result.Bones.Add(Check);
        ++Result.NumOk;
    }

    if (!IsSkeletonTypeAvailable())
    {
        Result.bIsCompatible = false;
        Result.Summary = FString::Printf(
            TEXT("Tipo de skeleton '%s' ainda não está disponível nesta versão. "
                 "Use 'Default' (humanoide)."), *GetSkeletonTypeName());
    }
    else if (bCountOk)
    {
        Result.bIsCompatible = true;
        Result.Summary = FString::Printf(
            TEXT("%d/%d bones — compatível com '%s'. Assinatura: %s"),
            Result.NumBones, ExpectedBoneCount, *GetSkeletonTypeName(),
            *GetSkeletonSignature());
    }
    else
    {
        Result.bIsCompatible = false;
        Result.NumMissing = FMath::Abs(ExpectedBoneCount - Result.NumBones);
        Result.Summary = FString::Printf(
            TEXT("Contagem de bones diferente: mesh tem %d, esperado %d para '%s'. "
                 "O modelo de treino pode não encaixar. Verifique o skeleton."),
            Result.NumBones, ExpectedBoneCount, *GetSkeletonTypeName());
    }

    LastValidation = Result;
    return Result;
}
