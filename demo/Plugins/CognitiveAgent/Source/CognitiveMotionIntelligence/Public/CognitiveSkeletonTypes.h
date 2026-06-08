#pragma once

#include "CoreMinimal.h"
#include "CognitiveSkeletonTypes.generated.h"

/**
 * Tipo de skeleton suportado pelo sistema Cognitive.
 * Hoje SOMENTE Default (humanoide UE5 Mannequin / MetaHuman via retarget) está
 * ativo. Os demais existem para deixar a estrutura pronta para versões futuras
 * — aparecem no editor como "(em breve)" e ainda não têm pipeline de treino.
 */
UENUM(BlueprintType)
enum class ECognitiveSkeletonType : uint8
{
    // Humanoide padrão — 89 bones, UE5 Mannequin/MetaHuman. ÚNICO ativo hoje.
    Default     UMETA(DisplayName="Humanoide (Default)"),
    // Reservados para expansão futura:
    Quadruped   UMETA(DisplayName="Quadrúpede (em breve)"),
    Creature    UMETA(DisplayName="Criatura/Monstro (em breve)"),
    Custom      UMETA(DisplayName="Personalizado (em breve)")
};

/**
 * Status de validação de um bone individual — usado pela UI de validação visual
 * (ponto verde = presente/compatível, vermelho = ausente, amarelo = inesperado).
 */
UENUM(BlueprintType)
enum class ECognitiveBoneStatus : uint8
{
    Ok          UMETA(DisplayName="Compatível"),       // bone esperado e presente
    Missing     UMETA(DisplayName="Ausente"),          // esperado mas não existe no mesh
    Unexpected  UMETA(DisplayName="Inesperado")        // presente mas fora do perfil esperado
};

/** Resultado da checagem de um bone (para exibição na UI). */
USTRUCT(BlueprintType)
struct FCognitiveBoneCheck
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category="Cognitive|Skeleton")
    FName BoneName;

    UPROPERTY(BlueprintReadOnly, Category="Cognitive|Skeleton")
    ECognitiveBoneStatus Status = ECognitiveBoneStatus::Ok;
};

/** Resultado completo da validação do skeleton de um mesh. */
USTRUCT(BlueprintType)
struct FCognitiveSkeletonValidation
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category="Cognitive|Skeleton")
    bool bIsCompatible = false;

    UPROPERTY(BlueprintReadOnly, Category="Cognitive|Skeleton")
    int32 NumBones = 0;

    UPROPERTY(BlueprintReadOnly, Category="Cognitive|Skeleton")
    int32 NumOk = 0;

    UPROPERTY(BlueprintReadOnly, Category="Cognitive|Skeleton")
    int32 NumMissing = 0;

    UPROPERTY(BlueprintReadOnly, Category="Cognitive|Skeleton")
    TArray<FCognitiveBoneCheck> Bones;

    // Resumo legível (ex.: "89/89 bones OK — compatível com 'Default'").
    UPROPERTY(BlueprintReadOnly, Category="Cognitive|Skeleton")
    FString Summary;
};
