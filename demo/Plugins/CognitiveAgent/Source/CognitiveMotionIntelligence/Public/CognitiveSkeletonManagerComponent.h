#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "CognitiveSkeletonTypes.h"
#include "CognitiveSkeletonManagerComponent.generated.h"

class USkeletalMeshComponent;

/**
 * UCognitiveSkeletonManagerComponent
 *
 * Gerencia o SKELETON do NPC para o sistema Cognitive: detecta o mesh, lê os
 * bones reais, valida a compatibilidade com o tipo de skeleton escolhido e
 * expõe uma ASSINATURA estável usada para casar treino ↔ inferência.
 *
 * Hoje só o tipo Default (humanoide 89 bones) está ativo. Os demais tipos
 * existem como estrutura pronta para versões futuras (aparecem como "em breve").
 *
 * A assinatura gerada aqui é a MESMA lógica do lado Python
 * (world_model/skeleton_profile.py): num_bones + hash dos nomes. Assim, se você
 * tentar usar um modelo .pt treinado num skeleton diferente, o sistema detecta
 * e avisa — em vez de gerar movimento corrompido.
 *
 * Uso: o wizard adiciona este componente ao Blueprint do NPC automaticamente.
 */
UCLASS(ClassGroup=(Cognitive), meta=(BlueprintSpawnableComponent),
       DisplayName="Cognitive Skeleton Manager")
class COGNITIVEMOTIONINTELLIGENCE_API UCognitiveSkeletonManagerComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    UCognitiveSkeletonManagerComponent();

    // ── Configuração ──────────────────────────────────────────────────────────

    // Tipo de skeleton deste NPC. Hoje use Default; os demais são reservados.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Skeleton")
    ECognitiveSkeletonType SkeletonType = ECognitiveSkeletonType::Default;

    // Nº de bones esperado para o tipo Default. Deve casar com num_bones do
    // servidor Python (config). Padrão 89 (UE5 Mannequin).
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Skeleton",
              meta=(ClampMin="1"))
    int32 ExpectedBoneCount = 89;

    // Se verdadeiro, valida o skeleton automaticamente no BeginPlay e loga o
    // resultado (útil para descobrir cedo um Blueprint mal configurado).
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Cognitive|Skeleton")
    bool bValidateOnBeginPlay = true;

    // ── API ────────────────────────────────────────────────────────────────────

    // Valida o skeleton do mesh do dono contra o tipo/contagem esperados.
    // Preenche e retorna o resultado (usado pela UI visual e pelos logs).
    UFUNCTION(BlueprintCallable, Category="Cognitive|Skeleton")
    FCognitiveSkeletonValidation ValidateSkeleton();

    // Assinatura estável do skeleton atual ("Default:89:<hash>").
    // Mesma fórmula do lado Python para casar treino ↔ inferência.
    UFUNCTION(BlueprintCallable, Category="Cognitive|Skeleton")
    FString GetSkeletonSignature() const;

    // Nome do tipo como string ("Default", "Quadruped"...).
    UFUNCTION(BlueprintCallable, Category="Cognitive|Skeleton")
    FString GetSkeletonTypeName() const;

    // Verdadeiro se o tipo escolhido está ativo nesta versão (só Default hoje).
    UFUNCTION(BlueprintCallable, Category="Cognitive|Skeleton")
    bool IsSkeletonTypeAvailable() const;

    // Último resultado de validação (cacheado).
    UPROPERTY(BlueprintReadOnly, Category="Cognitive|Skeleton")
    FCognitiveSkeletonValidation LastValidation;

protected:
    virtual void BeginPlay() override;

private:
    // Acha o SkeletalMeshComponent do dono (mesmo critério do BoneDriver).
    USkeletalMeshComponent* ResolveMesh() const;

    // Lê os nomes de todos os bones do mesh.
    void ReadBoneNames(USkeletalMeshComponent* Mesh, TArray<FName>& OutNames) const;

    // Hash estável dos nomes (espelha o sha1[:12] do Python, de forma simples).
    static FString HashBoneNames(const TArray<FName>& Names);
};
