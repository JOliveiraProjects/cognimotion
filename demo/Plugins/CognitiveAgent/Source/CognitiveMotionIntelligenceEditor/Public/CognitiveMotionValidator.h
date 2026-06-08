#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "CognitiveMotionTypes.h"
#include "CognitiveMotionValidator.generated.h"

USTRUCT(BlueprintType)
struct COGNITIVEMOTIONINTELLIGENCEEDITOR_API FCognitiveValidationResult
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly) bool  bPassed        = true;
    UPROPERTY(BlueprintReadOnly) TArray<FString> Errors;
    UPROPERTY(BlueprintReadOnly) TArray<FString> Warnings;
    UPROPERTY(BlueprintReadOnly) FString Summary;
};

UCLASS(BlueprintType)
class COGNITIVEMOTIONINTELLIGENCEEDITOR_API UCognitiveMotionValidator : public UObject
{
    GENERATED_BODY()

public:
    UFUNCTION(BlueprintCallable, Category = "Cognitive|Validation")
    static FCognitiveValidationResult ValidateActor(AActor* Actor);

    UFUNCTION(BlueprintCallable, Category = "Cognitive|Validation")
    static FCognitiveValidationResult ValidateAnimBlueprint(UAnimBlueprint* AnimBP);

    UFUNCTION(BlueprintCallable, Category = "Cognitive|Validation")
    static FCognitiveValidationResult ValidatePoseDatabase(UObject* PoseDB);

    UFUNCTION(BlueprintCallable, Category = "Cognitive|Validation")
    static FCognitiveValidationResult ValidateProtocolCompatibility(
        const FString& PythonHost, int32 PythonPort);

private:
    static void CheckComponent(AActor* Actor, TSubclassOf<UActorComponent> CompClass,
        const FString& CompName, FCognitiveValidationResult& Result);
};
