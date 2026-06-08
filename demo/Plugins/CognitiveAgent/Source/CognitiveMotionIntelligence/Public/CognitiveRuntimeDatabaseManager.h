#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "CognitiveMotionTypes.h"
#include "CognitiveRuntimeDatabaseManager.generated.h"

// UPoseSearchDatabase removido — plugin não usa mais Motion Matching.
// FCognitiveDatabaseEntry mantido como stub para não quebrar referências existentes.
// Campo Database agora é UObject* genérico.

USTRUCT(BlueprintType)
struct COGNITIVEMOTIONINTELLIGENCE_API FCognitiveDatabaseEntry
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite) ECognitiveMovementMode MovementMode = ECognitiveMovementMode::Walk;
    UPROPERTY(EditAnywhere, BlueprintReadWrite) ECognitiveMotionStyle  Style        = ECognitiveMotionStyle::Neutral;
    UPROPERTY(EditAnywhere, BlueprintReadWrite) float                  MinSpeed     = 0.f;
    UPROPERTY(EditAnywhere, BlueprintReadWrite) float                  MaxSpeed     = 600.f;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite) TObjectPtr<UObject>    Database;
};

UCLASS(BlueprintType)
class COGNITIVEMOTIONINTELLIGENCE_API UCognitiveRuntimeDatabaseManager : public UObject
{
    GENERATED_BODY()

public:
    static UCognitiveRuntimeDatabaseManager* GetGlobal();
    static void SetGlobal(UCognitiveRuntimeDatabaseManager* Manager);

    UFUNCTION(BlueprintCallable, Category = "Cognitive|Database")
    void RegisterDatabase(const FCognitiveDatabaseEntry& Entry);

    
    UFUNCTION(BlueprintCallable, Category = "Cognitive|Database")
    UObject* SelectDatabase(
        ECognitiveMovementMode MovementMode,
        ECognitiveMotionStyle  Style,
        float                  Speed) const;

    UFUNCTION(BlueprintCallable, Category = "Cognitive|Database")
    UObject* GetFallbackDatabase() const;

    UFUNCTION(BlueprintCallable, Category = "Cognitive|Database")
    void SetFallbackDatabase(UObject* DB);

    UFUNCTION(BlueprintPure, Category = "Cognitive|Database")
    bool HasValidDatabase() const;

    UFUNCTION(BlueprintPure, Category = "Cognitive|Database")
    int32 GetDatabaseCount() const { return Databases.Num(); }

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cognitive|Database")
    TArray<FCognitiveDatabaseEntry> Databases;

private:
    UPROPERTY()
    TObjectPtr<UObject> FallbackDatabase;

    static TObjectPtr<UCognitiveRuntimeDatabaseManager> GlobalInstance;
};
