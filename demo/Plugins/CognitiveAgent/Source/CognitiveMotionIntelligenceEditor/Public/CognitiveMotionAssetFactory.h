#pragma once

#include "CoreMinimal.h"
#include "Factories/Factory.h"
#include "CognitiveMotionAssetFactory.generated.h"

UCLASS()
class COGNITIVEMOTIONINTELLIGENCEEDITOR_API UCognitiveMotionDatabaseAssetFactory : public UFactory
{
    GENERATED_BODY()

public:
    UCognitiveMotionDatabaseAssetFactory();

    virtual UObject* FactoryCreateNew(UClass* InClass, UObject* InParent,
        FName InName, EObjectFlags Flags, UObject* Context, FFeedbackContext* Warn) override;
    virtual bool ShouldShowInNewMenu() const override { return true; }
    virtual FText GetDisplayName() const override;
    virtual FString GetDefaultNewAssetName() const override { return TEXT("CMI_Database"); }
};

UCLASS()
class COGNITIVEMOTIONINTELLIGENCEEDITOR_API UCognitiveMotionConfigAssetFactory : public UFactory
{
    GENERATED_BODY()

public:
    UCognitiveMotionConfigAssetFactory();

    virtual UObject* FactoryCreateNew(UClass* InClass, UObject* InParent,
        FName InName, EObjectFlags Flags, UObject* Context, FFeedbackContext* Warn) override;
    virtual bool ShouldShowInNewMenu() const override { return true; }
    virtual FText GetDisplayName() const override;
};
