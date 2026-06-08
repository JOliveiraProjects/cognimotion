#include "CognitiveMotionAssetFactory.h"
#include "CognitiveRuntimeDatabaseManager.h"
#include "CognitiveTrajectoryGenerator.h"  // TECH DEBT FIX: needed for UCognitiveTrajectoryConfig

UCognitiveMotionDatabaseAssetFactory::UCognitiveMotionDatabaseAssetFactory()
{
    bCreateNew    = true;
    bEditAfterNew = true;
    SupportedClass = UCognitiveRuntimeDatabaseManager::StaticClass();
}

UObject* UCognitiveMotionDatabaseAssetFactory::FactoryCreateNew(
    UClass* InClass, UObject* InParent, FName InName,
    EObjectFlags Flags, UObject* Context, FFeedbackContext* Warn)
{
    return NewObject<UCognitiveRuntimeDatabaseManager>(InParent, InClass, InName, Flags);
}

FText UCognitiveMotionDatabaseAssetFactory::GetDisplayName() const
{
    return NSLOCTEXT("CMI", "DatabaseFactory", "Cognitive Motion Database");
}

UCognitiveMotionConfigAssetFactory::UCognitiveMotionConfigAssetFactory()
{
    bCreateNew    = true;
    bEditAfterNew = true;
    // TECH DEBT FIX: SupportedClass era UObject::StaticClass() — criava um UObject
    // genérico sem propriedades nem utilidade. Agora cria UCognitiveTrajectoryGenerator
    // que é o config UObject relevante com FCognitiveTrajectoryConfig editável.
    SupportedClass = UCognitiveTrajectoryGenerator::StaticClass();
}

UObject* UCognitiveMotionConfigAssetFactory::FactoryCreateNew(
    UClass* InClass, UObject* InParent, FName InName,
    EObjectFlags Flags, UObject* Context, FFeedbackContext* Warn)
{
    return NewObject<UCognitiveTrajectoryGenerator>(InParent, InClass, InName, Flags);
}

FText UCognitiveMotionConfigAssetFactory::GetDisplayName() const
{
    return NSLOCTEXT("CMI", "ConfigFactory", "Cognitive Motion Config");
}
