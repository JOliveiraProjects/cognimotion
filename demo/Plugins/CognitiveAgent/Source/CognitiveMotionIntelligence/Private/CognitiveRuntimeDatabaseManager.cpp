#include "CognitiveRuntimeDatabaseManager.h"

TObjectPtr<UCognitiveRuntimeDatabaseManager> UCognitiveRuntimeDatabaseManager::GlobalInstance = nullptr;

UCognitiveRuntimeDatabaseManager* UCognitiveRuntimeDatabaseManager::GetGlobal()
{
    return GlobalInstance;
}

void UCognitiveRuntimeDatabaseManager::SetGlobal(UCognitiveRuntimeDatabaseManager* Manager)
{
    // BM-07 FIX: TObjectPtr em static pode ser coletado pelo GC se não houver
    // outra referência forte (ex: nenhum UObject owner). AddToRoot() impede a
    // coleta; RemoveFromRoot() no Manager anterior evita leak.
    if (GlobalInstance)
    {
        GlobalInstance->RemoveFromRoot();
    }
    GlobalInstance = Manager;
    if (GlobalInstance)
    {
        GlobalInstance->AddToRoot();
    }
}

void UCognitiveRuntimeDatabaseManager::RegisterDatabase(const FCognitiveDatabaseEntry& Entry)
{
    Databases.Add(Entry);
}

UObject* UCognitiveRuntimeDatabaseManager::SelectDatabase(
    ECognitiveMovementMode MovementMode,
    ECognitiveMotionStyle  Style,
    float                  Speed) const
{
    UObject* BestDB    = nullptr;
    float                BestScore = -1.f;

    for (const FCognitiveDatabaseEntry& Entry : Databases)
    {
        if (!Entry.Database) continue;
        if (Speed < Entry.MinSpeed || Speed > Entry.MaxSpeed) continue;

        float Score = 0.f;
        if (Entry.MovementMode == MovementMode) Score += 2.f;
        if (Entry.Style == Style)               Score += 1.f;

        if (Score > BestScore)
        {
            BestScore = Score;
            BestDB    = Entry.Database;
        }
    }

    return BestDB ? BestDB : GetFallbackDatabase();
}

UObject* UCognitiveRuntimeDatabaseManager::GetFallbackDatabase() const
{
    return FallbackDatabase;
}

void UCognitiveRuntimeDatabaseManager::SetFallbackDatabase(UObject* DB)
{
    FallbackDatabase = DB;
}

bool UCognitiveRuntimeDatabaseManager::HasValidDatabase() const
{
    if (FallbackDatabase) return true;
    for (const FCognitiveDatabaseEntry& E : Databases)
        if (E.Database) return true;
    return false;
}
