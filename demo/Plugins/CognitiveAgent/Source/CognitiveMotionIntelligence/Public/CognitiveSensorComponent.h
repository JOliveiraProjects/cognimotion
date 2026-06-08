#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "CognitiveMotionTypes.h"
#include "CognitiveSensorComponent.generated.h"

UCLASS(ClassGroup=(Cognitive), meta=(BlueprintSpawnableComponent))
class COGNITIVEMOTIONINTELLIGENCE_API UCognitiveSensorComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    UCognitiveSensorComponent();

    virtual void BeginPlay() override;
    virtual void TickComponent(float DeltaTime, ELevelTick TickType,
        FActorComponentTickFunction* ThisTickFunction) override;

    UFUNCTION(BlueprintCallable, Category = "Cognitive|Sensor")
    void BroadcastSoundEvent(const FVector& Origin, float Volume, float MaxRadius);

    UFUNCTION(BlueprintCallable, Category = "Cognitive|Sensor")
    void BroadcastExplosionEvent(const FVector& Origin, float Radius);

    UFUNCTION(BlueprintCallable, Category = "Cognitive|Sensor")
    void BroadcastAllyDeath(const FVector& Location);

    UFUNCTION(BlueprintPure, Category = "Cognitive|Sensor")
    bool CanSeeTarget(AActor* Target) const;

    UFUNCTION(BlueprintPure, Category = "Cognitive|Sensor")
    bool CanHearEvent(const FVector& EventOrigin, float Volume) const;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cognitive|Sensor|Vision")
    float VisionRange = 2000.f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cognitive|Sensor|Vision")
    float VisionAngleDegrees = 90.f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cognitive|Sensor|Vision")
    float PeripheralVisionAngle = 150.f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cognitive|Sensor|Vision")
    float PeripheralVisionRange = 600.f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cognitive|Sensor|Hearing")
    float HearingRadius = 1200.f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cognitive|Sensor|Hearing")
    float HearingFalloff = 1.f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cognitive|Sensor")
    float SensorTickInterval = 0.1f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cognitive|Sensor")
    TArray<TSubclassOf<AActor>> TargetClasses;

    UPROPERTY(BlueprintAssignable, Category = "Cognitive|Sensor")
    FCognitiveSensorEventDelegate OnSensorEventDetected;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Cognitive|Sensor|Debug")
    TArray<FCognitiveSensorData> RecentEvents;

private:
    void ScanVision();
    void CleanupOldEvents(float DeltaTime);

    float EventCleanupTimer = 0.f;
    static constexpr float MaxEventAge    = 5.f;
    static constexpr int32 MaxStoredEvents = 32;

    // BA-03 FIX: cache de atores para evitar GetAllActorsOfClass() O(n) por tick.
    // Refresh automático a cada 2s ou quando TargetClasses mudar.
    TArray<AActor*>              CachedActors;
    TArray<TSubclassOf<AActor>>  CachedTargetClasses;
    double                       CachedActorsTimestamp = -999.0;
};
