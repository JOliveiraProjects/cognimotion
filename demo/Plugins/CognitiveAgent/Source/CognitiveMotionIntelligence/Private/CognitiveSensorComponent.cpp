#include "CognitiveSensorComponent.h"
#include "GameFramework/Actor.h"
#include "Engine/World.h"
#include "DrawDebugHelpers.h"
#include "Kismet/GameplayStatics.h"

UCognitiveSensorComponent::UCognitiveSensorComponent()
{
    PrimaryComponentTick.bCanEverTick = true;
    PrimaryComponentTick.TickInterval = SensorTickInterval;
    RecentEvents.Reserve(MaxStoredEvents);
}

void UCognitiveSensorComponent::BeginPlay()
{
    Super::BeginPlay();
    PrimaryComponentTick.TickInterval = SensorTickInterval;
}

void UCognitiveSensorComponent::TickComponent(
    float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* Func)
{
    Super::TickComponent(DeltaTime, TickType, Func);

    ScanVision();
    CleanupOldEvents(DeltaTime);
}

bool UCognitiveSensorComponent::CanSeeTarget(AActor* Target) const
{
    if (!Target || !GetOwner() || !GetWorld()) return false;

    const FVector OwnerLoc  = GetOwner()->GetActorLocation();
    const FVector TargetLoc = Target->GetActorLocation();
    const float   Distance  = FVector::Dist(OwnerLoc, TargetLoc);

    if (Distance > VisionRange) return false;

    const FVector Forward   = GetOwner()->GetActorForwardVector();
    const FVector ToTarget  = (TargetLoc - OwnerLoc).GetSafeNormal();
    const float   DotProduct = FVector::DotProduct(Forward, ToTarget);
    const float   HalfAngle  = FMath::DegreesToRadians(VisionAngleDegrees * 0.5f);

    if (DotProduct < FMath::Cos(HalfAngle))
    {
        if (Distance > PeripheralVisionRange) return false;
        const float PeriphHalf = FMath::DegreesToRadians(PeripheralVisionAngle * 0.5f);
        if (DotProduct < FMath::Cos(PeriphHalf)) return false;
    }

    FHitResult Hit;
    FCollisionQueryParams Params;
    Params.AddIgnoredActor(GetOwner());
    Params.bTraceComplex = false;

    const bool bBlocked = GetWorld()->LineTraceSingleByChannel(
        Hit, OwnerLoc + FVector(0, 0, 60.f), TargetLoc + FVector(0, 0, 60.f),
        ECC_Visibility, Params);

    return !bBlocked || Hit.GetActor() == Target;
}

bool UCognitiveSensorComponent::CanHearEvent(const FVector& EventOrigin, float Volume) const
{
    if (!GetOwner()) return false;
    const float Distance = FVector::Dist(GetOwner()->GetActorLocation(), EventOrigin);
    const float EffectiveRadius = HearingRadius * Volume;
    return Distance <= EffectiveRadius;
}

void UCognitiveSensorComponent::BroadcastSoundEvent(
    const FVector& Origin, float Volume, float MaxRadius)
{
    if (!CanHearEvent(Origin, Volume)) return;

    const float Distance  = FVector::Dist(GetOwner()->GetActorLocation(), Origin);
    const float Intensity = FMath::Clamp(1.f - (Distance / (MaxRadius * Volume)), 0.f, 1.f);

    FCognitiveSensorData Event;
    Event.EventType     = ECognitiveSensorEvent::HearingEvent;
    Event.EventLocation = Origin;
    Event.Intensity     = Intensity;
    Event.Distance      = Distance;
    Event.Timestamp     = FPlatformTime::Seconds();

    if (RecentEvents.Num() < MaxStoredEvents)
        RecentEvents.Add(Event);

    OnSensorEventDetected.Broadcast(Event);
}

void UCognitiveSensorComponent::BroadcastExplosionEvent(const FVector& Origin, float Radius)
{
    if (!GetOwner()) return;
    const float Distance = FVector::Dist(GetOwner()->GetActorLocation(), Origin);
    if (Distance > Radius * 3.f) return;

    FCognitiveSensorData Event;
    Event.EventType     = ECognitiveSensorEvent::ExplosionNearby;
    Event.EventLocation = Origin;
    Event.Intensity     = FMath::Clamp(1.f - Distance / (Radius * 3.f), 0.f, 1.f);
    Event.Distance      = Distance;
    Event.Timestamp     = FPlatformTime::Seconds();

    OnSensorEventDetected.Broadcast(Event);
}

void UCognitiveSensorComponent::BroadcastAllyDeath(const FVector& Location)
{
    FCognitiveSensorData Event;
    Event.EventType     = ECognitiveSensorEvent::AllyDeathNearby;
    Event.EventLocation = Location;
    Event.Intensity     = 1.f;
    Event.Distance      = FVector::Dist(GetOwner()->GetActorLocation(), Location);
    Event.Timestamp     = FPlatformTime::Seconds();
    OnSensorEventDetected.Broadcast(Event);
}

void UCognitiveSensorComponent::ScanVision()
{
    if (!GetOwner() || !GetWorld()) return;

    // BA-03 FIX: GetAllActorsOfClass() é O(n) sobre todos os atores do mundo.
    // Chamada a cada SensorTickInterval (0.1s) para CADA TargetClass = gargalo
    // severo em mapas grandes. Correção: cache com refresh a cada 2.0s.
    // O cache é por TargetClass; se o array de classes mudar, o cache se invalida.
    const float CacheLifetime = 2.0f;
    const double Now = GetWorld()->GetTimeSeconds();
    if ((Now - CachedActorsTimestamp) > CacheLifetime || CachedTargetClasses != TargetClasses)
    {
        CachedActors.Reset();
        for (TSubclassOf<AActor>& TargetClass : TargetClasses)
        {
            if (!TargetClass) continue;
            TArray<AActor*> Found;
            UGameplayStatics::GetAllActorsOfClass(GetWorld(), TargetClass, Found);
            CachedActors.Append(Found);
        }
        CachedActorsTimestamp = Now;
        CachedTargetClasses   = TargetClasses;
    }

    for (AActor* Actor : CachedActors)
    {
        if (!IsValid(Actor) || Actor == GetOwner()) continue;
        if (!CanSeeTarget(Actor)) continue;

        const float Distance  = FVector::Dist(GetOwner()->GetActorLocation(), Actor->GetActorLocation());
        const float Intensity = FMath::Clamp(1.f - Distance / VisionRange, 0.2f, 1.f);

        FCognitiveSensorData Event;
        Event.EventType     = ECognitiveSensorEvent::VisualContact;
        Event.EventLocation = Actor->GetActorLocation();
        Event.SourceActor   = Actor;
        Event.Intensity     = Intensity;
        Event.Distance      = Distance;
        Event.Timestamp     = FPlatformTime::Seconds();

        OnSensorEventDetected.Broadcast(Event);
    }
}

void UCognitiveSensorComponent::CleanupOldEvents(float DeltaTime)
{
    EventCleanupTimer += DeltaTime;
    if (EventCleanupTimer < 1.f) return;
    EventCleanupTimer = 0.f;

    const double Now = FPlatformTime::Seconds();
    RecentEvents.RemoveAll([Now](const FCognitiveSensorData& E)
    {
        return (Now - E.Timestamp) > MaxEventAge;
    });
}
