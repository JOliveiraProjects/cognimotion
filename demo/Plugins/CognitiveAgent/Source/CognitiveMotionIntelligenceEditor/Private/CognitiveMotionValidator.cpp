#include "CognitiveMotionValidator.h"
#include "CognitiveMotionLearnerComponent.h"
#include "CognitivePoseRecorderComponent.h"
#include "CognitiveAnimInstance.h"
#include "GameFramework/Character.h"
#include "Animation/AnimBlueprint.h"
#include "Sockets.h"
#include "SocketSubsystem.h"

FCognitiveValidationResult UCognitiveMotionValidator::ValidateActor(AActor* Actor)
{
    FCognitiveValidationResult Result;

    if (!Actor)
    {
        Result.bPassed = false;
        Result.Errors.Add(TEXT("Actor is null."));
        Result.Summary = TEXT("FAILED: null actor.");
        return Result;
    }

    CheckComponent(Actor, UCognitiveMotionLearnerComponent::StaticClass(),
        TEXT("UCognitiveMotionLearnerComponent"), Result);
    CheckComponent(Actor, UCognitivePoseRecorderComponent::StaticClass(),
        TEXT("UCognitivePoseRecorderComponent"), Result);

    if (ACharacter* Char = Cast<ACharacter>(Actor))
    {
        if (USkeletalMeshComponent* Mesh = Char->GetMesh())
        {
            if (!Cast<UCognitiveAnimInstance>(Mesh->GetAnimInstance()))
                Result.Warnings.Add(TEXT("AnimInstance is not UCognitiveAnimInstance. Assign it in the Mesh component."));

            if (!Mesh->GetAnimClass())
                Result.Errors.Add(TEXT("No AnimBP assigned to Skeletal Mesh."));
        }
        else
        {
            Result.Errors.Add(TEXT("Character has no Skeletal Mesh component."));
        }
    }

    Result.bPassed  = Result.Errors.IsEmpty();
    Result.Summary  = Result.bPassed
        ? FString::Printf(TEXT("OK — %d warning(s)"), Result.Warnings.Num())
        : FString::Printf(TEXT("FAILED — %d error(s), %d warning(s)"),
            Result.Errors.Num(), Result.Warnings.Num());
    return Result;
}

FCognitiveValidationResult UCognitiveMotionValidator::ValidateAnimBlueprint(UAnimBlueprint* AnimBP)
{
    FCognitiveValidationResult Result;

    if (!AnimBP)
    {
        Result.bPassed = false;
        Result.Errors.Add(TEXT("AnimBlueprint is null."));
        Result.Summary = TEXT("FAILED");
        return Result;
    }

    if (!AnimBP->ParentClass || !AnimBP->ParentClass->IsChildOf(UCognitiveAnimInstance::StaticClass()))
        Result.Warnings.Add(TEXT("AnimBP parent is not UCognitiveAnimInstance. Motion data will not be available in the graph."));

    Result.bPassed = Result.Errors.IsEmpty();
    Result.Summary = Result.bPassed ? TEXT("OK") : TEXT("FAILED");
    return Result;
}

FCognitiveValidationResult UCognitiveMotionValidator::ValidatePoseDatabase(UObject* PoseDB)
{
    // Motion Matching e PoseSearch removidos — esta validação não é mais necessária.
    FCognitiveValidationResult Result;
    Result.bPassed = true;
    Result.Summary  = TEXT("OK — PoseSearch não é mais usado neste plugin.");
    return Result;
}

FCognitiveValidationResult UCognitiveMotionValidator::ValidateProtocolCompatibility(
    const FString& PythonHost, int32 PythonPort)
{
    FCognitiveValidationResult Result;

    ISocketSubsystem* SS = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM);
    if (!SS)
    {
        Result.Errors.Add(TEXT("SocketSubsystem not available."));
        Result.bPassed = false;
        Result.Summary = TEXT("FAILED");
        return Result;
    }

    FSocket* TestSocket = SS->CreateSocket(NAME_Stream, TEXT("CognitiveValidate"), false);
    if (TestSocket)
    {
        TSharedRef<FInternetAddr> Addr = SS->CreateInternetAddr();
        bool bValid = false;
        Addr->SetIp(*PythonHost, bValid);
        Addr->SetPort(PythonPort);

        if (bValid && TestSocket->Connect(*Addr))
        {
            Result.Summary = FString::Printf(TEXT("OK — Python reachable at %s:%d"), *PythonHost, PythonPort);
            TestSocket->Close();
        }
        else
        {
            Result.Warnings.Add(FString::Printf(
                TEXT("Cannot connect to Python at %s:%d. Ensure the server is running."),
                *PythonHost, PythonPort));
            Result.Summary = TEXT("WARNING — Python not reachable");
        }
        SS->DestroySocket(TestSocket);
    }

    Result.bPassed = Result.Errors.IsEmpty();
    return Result;
}

void UCognitiveMotionValidator::CheckComponent(
    AActor* Actor,
    TSubclassOf<UActorComponent> CompClass,
    const FString& CompName,
    FCognitiveValidationResult& Result)
{
    // TECH DEBT FIX: componentes obrigatórios faltando eram reportados como Warnings,
    // fazendo bPassed = true (que depende apenas de Errors.IsEmpty()). Um NPC sem
    // UCognitivePoseRecorderComponent "passava" na validação e travava em runtime.
    // Correção: componentes listados aqui são OBRIGATÓRIOS → Errors.Add().
    if (!Actor->FindComponentByClass(CompClass))
        Result.Errors.Add(FString::Printf(TEXT("Missing required component: %s"), *CompName));
}
