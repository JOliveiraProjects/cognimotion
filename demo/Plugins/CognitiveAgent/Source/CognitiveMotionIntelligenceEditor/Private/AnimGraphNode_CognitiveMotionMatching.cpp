#include "AnimGraphNode_CognitiveMotionMatching.h"
#include "BlueprintActionDatabaseRegistrar.h"
#include "BlueprintNodeSpawner.h"
#include "Kismet2/CompilerResultsLog.h"

FText UAnimGraphNode_CognitiveMotionMatching::GetNodeTitle(ENodeTitleType::Type TitleType) const
{
    return NSLOCTEXT("CognitiveMotion", "NodeTitle", "Cognitive Motion Matching");
}

FText UAnimGraphNode_CognitiveMotionMatching::GetTooltipText() const
{
    return NSLOCTEXT("CognitiveMotion", "NodeTooltip",
        "Applies bone transforms received from Python directly to the NPC skeleton.");
}

FLinearColor UAnimGraphNode_CognitiveMotionMatching::GetNodeTitleColor() const
{
    return FLinearColor(0.1f, 0.4f, 0.9f, 1.f);
}

FString UAnimGraphNode_CognitiveMotionMatching::GetNodeCategory() const
{
    return TEXT("Cognitive Motion Intelligence");
}

void UAnimGraphNode_CognitiveMotionMatching::GetMenuActions(
    FBlueprintActionDatabaseRegistrar& ActionRegistrar) const
{
    UClass* ActionKey = GetClass();
    if (ActionRegistrar.IsOpenForRegistration(ActionKey))
    {
        UBlueprintNodeSpawner* Spawner = UBlueprintNodeSpawner::Create(GetClass());
        check(Spawner);
        ActionRegistrar.AddBlueprintAction(ActionKey, Spawner);
    }
}

void UAnimGraphNode_CognitiveMotionMatching::ValidateAnimNodeDuringCompilation(
    USkeleton* ForSkeleton, FCompilerResultsLog& MessageLog)
{
    Super::ValidateAnimNodeDuringCompilation(ForSkeleton, MessageLog);
    // MinConfidenceThreshold foi removido — o nó agora aplica bone transforms
    // diretamente do Python sem blending por confiança.
}

void UAnimGraphNode_CognitiveMotionMatching::GetOutputLinkAttributes(
    FNodeAttributeArray& OutAttributes) const
{
}


