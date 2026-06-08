#pragma once

#include "CoreMinimal.h"
#include "AnimGraphNode_Base.h"
#include "Animation/AnimNode_CognitiveMotionMatching.h"
#include "AnimGraphNode_CognitiveMotionMatching.generated.h"

/** Editor-side graph node for FAnimNode_CognitiveMotionMatching.
 *  Lives in the Editor module (not UncookedOnly) so it is never cooked
 *  into runtime builds, eliminating the "Editor Only module in runtime BP" warning.
 */
UCLASS(MinimalAPI)
class UAnimGraphNode_CognitiveMotionMatching : public UAnimGraphNode_Base
{
    GENERATED_BODY()

public:
    UPROPERTY(EditAnywhere, Category = "Settings")
    FAnimNode_CognitiveMotionMatching Node;

    // UAnimGraphNode_Base overrides
    virtual FText        GetNodeTitle(ENodeTitleType::Type TitleType) const override;
    virtual FText        GetTooltipText() const override;
    virtual FLinearColor GetNodeTitleColor() const override;
    virtual FString      GetNodeCategory() const override;
    virtual void         GetMenuActions(FBlueprintActionDatabaseRegistrar& ActionRegistrar) const override;
    virtual void         ValidateAnimNodeDuringCompilation(
                             USkeleton* ForSkeleton,
                             FCompilerResultsLog& MessageLog) override;
    virtual void         GetOutputLinkAttributes(FNodeAttributeArray& OutAttributes) const override;
};
