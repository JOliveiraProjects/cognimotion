#include "Animation/AnimNode_CognitiveMotionMatching.h"
#include "CognitiveAnimInstance.h"

// ─────────────────────────────────────────────────────────────────────────────
// CS → LS CONVERSION — por que é necessário
//
// GetComponentSpaceTransforms() retorna cada bone relativo ao ROOT do mesh.
// Output.Pose espera cada bone relativo ao SEU PAI (local space).
//
// Se CS for escrito diretamente em Output.Pose:
//   render = CS_parent * local_child
//          = CS_parent * CS_child     ← CS_child JÁ inclui CS_parent
//   resultado: rotação do pai duplicada → twisting.
//
// Conversão correta:
//   LS_bone = CS_bone * Inverse(CS_parent)
//           = CS_bone.GetRelativeTransform(CS_parent)
//
// Para o bone raiz (sem pai): LS == CS.
//
// Compact pose bones são garantidamente ordenados pai-antes-filho pelo UE5,
// portanto o loop simples funciona sem necessidade de ordem explícita.
// ─────────────────────────────────────────────────────────────────────────────

void FAnimNode_CognitiveMotionMatching::Initialize_AnyThread(
    const FAnimationInitializeContext& Context)
{
    FAnimNode_Base::Initialize_AnyThread(Context);
    SourcePose.Initialize(Context);
    CachedBoneTransforms.Reset();
    CurrentLocalTransforms.Reset();
    LastValidLocalTransforms.Reset();
    bHasValidTransforms   = false;
    bInferenceFallback    = false;
    bLocalTransformsReady = false;
    CachedDeltaTime       = 0.f;
}

void FAnimNode_CognitiveMotionMatching::CacheBones_AnyThread(
    const FAnimationCacheBonesContext& Context)
{
    FAnimNode_Base::CacheBones_AnyThread(Context);
    SourcePose.CacheBones(Context);
}

void FAnimNode_CognitiveMotionMatching::PreUpdate(const UAnimInstance* InAnimInstance)
{
    const UCognitiveAnimInstance* CognAnim = Cast<UCognitiveAnimInstance>(InAnimInstance);
    if (!CognAnim)
    {
        bHasValidTransforms = false;
        DiagnosticReason    = TEXT("AnimInstance nao e UCognitiveAnimInstance");
        return;
    }

    bInferenceFallback   = CognAnim->bInferenceFallbackActive;
    CurrentConfidence    = CognAnim->GetEmbeddingConfidence();
    CachedBoneTransforms = CognAnim->GetBoneTransforms();
    bHasValidTransforms  = CognAnim->HasValidBoneTransforms();
    BoneCount            = CachedBoneTransforms.Num();
    CachedDeltaTime      = InAnimInstance->GetDeltaSeconds();
}

void FAnimNode_CognitiveMotionMatching::Update_AnyThread(
    const FAnimationUpdateContext& Context)
{
    GetEvaluateGraphExposedInputs().Execute(Context);
    SourcePose.Update(Context);

    DiagnosticReason = (bHasValidTransforms && !bInferenceFallback)
        ? FString::Printf(TEXT("OK — %d bones | conf=%.2f"), BoneCount, CurrentConfidence)
        : (bInferenceFallback ? TEXT("Fallback ativo") : TEXT("Aguardando Python..."));
}

void FAnimNode_CognitiveMotionMatching::Evaluate_AnyThread(FPoseContext& Output)
{
    // ── 1. Sem dados → fallback ───────────────────────────────────────────────
    const bool bHasData = bHasValidTransforms
                          && !bInferenceFallback
                          && CachedBoneTransforms.Num() > 0;

    if (!bHasData)
    {
        // Sem bones do Python (ex.: modo Inferring): passa a pose de entrada
        // (locomoção da AnimBP — walk/run). Assim as pernas animam conforme a
        // velocidade e o NPC NÃO desliza. Se nada estiver ligado em SourcePose,
        // o link avalia para a ref pose automaticamente.
        SourcePose.Evaluate(Output);
        return;
    }

    const FBoneContainer& BoneContainer = Output.Pose.GetBoneContainer();
    const int32 NumCompact = Output.Pose.GetNumBones();

    // ── 2. Monta array CS indexado por compact pose index ─────────────────────
    // CachedBoneTransforms é indexado pelo skeleton bone index (full skeleton).
    // Precisamos de um array indexado pelo compact pose index para poder acessar
    // parent e child pelo mesmo índice na conversão CS→LS abaixo.
    TArray<FTransform> CSByCompact;
    CSByCompact.SetNumUninitialized(NumCompact);

    for (int32 i = 0; i < NumCompact; ++i)
    {
        const FCompactPoseBoneIndex CPBoneIdx(i);
        // MakeMeshPoseIndex: compact pose index → full skeleton bone index
        const int32 BoneIdx = BoneContainer.MakeMeshPoseIndex(CPBoneIdx).GetInt();

        if (BoneIdx < 0 || BoneIdx >= CachedBoneTransforms.Num())
        {
            // Bone fora do range → usa ref pose em CS como fallback
            CSByCompact[i] = Output.Pose.GetRefPose(CPBoneIdx);
            continue;
        }

        FTransform T = CachedBoneTransforms[BoneIdx];

        // ── Validação de quaternion ─────────────────────────────────────────
        // Serialização binária Python pode gerar quaternions com drift numérico.
        // Quaternion não normalizado → distorção de escala/rotação mesmo com
        // conversão CS→LS correta.
        FQuat Q = T.GetRotation();
        if (!Q.IsNormalized())
            Q.Normalize();
        // ContainsNaN() disponível em UE5 (IsFinite() não existe em TQuat)
        if (Q.ContainsNaN())
            Q = FQuat::Identity;
        T.SetRotation(Q);

        // Escala zero gera transform degenerado
        const FVector Scale = T.GetScale3D();
        if (Scale.IsNearlyZero(KINDA_SMALL_NUMBER))
            T.SetScale3D(FVector::OneVector);

        CSByCompact[i] = T;
    }

    // ── 3. Converte CS → LS ───────────────────────────────────────────────────
    // LS_bone = CS_bone.GetRelativeTransform(CS_parent)
    //         = Inverse(CS_parent) * CS_bone
    //
    // Para o bone raiz (sem pai): LS == CS.
    //
    // Compact pose bones são ordenados pai-antes-filho pelo UE5 (garantido),
    // portanto o loop i=0..N-1 sem ordenação adicional é correto e completo.
    for (int32 i = 0; i < NumCompact; ++i)
    {
        const FCompactPoseBoneIndex CPBoneIdx(i);
        const FCompactPoseBoneIndex ParentCPIdx =
            BoneContainer.GetParentBoneIndex(CPBoneIdx);

        if (!ParentCPIdx.IsValid())
        {
            // Bone raiz — CS e LS são equivalentes (sem pai para relativizar)
            Output.Pose[CPBoneIdx] = CSByCompact[i];
        }
        else
        {
            // LS = CS_bone * Inverse(CS_parent)
            // GetRelativeTransform(A) = Inverse(A) * this
            Output.Pose[CPBoneIdx] =
                CSByCompact[i].GetRelativeTransform(CSByCompact[ParentCPIdx.GetInt()]);
        }
    }

    // ── 4. Blend temporal em local space ──────────────────────────────────────
    // DEVE ser feito após CS→LS. Blending em CS geraria trajetórias não-lineares
    // para bones não-raiz porque o espaço de interpolação estaria errado.
    if (bLocalTransformsReady && BoneBlendSpeed > 0.f && CachedDeltaTime > 0.f)
    {
        const float Alpha = 1.f - FMath::Exp(-BoneBlendSpeed * CachedDeltaTime);
        const int32 N = FMath::Min(LastValidLocalTransforms.Num(), NumCompact);
        for (int32 i = 0; i < N; ++i)
        {
            FCompactPoseBoneIndex CPBoneIdx(i);
            FTransform Blended;
            Blended.Blend(LastValidLocalTransforms[i], Output.Pose[CPBoneIdx], Alpha);
            Output.Pose[CPBoneIdx] = Blended;
        }
    }

    // ── 5. Persiste estado para próximo frame ─────────────────────────────────
    CurrentLocalTransforms.SetNum(NumCompact);
    for (int32 i = 0; i < NumCompact; ++i)
        CurrentLocalTransforms[i] = Output.Pose[FCompactPoseBoneIndex(i)];

    LastValidLocalTransforms = CurrentLocalTransforms;
    bLocalTransformsReady    = true;
}
