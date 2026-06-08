#include "CognitiveNPCBoneDriverDetails.h"

#include "CognitiveNPCBoneDriver.h"
#include "CognitiveBehaviorTypes.h"

#include "DetailLayoutBuilder.h"
#include "DetailCategoryBuilder.h"
#include "DetailWidgetRow.h"
#include "IDetailPropertyRow.h"
#include "PropertyHandle.h"
#include "Widgets/Text/STextBlock.h"
#include "Widgets/SBoxPanel.h"
#include "Styling/AppStyle.h"

#define LOCTEXT_NAMESPACE "CognitiveNPCBoneDriverDetails"

TSharedRef<IDetailCustomization> FCognitiveNPCBoneDriverDetails::MakeInstance()
{
    return MakeShared<FCognitiveNPCBoneDriverDetails>();
}

// ─────────────────────────────────────────────────────────────────────────────
void FCognitiveNPCBoneDriverDetails::CustomizeDetails(IDetailLayoutBuilder& DetailBuilder)
{
    // Categoria de destaque no topo, agrupando os controles que antes exigiam
    // nodes no Blueprint. Tudo editável direto no painel de detalhes.
    IDetailCategoryBuilder& Setup = DetailBuilder.EditCategory(
        "Cognitive Setup",
        LOCTEXT("SetupCat", "Cognitive | Setup"),
        ECategoryPriority::Important);

    // Nota explicativa (estilo Details panel)
    Setup.AddCustomRow(LOCTEXT("SetupNoteFilter", "Setup"))
    .WholeRowContent()
    [
        SNew(STextBlock)
        .Text(LOCTEXT("SetupNote",
            "Configure tudo aqui — sem nodes no Blueprint. Escolha o Observation "
            "State e o contexto de treino (categoria + subtipo). "
            "Para categoria própria, escolha Custom e nomeie no campo."))
        .ColorAndOpacity(FSlateColor::UseSubduedForeground())
        .AutoWrapText(true)
    ];

    // ── Observation State ─────────────────────────────────────────────────────
    TSharedRef<IPropertyHandle> ObsStateProp = DetailBuilder.GetProperty(
        GET_MEMBER_NAME_CHECKED(UCognitiveNPCBoneDriver, ObservationState));
    Setup.AddProperty(ObsStateProp);

    // ── Training Context (Category/Subtype/LocomotionState) ───────────────────
    // BehaviorContext é uma struct EditAnywhere; expomos seus membros internos
    // diretamente para que apareçam como dropdowns/campos simples.
    TSharedRef<IPropertyHandle> BehaviorProp = DetailBuilder.GetProperty(
        GET_MEMBER_NAME_CHECKED(UCognitiveNPCBoneDriver, BehaviorContext));

    // Esconde a struct-pai no local padrão; expomos os membros direto na seção Setup
    DetailBuilder.HideProperty(BehaviorProp);

    if (BehaviorProp->IsValidHandle())
    {
        uint32 NumChildren = 0;
        BehaviorProp->GetNumChildren(NumChildren);
        for (uint32 i = 0; i < NumChildren; ++i)
        {
            TSharedPtr<IPropertyHandle> Child = BehaviorProp->GetChildHandle(i);
            if (Child.IsValid())
            {
                Setup.AddProperty(Child.ToSharedRef());
            }
        }
    }

    // Conexão também é parte do setup essencial
    Setup.AddProperty(DetailBuilder.GetProperty(
        GET_MEMBER_NAME_CHECKED(UCognitiveNPCBoneDriver, PythonHost)));
    Setup.AddProperty(DetailBuilder.GetProperty(
        GET_MEMBER_NAME_CHECKED(UCognitiveNPCBoneDriver, PythonPort)));
    Setup.AddProperty(DetailBuilder.GetProperty(
        GET_MEMBER_NAME_CHECKED(UCognitiveNPCBoneDriver, bAutoConnect)));
}

#undef LOCTEXT_NAMESPACE
