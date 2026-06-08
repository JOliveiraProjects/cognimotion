#pragma once

#include "CoreMinimal.h"
#include "IDetailCustomization.h"

class IDetailLayoutBuilder;

/**
 * FCognitiveNPCBoneDriverDetails
 *
 * Customização do painel de detalhes do UCognitiveNPCBoneDriver.
 *
 * Objetivo: tornar a configuração 100% via editor, sem nodes no EventGraph.
 *   - ObservationState: dropdown direto (já é EditAnywhere; aqui agrupamos no topo)
 *   - Behavior Mode / Type: dropdowns
 *   - Custom Mode/Type Name: campos de texto — para criar um modo/tipo novo,
 *     basta selecionar "Custom" e digitar o nome.
 *
 * Toda a seção fica organizada numa categoria "Cognitive | Setup" destacada,
 * seguindo o design system padrão do editor da Unreal.
 */
class FCognitiveNPCBoneDriverDetails : public IDetailCustomization
{
public:
    static TSharedRef<IDetailCustomization> MakeInstance();

    virtual void CustomizeDetails(IDetailLayoutBuilder& DetailBuilder) override;
};
