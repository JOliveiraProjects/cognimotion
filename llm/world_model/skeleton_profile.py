"""
Perfil e assinatura de skeleton.

Um modelo treinado (PoseDecoder) tem uma camada de saída de tamanho fixo
(num_bones × 7). Um modelo treinado num humanoide de 89 bones NÃO pode dirigir
um skeleton diferente — os tensores não encaixam. Para evitar que o usuário
carregue um .pt incompatível (e receba lixo ou crash), todo checkpoint grava a
ASSINATURA do skeleton com que foi treinado, e a verificamos antes de usar.

Hoje só o perfil "Default" (humanoide UE5 Mannequin, 89 bones) é usado de fato.
A estrutura já suporta outros tipos (quadrúpede, criatura) para uma versão
futura — basta registrar o perfil. Nada aqui treina múltiplos skeletons de uma
vez; cada tipo tem seu próprio arquivo de treino.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import List, Optional


def _fnv1a_64(s: str) -> int:
    """FNV-1a de 64 bits sobre os bytes UTF-8 — idêntico ao lado C++."""
    h = 1469598103934665603        # offset basis
    prime = 1099511628211
    mask = 0xFFFFFFFFFFFFFFFF
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * prime) & mask
    return h


@dataclass
class SkeletonProfile:
    """Descreve um tipo de skeleton treinável."""
    name: str                              # "Default", "Quadruped", "Monster_Spider"...
    num_bones: int                         # nº de bones que o PoseDecoder gera
    bone_names: List[str] = field(default_factory=list)  # opcional, p/ validação visual
    display_name: str = ""                 # rótulo amigável no editor
    available: bool = True                 # se False, aparece como "em breve" no wizard

    def signature(self) -> str:
        """
        Assinatura estável do skeleton: num_bones + hash dos nomes dos bones.
        Dois skeletons com mesmo num_bones mas bones diferentes têm assinaturas
        diferentes (evita falso "compatível").

        IMPORTANTE: usa FNV-1a de 64 bits sobre os nomes em minúsculo unidos por
        '|', EXATAMENTE igual ao lado C++ (UCognitiveSkeletonManagerComponent::
        HashBoneNames). Os dois lados PRECISAM concordar, senão a verificação de
        compatibilidade seria falsa.
        """
        h = _fnv1a_64("|".join(n.lower() for n in self.bone_names)) if self.bone_names else _fnv1a_64("")
        hash_hex = f"{h & 0xFFFFFFFFFFFF:012x}"
        return f"{self.name}:{self.num_bones}:{hash_hex}"


# ── Registro de perfis. Só "Default" está ativo hoje. ──────────────────────────
# A estrutura está pronta para ampliar; os demais ficam available=False até que
# o pipeline de captura/treino daquele tipo seja validado.
SKELETON_PROFILES = {
    "Default": SkeletonProfile(
        name="Default",
        num_bones=89,
        display_name="Humanoide (UE5 Mannequin / MetaHuman via retarget)",
        available=True,
    ),
    # Exemplos de extensão futura (desativados):
    "Quadruped": SkeletonProfile(
        name="Quadruped", num_bones=0, display_name="Quadrúpede (em breve)",
        available=False,
    ),
    "Creature": SkeletonProfile(
        name="Creature", num_bones=0, display_name="Criatura/Monstro (em breve)",
        available=False,
    ),
}

DEFAULT_PROFILE = SKELETON_PROFILES["Default"]


def profile_for(num_bones: int, name: str = "Default",
                bone_names: Optional[List[str]] = None) -> SkeletonProfile:
    """
    Constrói um perfil a partir dos dados reais do skeleton em uso (vindo do
    Unreal). Se o nome não estiver registrado, cria um perfil ad-hoc.
    """
    base = SKELETON_PROFILES.get(name)
    if base is not None and num_bones in (base.num_bones, 0):
        # Usa o perfil registrado, mas grava os nomes de bone reais se vierem.
        return SkeletonProfile(
            name=base.name, num_bones=num_bones or base.num_bones,
            bone_names=bone_names or base.bone_names,
            display_name=base.display_name, available=base.available,
        )
    return SkeletonProfile(name=name, num_bones=num_bones,
                           bone_names=bone_names or [], display_name=name)


def check_compatibility(trained_sig: str, current: SkeletonProfile) -> Optional[str]:
    """
    Compara a assinatura gravada no checkpoint com o skeleton atual.
    Retorna None se compatível, ou uma MENSAGEM DE ERRO clara se não for —
    informando qual skeleton o treino usou, para o usuário configurar o
    Blueprint/animação corretamente.
    """
    if not trained_sig:
        return None  # checkpoint antigo sem assinatura — não bloqueia (compat.)

    current_sig = current.signature()
    if trained_sig == current_sig:
        return None

    # Extrai num_bones da assinatura gravada (formato name:num_bones:hash)
    parts = trained_sig.split(":")
    trained_name = parts[0] if parts else "?"
    trained_bones = parts[1] if len(parts) > 1 else "?"

    return (
        "SKELETON INCOMPATÍVEL — este treino/modelo NÃO pode ser usado com o "
        "skeleton atual.\n"
        f"  • Modelo treinado com: '{trained_name}' ({trained_bones} bones)\n"
        f"  • Skeleton atual:      '{current.name}' ({current.num_bones} bones)\n"
        "  Um modelo de pose é específico do skeleton (a camada de saída tem "
        "tamanho fixo = num_bones × 7).\n"
        "  Soluções: (1) use um Blueprint/skeleton igual ao do treino, ou "
        "(2) treine um novo modelo para este skeleton."
    )
