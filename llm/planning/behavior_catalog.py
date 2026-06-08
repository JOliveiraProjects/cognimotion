"""
behavior_catalog.py — Catálogo comportamental para NPCs cognitivos.

Este módulo NÃO cria nova arquitetura de rede neural. Ele é a camada
DETERMINÍSTICA de decisão de alto nível: a partir do que o NPC percebe
(entidades, distância, ângulo, ameaça, disposição) e do seu estado interno
(medo, raiva, vida, perfil), decide uma INTENÇÃO comportamental, que a
ReactiveDecisionLayer converte em ação de locomoção + physical_state.

Por que determinístico e não aprendido:
  O cérebro neural (RSSM/DreamerV3) aprende a IMITAR movimento a partir do
  obs de 256-dim. Meter percepção + relações + perfis nesse obs recriaria o
  modelo do zero. Então o "aprendizado" de comportamento aqui é por
  configuração + regras inspiradas em como um instrutor ensina: "isto é um
  inimigo, reaja assim"; "isto é um aliado, proteja"; "refém presente, não
  atire". É extensível: adicionar comportamento = adicionar regra/perfil aqui.

Tudo o que este módulo usa JÁ trafega no wire de percepção:
  category_name, disposition_name, reaction_name, distance, threat_weight,
  direction (espaço local do NPC: +X frente, -X trás, +Y direita, -Y esquerda).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# EMOÇÕES — derivadas da percepção + estado, como um instrutor rotularia.
# ─────────────────────────────────────────────────────────────────────────────
class Emotion(str, Enum):
    CALM    = "calm"      # nada relevante percebido
    HAPPY   = "happy"     # aliado/amigo perto, sem ameaça
    ALERT   = "alert"     # algo percebido, ainda avaliando
    FEAR    = "fear"      # ameaça dominante e supera capacidade de reagir
    ANGER   = "anger"     # ameaça + agressividade alta → confrontar
    PANIC   = "panic"     # ameaça crítica muito perto


# ─────────────────────────────────────────────────────────────────────────────
# RELAÇÃO — quem é o outro em relação ao NPC.
# ─────────────────────────────────────────────────────────────────────────────
class Relation(str, Enum):
    SELF     = "self"
    FRIEND   = "friend"
    ALLY     = "ally"
    NEUTRAL  = "neutral"
    ENEMY    = "enemy"
    HOSTAGE  = "hostage"      # refém (proteger; não atingir)
    CAPTOR   = "captor"       # sequestrador (neutralizar com cuidado)
    UNKNOWN  = "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRIA SOCIAL — onde o alvo está em relação ao NPC (da direction local).
# ─────────────────────────────────────────────────────────────────────────────
class Bearing(str, Enum):
    FRONT      = "front"        # à frente, dentro do FOV
    LEFT       = "left"
    RIGHT      = "right"
    BACK       = "back"         # atrás (fora do FOV) — vulnerável
    OUT_OF_VIEW = "out_of_view"  # não percebido visualmente


def bearing_from_direction(direction: list, in_view: bool = True) -> Bearing:
    """
    Converte a direção relativa (espaço local do NPC) num rumo discreto.
    Convenção do C++: +X frente, +Y direita (mão direita), -X trás.
    """
    if not in_view:
        return Bearing.OUT_OF_VIEW
    if not direction or len(direction) < 2:
        return Bearing.OUT_OF_VIEW
    x, y = float(direction[0]), float(direction[1])
    ang = math.degrees(math.atan2(y, x))  # 0=frente, +90=direita, 180/-180=trás
    a = abs(ang)
    if a <= 45.0:
        return Bearing.FRONT
    if a >= 135.0:
        return Bearing.BACK
    return Bearing.RIGHT if ang > 0 else Bearing.LEFT


# ─────────────────────────────────────────────────────────────────────────────
# PERFIL COMPORTAMENTAL — define COMO o NPC tende a reagir.
# Inspiração: um instrutor define o "tipo" do agente. Valores em [0,1].
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class BehaviorProfile:
    name: str
    aggression: float = 0.5      # tende a atacar vs recuar
    courage: float = 0.5         # resiste ao medo
    caution: float = 0.5         # busca cobertura, avalia antes
    protectiveness: float = 0.5  # defende aliados/reféns
    # Repertório: quais reações este perfil pode emitir.
    repertoire: tuple = field(default_factory=lambda: (
        "attack", "flee", "hide", "pickup", "enter", "approach", "wait"
    ))
    # Distância em que considera "perto demais" (cm).
    danger_radius: float = 400.0


# Catálogo de perfis. Adicionar um perfil = adicionar uma entrada aqui.
PROFILES = {
    # ── URBANO: civil. Pouca agressão, foge cedo, busca cobertura. ──────────
    "urbano": BehaviorProfile(
        name="urbano", aggression=0.15, courage=0.30, caution=0.80,
        protectiveness=0.40,
        repertoire=("flee", "hide", "wait", "pickup", "enter", "approach"),
        danger_radius=600.0,
    ),
    # ── MILITAR: treinado. Agressivo, corajoso, usa cobertura tática. ───────
    "militar": BehaviorProfile(
        name="militar", aggression=0.85, courage=0.85, caution=0.60,
        protectiveness=0.70,
        repertoire=("attack", "hide", "approach", "flee", "wait", "pickup"),
        danger_radius=300.0,
    ),
    # ── ESPORTIVO: atlético. Movimento rápido, evasão, sem combate letal. ───
    "esportivo": BehaviorProfile(
        name="esportivo", aggression=0.40, courage=0.70, caution=0.40,
        protectiveness=0.50,
        repertoire=("approach", "flee", "pickup", "wait"),
        danger_radius=350.0,
    ),
    # ── PILOTO: prioriza entrar/operar veículo; evita combate a pé. ─────────
    "piloto": BehaviorProfile(
        name="piloto", aggression=0.30, courage=0.60, caution=0.55,
        protectiveness=0.45,
        repertoire=("enter", "flee", "approach", "wait", "hide"),
        danger_radius=500.0,
    ),
    # ── LUTADOR: combate corpo a corpo. Muito agressivo, corajoso. ──────────
    "lutador": BehaviorProfile(
        name="lutador", aggression=0.95, courage=0.90, caution=0.25,
        protectiveness=0.55,
        repertoire=("attack", "approach", "wait", "flee"),
        danger_radius=200.0,
    ),
}

DEFAULT_PROFILE = BehaviorProfile(name="default")


def get_profile(name: Optional[str]) -> BehaviorProfile:
    if not name:
        return DEFAULT_PROFILE
    return PROFILES.get(name.strip().lower(), DEFAULT_PROFILE)


# ─────────────────────────────────────────────────────────────────────────────
# APPRAISAL — avalia a cena percebida e produz emoção + alvo prioritário.
# É o "ler a situação" que um instrutor ensina antes de mandar agir.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Appraisal:
    emotion: Emotion
    relation: Relation        # relação com o alvo prioritário
    bearing: Bearing          # onde está o alvo prioritário
    target: Optional[dict]    # entidade prioritária (ou None)
    hostage_present: bool     # há refém na cena
    captor_present: bool      # há sequestrador na cena
    reason: str


def _relation_of(entity: dict) -> Relation:
    """Mapeia disposição/categoria/tags da entidade para uma relação social."""
    disp = entity.get("disposition_name", "neutral")
    cat  = entity.get("category_name", "unknown")
    # O parser expõe role_name ("hostage"/"captor"/...); aceita também "role"
    # como string para chamadas diretas/testes.
    role = entity.get("role_name") or entity.get("role", "")
    if isinstance(role, int):
        role = ""  # role numérico sem role_name → sem papel especial

    if role == "hostage":
        return Relation.HOSTAGE
    if role == "captor":
        return Relation.CAPTOR
    if disp == "enemy":
        return Relation.ENEMY
    if disp == "ally":
        return Relation.ALLY
    if disp == "friend":
        return Relation.FRIEND
    if cat == "hazard":
        return Relation.ENEMY  # perigo ambiental tratado como ameaça
    return Relation.NEUTRAL


def appraise(perception: list, blackboard: dict,
             profile: BehaviorProfile) -> Appraisal:
    """
    Lê a percepção + estado e devolve uma avaliação emocional da cena.
    """
    fear = float(blackboard.get("fear_level", 0.0))
    aggr = float(blackboard.get("aggression_level", 0.0))

    if not perception:
        return Appraisal(Emotion.CALM, Relation.UNKNOWN, Bearing.OUT_OF_VIEW,
                         None, False, False, "nada percebido")

    hostage_present = any(_relation_of(e) == Relation.HOSTAGE for e in perception)
    captor_present  = any(_relation_of(e) == Relation.CAPTOR  for e in perception)

    # Alvo prioritário: maior ameaça; empate → mais perto.
    def prio(e):
        return (float(e.get("threat_weight", 0.0)),
                -float(e.get("distance", 1e9)))
    target = max(perception, key=prio)
    rel = _relation_of(target)
    threat = float(target.get("threat_weight", 0.0))
    dist = float(target.get("distance", 1e9))
    bearing = bearing_from_direction(target.get("direction", [0, 0, 0]),
                                     in_view=True)

    # Emoção: combina ameaça percebida, estado interno e perfil.
    effective_fear = fear * (1.0 - profile.courage)
    effective_aggr = aggr * profile.aggression

    if rel in (Relation.FRIEND, Relation.ALLY) and threat < 0.2:
        emotion = Emotion.HAPPY
        reason = f"aliado/amigo perto ({rel.value})"
    elif threat >= 0.8 and dist <= profile.danger_radius * 0.5:
        emotion = Emotion.PANIC
        reason = f"ameaça crítica muito perto (threat={threat:.2f} dist={dist:.0f})"
    elif threat >= 0.5 and effective_aggr >= effective_fear:
        emotion = Emotion.ANGER
        reason = f"ameaça + agressividade (aggr_ef={effective_aggr:.2f})"
    elif threat >= 0.5:
        emotion = Emotion.FEAR
        reason = f"ameaça + medo (fear_ef={effective_fear:.2f})"
    elif threat > 0.0:
        emotion = Emotion.ALERT
        reason = f"ameaça baixa, avaliando (threat={threat:.2f})"
    else:
        emotion = Emotion.CALM
        reason = "sem ameaça relevante"

    return Appraisal(emotion, rel, bearing, target,
                     hostage_present, captor_present, reason)


# ─────────────────────────────────────────────────────────────────────────────
# DECISÃO COMPORTAMENTAL — da avaliação + perfil para uma INTENÇÃO.
# Retorna uma reação canônica (attack/flee/hide/approach/pickup/enter/wait) que
# a ReactiveDecisionLayer já sabe converter em ação de locomoção + estado.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class BehaviorIntent:
    reaction: str          # canônica, compatível com reaction_name do wire
    emotion: Emotion
    reason: str


def decide_behavior(perception: list, blackboard: dict,
                    profile_name: Optional[str]) -> Optional[BehaviorIntent]:
    """
    Decisão de alto nível. Retorna None se não há nada que justifique
    sobrepor a política de locomoção (deixa o NPC seguir/imitar).
    """
    profile = get_profile(profile_name)
    ap = appraise(perception, blackboard, profile)

    if ap.emotion == Emotion.CALM:
        return None

    # ── CENÁRIO COMPLEXO: sequestrador + refém ────────────────────────────
    # Regra que um instrutor ensinaria: NÃO atire se o refém está na linha;
    # só neutralize o sequestrador se tiver ângulo limpo e perfil treinado.
    if ap.captor_present:
        captor = next((e for e in perception
                       if _relation_of(e) == Relation.CAPTOR), None)
        hostage = next((e for e in perception
                        if _relation_of(e) == Relation.HOSTAGE), None)
        if captor is not None:
            clean_shot = _has_clean_shot(captor, hostage)
            trained = profile.aggression >= 0.7 and profile.courage >= 0.7
            if clean_shot and trained:
                return BehaviorIntent("attack", Emotion.ANGER,
                    "sequestrador com ângulo limpo e perfil treinado → neutralizar")
            # Sem ângulo limpo OU sem treino → NÃO atira (protege o refém).
            if profile.caution >= 0.5 or hostage is not None:
                return BehaviorIntent("wait", Emotion.ALERT,
                    "refém em risco / sem ângulo limpo → aguardar, não atirar")
            return BehaviorIntent("approach", Emotion.ALERT,
                "aproximar com cautela para obter ângulo")

    # ── PANIC: sempre foge/esconde, independente do perfil ────────────────
    if ap.emotion == Emotion.PANIC:
        if "hide" in profile.repertoire and ap.bearing == Bearing.BACK:
            return BehaviorIntent("hide", Emotion.PANIC,
                "pânico + ameaça nas costas → esconder")
        return BehaviorIntent("flee", Emotion.PANIC, "pânico → fugir")

    # ── Proteção de aliado/refém: perfil protetor intervém ────────────────
    if ap.hostage_present and profile.protectiveness >= 0.6:
        return BehaviorIntent("approach", ap.emotion,
            "refém presente e perfil protetor → aproximar para proteger")

    # ── ANGER: confronta se o repertório permite ──────────────────────────
    if ap.emotion == Emotion.ANGER and "attack" in profile.repertoire:
        # Inimigo nas costas: vira/recua antes (não ataca às cegas).
        if ap.bearing == Bearing.BACK:
            return BehaviorIntent("approach", Emotion.ANGER,
                "inimigo atrás → reposicionar antes de atacar")
        return BehaviorIntent("attack", Emotion.ANGER,
            f"raiva + repertório de ataque ({profile.name})")

    # Sente raiva mas NÃO tem ataque no repertório (ex.: esportivo, piloto):
    # confronta aproximando-se OU recua, conforme a cautela.
    if ap.emotion == Emotion.ANGER:
        if profile.caution >= 0.6 and "flee" in profile.repertoire:
            return BehaviorIntent("flee", Emotion.ANGER,
                f"raiva mas cauteloso e sem ataque → recuar ({profile.name})")
        if "approach" in profile.repertoire:
            return BehaviorIntent("approach", Emotion.ANGER,
                f"raiva sem ataque → confrontar aproximando ({profile.name})")
        if "flee" in profile.repertoire:
            return BehaviorIntent("flee", Emotion.ANGER,
                f"raiva sem ataque nem aproximação → recuar ({profile.name})")

    # ── FEAR: foge ou esconde conforme cautela ────────────────────────────
    if ap.emotion in (Emotion.FEAR, Emotion.ALERT):
        if profile.caution >= 0.6 and "hide" in profile.repertoire:
            return BehaviorIntent("hide", ap.emotion,
                f"medo + cautela alta → esconder ({profile.name})")
        if "flee" in profile.repertoire:
            return BehaviorIntent("flee", ap.emotion,
                f"medo → fugir ({profile.name})")

    # ── HAPPY: aproxima do aliado, sem combate ────────────────────────────
    if ap.emotion == Emotion.HAPPY and "approach" in profile.repertoire:
        return BehaviorIntent("approach", Emotion.HAPPY, "aliado perto → aproximar")

    return None


def _has_clean_shot(captor: dict, hostage: Optional[dict]) -> bool:
    """
    Há ângulo limpo para o sequestrador sem atingir o refém?
    Heurística geométrica: se o refém está aproximadamente na mesma direção
    e mais perto que o sequestrador, a linha está bloqueada.
    """
    if hostage is None:
        return True
    cdir = captor.get("direction", [1, 0, 0])
    hdir = hostage.get("direction", [0, 1, 0])
    cdist = float(captor.get("distance", 1e9))
    hdist = float(hostage.get("distance", 1e9))
    # Ângulo entre as direções (produto escalar normalizado em 2D).
    cx, cy = float(cdir[0]), float(cdir[1])
    hx, hy = float(hdir[0]), float(hdir[1])
    cn = math.hypot(cx, cy) or 1.0
    hn = math.hypot(hx, hy) or 1.0
    cos_ang = (cx * hx + cy * hy) / (cn * hn)
    aligned = cos_ang > 0.85           # ~ até 30° de separação
    hostage_in_front = hdist < cdist   # refém mais perto = na linha
    return not (aligned and hostage_in_front)
