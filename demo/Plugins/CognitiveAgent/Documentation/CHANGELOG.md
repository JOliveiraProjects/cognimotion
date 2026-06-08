# Cognitive NPC — Changelog

## 1.1.0 — Lançamento

### Núcleo (sem ML)
- **World Perception**: NPCs detectam e classificam atores próximos
  (enemy/friend/weapon/pickup/vehicle/traffic-light/cover/hazard/objective).
- **Entity Tags**: marque qualquer ator no editor com sua categoria semântica.
- **Reactive Decisions**: reação sugerida por situação (attack, flee, hide,
  pick up, enter, cross, wait, approach).
- **Physical States**: estados alive/dead/falling/swimming/landing com eventos.
- **Health Component**: vida configurável, ApplyDamage/Heal/Kill/Revive, evento
  OnDeath para ligar a animação de morte.
- **Object Interaction**: NPC segura arma/item em socket do skeleton; soltar.
- **Factions**: NPCs da mesma facção são amigos automaticamente.
- **Line of Sight & FOV**: percepção realista com bloqueio por parede e cone de
  visão configurável.

### Premium (neural, opcional)
- **Neural Animation (DreamerV3)**: world model que aprende movimento observando
  um personagem-líder.
- **Native Inference (LibTorch)**: inferência 100% nativa em C++, sem rede e sem
  Python em runtime.
- **TorchScript Exporter**: pipeline treino-Python → inferência-C++.

### Geral
- 100% exposto a Blueprint; código C++ completo incluído.
- Suporte a UE 5.4 – 5.7.
- Demo project com cenas de Perception, Physical States e Urban Crossing.

### Notas
- O módulo neural é marcado como avançado/experimental e acompanha um
  checkpoint pré-treinado de amostra.
