# Cognitive NPC — Quick Start Guide

Welcome! This guide gets your first intelligent NPC reacting to the world in
under 10 minutes. No machine learning required for the core systems.

> **Two layers:** The **Core** (Perception, Tags, Decisions, Physical States,
> Object Interaction) works instantly with drag-and-drop components. The
> **Neural Animation** module is optional and advanced — see Section 6.

---

## 1. Installation

1. Copy the `CognitiveAgent` folder into your project's `Plugins/` directory.
2. Open your project. When prompted, let the editor build the plugin.
3. Go to **Edit → Plugins**, search "Cognitive", and ensure it's enabled.
4. Restart the editor.

Required engine plugins (auto-enabled): Enhanced Input, Gameplay Abilities.

---

## 2. Make any actor "known" to NPCs (Entity Tags)

Select any actor in your level (an enemy, a weapon, a car, a traffic light) and
**Add Component → Cognitive Entity Tag**. Then set its category in the Details
panel:

| You want… | Set Category to | Extra fields |
|---|---|---|
| An enemy | `Character` | Disposition = `Enemy`, ThreatWeight 0–1 |
| A friendly NPC | `Character` | Disposition = `Friend` (or set Faction) |
| A pickable weapon | `Weapon` | bCanPickUp = true, AttachSocket = `hand_r` |
| A drivable car | `Vehicle` | VehicleType = `Car`, bCanEnter = true |
| A traffic light | `TrafficLight` | update TrafficState at runtime |
| A hazard / danger | `Hazard` | ThreatWeight 0–1 |
| Something to ignore | `Ignore` | — |

**Factions:** give your NPCs and allies the same `Faction` name and they
automatically treat each other as friends.

---

## 3. Give an NPC awareness (World Perception)

On your NPC actor, **Add Component → Cognitive World Perception**. Tune in
Details:

- **PerceptionRadius** — how far the NPC senses (default 2000).
- **FieldOfViewDegrees** — vision cone (default 200; use 360 for omniscient).
- **bRequireLineOfSight** — if true, walls block perception.
- **ScanRateHz** — how often it scans (default 4 Hz).
- **SelfFaction** — this NPC's faction.

That's it. The NPC now perceives every tagged actor around it and classifies
each as friend/enemy/pickup/vehicle/etc., with a suggested reaction.

**Read results in Blueprint:**
- `Get Nearest Threat` → the closest enemy/hazard (with suggested reaction).
- `Get Nearest Pickup` → the closest weapon/item it can grab.
- `Get Nearest Traffic State` → red/yellow/green of the nearest light.

---

## 4. React to the world (example Blueprint logic)

A minimal "see enemy → act" in the NPC's Blueprint Event Graph:

```
Event Tick (or a Timer)
  → Get Nearest Threat
  → Branch (Threat.Actor is valid?)
      True → Switch on Threat.SuggestedReaction
                Attack → your attack montage
                Flee   → set move target away from threat
                Hide   → move to nearest Cover
```

The component already decides *which* reaction fits (strong+close enemy → Flee,
otherwise Attack). You just wire the reaction to your animations/movement.

---

## 5. Object interaction (hold a weapon)

To make an NPC pick up a tagged weapon, on the **World Perception** component
call:

- `Attach Object To Hand (Object, "hand_r")` — snaps the item to the skeleton
  socket and disables its physics.
- `Drop Held Object` — releases it and restores physics.
- `Get Held Object` — returns what it's currently holding.

Make sure your skeleton has the socket name you pass (e.g. `hand_r`).

---

## 6. Physical states (health, death, falling, swimming)

Physical states tell your Anim Blueprint which animation to play. They are
driven by the NPC's health and movement mode.

- Add **Cognitive Health** to the NPC and set **MaxHealth** in Details.
- From your gameplay (player hits/shoots/runs over the NPC), call
  **Apply Damage (Amount)**. Use **Heal**, **Kill**, **Revive** as needed.
- Bind **On Death** to play your death montage, and **On Health Changed** to
  update a health bar (`Get Health Fraction` gives 0..1).
- **Falling** and **Swimming** are detected automatically from the Character
  Movement mode.

Bind in Blueprint:
```
(NPC) Cognitive Motion Learner → On Physical State Changed (NewState)
   Switch on NewState:
     Dead     → play death montage, disable input
     Falling  → play falling pose
     Swimming → play swim blendspace
     Landing  → play land animation
```

Helpers: `Get Physical State`, `Is Dead`.

---

## 7. (Advanced / Optional) Neural Animation

This module lets an NPC **learn movement by observing a leader character**. It
is advanced and optional — the core systems above need none of it.

High level:
1. Add **Cognitive Leader Observer** + **Cognitive NPC Bone Driver** to the NPC.
2. Train by running the included Python trainer while the NPC observes the
   leader (see `NEURAL_ANIMATION_SETUP.md`).
3. Export the trained model to TorchScript and run it natively in-engine with
   the **Cognitive Native Inference** component — no runtime Python.

A pre-trained sample model is included so you can see it working before
training your own. Full details in `INFERENCIA_NATIVA_LIBTORCH.md`.

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| NPC ignores an actor | Confirm the actor has a Cognitive Entity Tag and Category ≠ Ignore |
| NPC senses through walls | Enable bRequireLineOfSight |
| Weapon won't attach | Check the socket name exists on the skeleton |
| Death animation doesn't play | Bind On Physical State Changed; confirm Health reaches 0 |
| Neural module does nothing | It's optional; confirm a model is loaded (see advanced docs) |

---

## Support

Questions or issues? Use the support channel listed on the product page.
Include your engine version and a short description; screenshots help.
