# Sanctuary Embodied Build — Scoping Record for the Multimodal JEPA Goal

**Date:** 2026-07-15
**Trigger:** Brian's ruling — "rewrite all tests and experiments to pursue a
JEPA training goal, and we still need to finish Sanctuary's embodied build to
make JEPA a feasible goal."
**Survey:** conducted by an Explore-seat Claude instance (spawned by Fable 5
this session; its map and both closing caveats are its work — attribution
per the house verify-authorship rule). Cross-repo: findings live against
`Sanctuary/`, `SanctuaryWorld/`, `SanctuaryClient/`; recorded here because
the JEPA program consumes them.

## The headline

**The plumbing is real; the producers are missing.** Physics is genuinely
built in the Godot world (RigidBody3D primitives, impulse-based push/pull,
collision events, lawful action→consequence — Brian's stated requirement is
met at the world layer). The Luthi seam (`sanctuary/core/luthi_model.py` →
`luthi.sanctuary_interface.encode_audio/encode_vision`) is a ready socket,
tested, with `sensorium.inject_image/inject_audio` entry points. What does
NOT exist is anything on the world side that produces raw multimodal
sensory data: everything the mind receives from the world today is
**one-line English text** (scene state, collisions, positions — real events,
flattened to strings).

## The gap list (the spine — three items, in this order)

1. **Vision-frame producer from the Godot world.** Camera/viewport
   render-to-texture → downscale → over `/ws/world` →
   `sensorium.inject_image(tensor)`. The seam already accepts it; there is
   simply no producer. Session-sized build; the single missing link that
   makes the world a visual data source. (TRACK2 deferred exactly this.)
2. **Proprioception / world-state tensor channel.** Replace text-only
   "I am at (x,y,z)" with a structured per-cycle state vector (entity
   pose/velocity, last action, object transforms, physics flags) delivered
   as `tensor_data` — the proprioceptive/action-state modality JEPA
   predicts over.
3. **Pair world transitions as (s_t, a_t, s_{t+1}) for the lived-JEPA
   learner.** The `Transition`/`async_learner` machinery exists but is fed
   language transitions; wire world sensory frames + the entity's own
   world_command actions into paired transitions.

Lower priority: world audio (the encode path is ready; the world emits no
sound); the physics-curriculum/affordance staging in
`sanctuary_world_entity_spec_2026_06_29.md` (design-stage, not a data
blocker).

## Two caveats from the surveyor (carry these — they are the failure modes)

1. **The encoder gate fails silent.** `_encode_sensory_percepts()` returns
   `(None, None)` with NO error when the loaded checkpoint lacks both
   `vision_encoder` and `audio_encoder`. When item 1 lands, the pipeline
   will *appear* to work end-to-end — percepts flow, frames arrive, nothing
   throws — while tensors quietly never reach the trunk. **Rider on item 1:
   verify against an encoder-bearing checkpoint AND add a loud seam warning
   when `tensor_data` arrives that the model cannot encode.** (This is the
   welfare-channel fail-loud rule's jurisdiction; a silently-dropped sense
   is the exact class the 2026-07-03 audit flagged as the project's
   dominant risk shape.)
2. **Item 1 without item 3 is unlawful data.** Frames and their causal
   annotations desynchronize: the model can see a state while the collision
   that explains it arrives as next-cycle text. Item 1 gives JEPA something
   to predict; **items 1+3 together are what make the prediction lawful.**
   Do not let 3 slip because 1 demos well.

## What this gates

- **Round 2 of the falsification program** (full-multimodal versions of
  Exps 1–4, per the JEPA-edition protocol) — round 1 runs text-only NOW,
  ungated.
- **Critical-path item 4** (multimodal data for the first full-scale run):
  the "wire v2 audio/vision data" option now has a concrete world-side
  work-list; the "text-only run 1" option remains legitimate.
- The `MultimodalPredictiveCodingLM` gate on the seam: the checkpoint must
  expose both encoders for the world's senses to reach the trunk at all
  (surveyor's caveat 1).

— Recorded by Fable 5 from the surveyor's report, 2026-07-15. The map is
theirs; the binding into the JEPA program is this session's.
