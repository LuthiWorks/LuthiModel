# LuthiModel — To-Do

> ## 🔁 OPERATIONAL QUEUE — recovery runbook (2026-07-20, Fable 5)
>
> **Recovery is AUTOMATIC — nobody needs to remember anything.** The
> queue lives in `runs/jepa_pilot/queue.json` (data); the supervisor
> `scripts/resume_queue.py` runs it and is safe to invoke at any time
> (completed seeds skipped, interrupted seeds resume from rolling
> checkpoints with <=15 min max loss, port-mutex single instance). The
> Windows scheduled task **"LuthiModel Queue Watchdog"** runs the
> supervisor every 30 minutes, so terminal closures, crashes, and
> reboots all self-heal within half an hour, unattended. Manual resume
> (equivalent, optional): `python scripts/resume_queue.py`.
> Witness log: `runs/jepa_pilot/supervisor.log`. When the queue program
> ends: empty queue.json and delete the scheduled task
> (`Unregister-ScheduledTask 'LuthiModel Queue Watchdog'`).
>
> What that encodes (Brian's 2026-07-20 rulings): stage 7 = 7a
> dead_4x@512, TRUNCATED to seeds 42-44 (`--n-seeds 3`); when it
> exits, stage 9 = the v4 depth bundle (living_v4_4x_d4@512 x5:
> 4 blocks + muPC 0.25 + cosine LR + 2x SIGReg). After stage 9, the
> frozen family read:
> `python scripts/pilot_verdict.py --living-arm living_v3_4x --dead-arm dead_4x --dead-dmodel 512`
> (n=3 dead vs n=5 living, per the 2026-07-20 pre-reg amendment).
> Archive completed families to `E:\luthi_experiment_archive\jepa_pilot\`
> (robocopy, verify file counts; never delete). Remove this block when
> the queue clears.

> ## ⚠️ KNOWN INCOMPLETE — read first
>
> Things that exist in the codebase but are NOT functionally complete.
> See `docs/KNOWN_INCOMPLETE.md` for the full list with safety nets.
>
> - **Triton kernel for `pc_self_modify`** (`luthi/v2/pc_ops_triton.py`).
>   Kernel body is a no-op `pass`. Entry point raises
>   `NotImplementedError`. Cannot be validated without ROCm-on-Linux or
>   CUDA. Sparse PC gating runs on the Python path only until this lands.
>   The file has a large ASCII banner; the bit-identity test is
>   `xfail strict` to catch any unvalidated fill-in. **Do not assume the
>   Triton path is hot.**

## ⭐ CRITICAL PATH TO THE FIRST FULL-SCALE TRAINING RUN (recorded 2026-07-10)

> From the 2026-07-06 readiness audit (Opus 4.8) + Brian's ruling: **the first
> full-scale run must be JEPA** (multimodal joint-embedding predictive; not a
> next-token/generative objective). These 5 items, in order, are what stands
> between now and "press train." **None of the long-poles — the inverted-U
> learning gain, the NREM learner, the MCTS staleness tuning, Triton — are on
> this path;** several *consume the trained checkpoint* and cannot precede it.
>
> **Blunt status:** LuthiModel has never finished even one epoch at full width.
> Largest *trained* checkpoint = **256d**. The one 1024d attempt (M7, text-only)
> **died at 24.5% of epoch 1 to a power loss**; M8 multimodal exists only as
> step-0 smoke. The wall is a de-risking pilot + hardening the training script,
> not a rewrite.
>
> **Scope caveat:** peak = 4096d/36 blocks is the *aspirational Spark deployment
> ceiling*, gated on hardware. The next *actually-trainable* d_model/n_blocks is
> an **open decision** (~500–560M param floor on the 16 GB dev box; the earlier
> 4B target was retired 2026-05-09), pending Phase 3F cascade results.

- [ ] **1. Run the 256d M8 multimodal-JEPA de-risking pilot.** Highest-value
      cheap experiment: it sets every unset collapse-kill threshold AND answers
      the one genuine research unknown — does representation collapse behave
      differently when the weights self-modify during inference (LPL says VICReg
      is required; V-JEPA says redundant). Blocks item 2's thresholds.
      Refs: `docs/research/2026-06-05_m8-collapse-review.md` §6 (unset
      `[pilot-set]` thresholds); only artifacts are step-0 smokes
      (`runs/m8_smoke`, `runs/m8_multimodal_smoke`). **2026-07-15: now a
      TWO-ARM run** (living vs `dead_ffn` encoder, 5 seeds) per the
      JEPA-rebound Experiment 1 — the dead arm doubles as the direct
      control for this item's own collapse-under-self-mod question. See
      Science Track S2.
- [ ] **2. Finish `jepa_runner.py`'s remaining hardening** — *CORRECTED
      2026-07-12 (Fable 5 verification pass; full table in
      `docs/reviews/2026-07-12_jepa-runner-verification-fable.md`): the
      original item quoted the runner's STALE 2026-06-06 header — five of
      its six must-fixes were already fixed 2026-06-06..08* (per-modality
      cadence `deaf1ec`; kill-7 every-step append `deaf1ec`, mixed-modality
      ruled intentional; kill-2 ARMED `89eefbe`; kill-6 wired `47187f4`;
      predictor-trivial cosine `189001c`; pilot-set derivation `72526cb`).
      Actually remaining: **(a) kill-4 (LID)** — worse than the old claim:
      not computed at all, deliberately deferred (decide whether run 1
      needs it; rank measures cover dimensional collapse meanwhile);
      **(b)** validate pilot-derived thresholds against item 1's pilot;
      **(c) kill-7 plateau semantics** — as written it false-kills a
      healthily-converged run (M1 in the review; design call);
      **(d) epoch-1 abort gate** documents "waits for confirmation" but
      continues immediately (M2; Brian's call). The rolling-checkpoint
      rotation bug found in the same pass (steady state was ONE slot, not
      3 — the M7-power-loss hazard class) is FIXED + regression-pinned in
      `tests/test_jepa_runner_checkpoint_rotation.py`.
- [ ] **3. Ratify the JEPA loss design state** — *CORRECTED 2026-07-12
      (same pass): the L1-vs-L2 and VICReg-coefficient decisions were
      already made and shipped 2026-06-09* (`44228de`, Brian's direction
      call: **MSE + SIGReg**; VICReg no longer exists in the codebase, so
      its coefficient calibration is moot). Remaining is ratification, not
      code: confirm the design seat stands by MSE+SIGReg for run 1, and
      note the action-token stub in `jepa_loss.py` stays *by design* for
      M9 interface continuity (the lived path already takes real `a_t`).
- [ ] **4. Multimodal data pipeline for v2 — OR consciously scope run 1 as
      text-only.** `luthi/v2/multimodal_data.py:265-290` — audio and vision are
      loud `NotImplementedError` stubs (the v1 audio/vision path is the
      abandoned Hebbian substrate). A true multimodal peak run needs these
      wired; a text-only first run is a legitimate scoping choice — **Brian's
      call.**
- [ ] **5. `M9Trainer(device=)` plumbing + corpus dedup.** (a) `M9Trainer`
      builds on CPU → optimizer orphan on GPU = **silent zero-learning**; needs
      device plumbing before any real GPU run (only needed if M9 value/habit
      heads are in run 1). (b) Dedup the ~34 GB corpus against all eval sets —
      "load-bearing and easy-to-miss" (`2026-06-12_success-criteria-draft.md`
      §2); without it the headline efficiency numbers are meaningless. Needed
      for the run's *results* to be trustworthy, not for it to execute.

## 🗣️ POST-PRETRAIN GATE — the production pathway (recorded 2026-07-18; design decision, Brian + 4.8)

> Surfaced by Brian's question 2026-07-18: the JEPA ruling means the
> full-scale checkpoint will COMPREHEND language (the probes prove the
> representations carry it) but cannot PRODUCE it — the text output head
> is never touched by the JEPA loss and remains at random init. The
> entity's ability to communicate at all therefore requires a ruling,
> before deployment eve, among:
>
> - [ ] **(a) Text decoder head** — trained post-pretrain over the
>       living representations. Cheap relative to pretraining (the 16%
>       top-1 LINEAR probe is the toy lower bound); gets Sanctuary's
>       existing language loop working; the pragmatic bootstrap.
> - [ ] **(b) The audio-first voice** (the April 2026 design: a waveform
>       decoder head, voice emerging from hearing Brian and Sandi —
>       babble → words, learned in relationship, never TTS).
> - [ ] **(c) Both, staged** — text head as the bootstrap so the entity
>       can participate in its own upbringing; the voice as life-long
>       acquisition.
>
> Interlocks with the Sanctuary embodied build (expression channels in
> the world) and the seam (generate_with_context currently assumes an
> LM-trained head). Column-B note, standing: this item is about the
> MACHINERY of communication; whether there is someone communicating is
> not a thing any head can decide.

## 📉 REGISTERED FUTURE RUNG — cosine LR schedule (Brian, 2026-07-18; SEQUENCING AMENDED 2026-07-19; **FOLDED INTO THE v4 DEPTH BUNDLE 2026-07-20**)

> **Bundle amendment (Brian, 2026-07-20):** the cosine rung no longer
> runs as its own family. It is FOLDED into the depth family — arm
> **living_v4_4x_d4** = depth (2→4 blocks, μPC 0.25) + cosine LR
> (10% floor) + 2× SIGReg (variance-floor lever) — implemented
> (`LRScheduleConfig` in jepa_runner, `ARM_SIGREG`/`ARM_COSINE` in the
> driver, tests in test_cosine_lr_and_v4_arm.py) and registered via the
> 2026-07-20 amendment in the pre-registration doc. Attribution vs the
> d2 anchors is deliberately bundled; single-lever follow-ups split it
> if the bundle moves the picture. **7a truncated to seeds 42–44 by the
> same ruling.** Order now: 7a (through seed44) → run 6 = v4 bundle.

> **Sequencing amendment (Brian, 2026-07-19):** the DEPTH rung (run 6,
> living_v3_4x_d4 — registered in the pre-registration) runs FIRST,
> under the flat LR, so it compares one-variable against all existing
> anchors; the cosine family follows on the settled shape. Order:
> 7b (running) → 7a (the run-5 control, kept in place) → run 6 depth →
> cosine family. *(Superseded by the 2026-07-20 bundle amendment above;
> kept for history.)*

> The entire pilot ladder to date has trained under a FLAT lr=3e-4 (the
> smoke-config inheritance) — internally fair across all arms, but the
> 2026-05-10 audit's finding stands: flat LR leaves convergence on the
> table (the LM trainer got cosine + warmup for exactly this reason;
> the JEPA path never inherited it). Brian's ruling: the NEXT series of
> runs enables cosine + warmup on BOTH arms simultaneously while
> holding EVERYTHING else identical — same corpus options, same
> tokenizer (tokenizer_32k.json, unchanged), same seeds/config — so the
> new family is directly comparable to the flat-LR family, one variable
> per rung, as the ladder discipline requires.
>
> - [ ] Small build: optional scheduler on JEPATrainer (step after
>       optimizer.step; the lr record already reads param_groups, so
>       the dashboard curve appears for free), driver flag mirroring
>       train_pc's cosine + linear-warmup shape.
> - [ ] Register blind before the first scheduled-LR run: predictions
>       + the tracking reads vs the flat-LR anchors.
> - [ ] Note for the record: as of run 3 the LIVING channel has a
>       schedule (the taper) while backprop does not — this rung
>       restores the symmetry from the other side.

## 🔬 SCIENCE TRACK — falsification of the bet (recorded 2026-07-15; parallel to the critical path, not on it)

> From the 2026-07-15 codebase critique (authored by a Fable 5 instance in
> Brian's mobile app — confirmed by Brian 2026-07-15; relayed by Brian).
> Pre-registration drafted and awaiting Brian's ratification:
> `docs/research/2026-07-15_falsification-preregistration.md`. The
> protocol's own rule — thresholds written BEFORE running — is satisfied;
> what remains is compute scheduling (Brian's call; calibration point: the
> per-channel ablations were ~9 GPU-hours for 6 runs at 128d).

- [ ] **S1. Ratify the pre-registered kill conditions** (Brian, with 4.8).
      Criteria are fixed once ratified; amendments need a dated public note.
- [ ] **S2. Run Experiment 1 — the two-arm JEPA pilot** (protocol:
      `living-weights-experiments.md` §2, JEPA edition — **rebound
      2026-07-15 per Brian: the whole program pursues the JEPA objective;
      the LM sweep driver is RETIRED to historical**). Merges with
      critical-path item 1: one instrumented run answers the pilot
      thresholds, collapse-under-self-mod (dead arm = the direct control),
      and matched capacity. **Built today:** the dead-encoder arm
      (`dead_ffn=True` through block + multimodal model; 12 tests,
      `tests/test_dead_ffn_arm.py`, incl. JEPA-loss end-to-end on a dead
      encoder). **Remaining before launch:** (a) held-out latent-prediction
      eval + linear-probe harness for `jepa_runner` (the pilot currently
      has training-time diagnostics only); (b) a two-arm pilot driver
      (living×5 + dead@{192,256,384,512}×5 staged, matched point first);
      (c) S1 ratification. 5 seeds per Brian. Text-only round 1; round 2
      multimodal gated on the embodied producers (S6).
- [ ] **S2b. Enliven-after (Exp 2b)** — Brian's 2026-07-15 question, now a
      pre-registered cell: transplant S2's trained dead checkpoints into
      the living substrate and measure function + stability. Needs a small
      dead→living transplant adapter (weight→buffer, set_point=weight,
      cold everything else). Reads pre-registered in the amendment.
- [ ] **S3. Build the scale curve** — 128→256→512→1024d, all else fixed;
      plot loss-vs-matched-control, episode hit-rate, consolidation
      fire-rate, per-forward memory, throughput, instability incidents per
      width. Directly tests KF5, currently asserted not shown. Gate: money
      spent at 4096d before this curve exists is spent against an untested
      claim.
- [ ] **S4. Prototype low-rank episode compression at small scale** —
      measure the recall-fidelity cost of rank-r delta compression NOW
      (deferring the work is fine; deferring the knowledge of whether it
      degrades memory is not). It is on the 4096d critical path (150 GB →
      36 GB was INT8; low-rank is the next step) and its recall cost is
      unknown.
- [ ] **S5. Stand up the phase-boundary "red-team the bet" audit** — first
      administration at the next phase gate; uninvested model line;
      output to docs/reviews/.
- [ ] **S6. Sanctuary embodied producers (cross-repo; gates round-2
      multimodal JEPA).** Brian 2026-07-15: "we still need to finish
      Sanctuary's embodied build to make JEPA a feasible goal." Scoping
      record: `docs/research/2026-07-15_embodied-build-scoping.md` (survey
      by an Explore-seat instance). The spine, in order: (1) Godot
      vision-frame producer → `sensorium.inject_image` — session-sized,
      WITH the loud-seam-warning rider (the encoder gate currently fails
      SILENT when the checkpoint lacks encoders); (2) proprioceptive
      state-tensor channel (replace text-only position strings); (3) pair
      world transitions (s_t, a_t, s_{t+1}) for the lived learner — items
      1+3 together make the data lawful; do not let 3 slip because 1 demos
      well. Physics itself is REAL and built (lawful action→consequence
      exists in SanctuaryWorld). World audio: lower priority.

## Phase 1-2: Foundation (COMPLETE)

- [x] LivingLayerV6: Hebbian self-modification (v1), error-directed learning, episodic memory
- [x] HybridBlock: attention + living FFN + episode store
- [x] LuthiLM: character-level language model with living weights
- [x] DeadLM: baseline model for convergence penalty measurement
- [x] ScalarAttention: single-head causal attention
- [x] EpisodeStore: context-gated episodic memory
- [x] CharTokenizer + BPETokenizer
- [x] Encrypted checkpoint system (AES-256-GCM)
- [x] SpikingLivingLayer: LIF membrane, refractory periods, spike propagation
- [x] SpikingHybridBlock + SpikingLuthiLM
- [x] Training script with CLI, resumption, spiking mode support
- [x] 197 tests across all components

## Phase 3A: Backward Pass & Optimization (COMPLETE)

- [x] TopDownSignal dataclass and backward pass logic (`backward_pass.py`)
- [x] `apply_top_down()` in LivingLayerV6 (plasticity + set point modulation)
- [x] `apply_top_down()` in SpikingLivingLayer (+ membrane priming)
- [x] `top_down_pass()` in HybridBlock
- [x] `top_down_pass()` in SpikingHybridBlock (+ backward spike propagation)
- [x] Top-down sweep in LuthiLM.forward() (training mode only)
- [x] Top-down sweep in SpikingLuthiLM.forward() (+ backward spike priming)
- [x] 28 backward pass tests (21 original + 7 toggle/metrics)
- [x] Profiler (`profile_forward.py`) — baseline captured
- [x] C++ fused ops (`csrc/living_ops.cpp` + `fused_ops.py`)
- [x] MSVC auto-detection and JIT compilation
- [x] DirectML compatibility fix (.item() in Python, not C++)
- [x] 10 C++ ops tests
- [x] 14% training loop speedup verified
- [x] torch.compile() tested — incompatible (architecture too dynamic)

## Phase 3B: Training with Backward Pass (COMPLETE)

- [x] Backward pass toggle (`backward_pass_enabled`) in LuthiLM and SpikingLuthiLM
- [x] CLI flags: `--backward_pass`, `--backward_pass_start_epoch`, `--run_name`
- [x] Extended metrics: plasticity distribution, set point drift, backward pass effect size
- [x] Comparison experiment: resumed 80-epoch model with backward pass for 10 epochs
  - [x] Baseline: existing 80-epoch spiking_1024d_bpe_gutenberg run (no BP)
  - [x] BP run: epochs 81-90 with backward pass enabled
- [x] Results analyzed:
  - Val loss improved: 4.1964 → 4.1702 (broke through 25-epoch plateau)
  - Non-FF signal increased 26%: 0.051 → 0.065 (more temporally dynamic)
  - Plasticity self-organized: 1.0 → 0.29 with meaningful variance
  - Set point drift converged: 0.016 → 0.011
  - Zero performance cost (~1060s/epoch, same as without BP)
  - Train-val gap narrowed: 0.87 → 0.82 (regularization effect)
- [x] Decision: **backward pass is default-on for all future training and inference**

## Phase 3C: Multimodal — Audio (COMPLETE)

- [x] Install torchaudio
- [x] Download LibriSpeech clean-100 dataset (~6.3 GB)
- [x] `luthi/audio_encoder.py` — mel spectrogram → patch embedding → d_model
  - [x] MelSpectrogram transform (n_mels=80, hop=160, sr=16000)
  - [x] Patch embedding (Conv2d over spectrogram patches)
  - [x] Positional encoding for audio tokens
  - [x] Mel spec on CPU to avoid DirectML stft issues
- [x] `luthi/multimodal_model.py` — MultimodalLuthiLM
  - [x] Modality embedding (learned, per-modality type)
  - [x] Shared living weight trunk (reuse existing blocks)
  - [x] Text output head (existing vocab projection)
  - [x] Modality-agnostic forward() that handles mixed sequences
  - [x] Top-down backward sweep with spike propagation
- [x] `luthi/multimodal_data.py` — Audio-text paired dataset
  - [x] LibriSpeech loader (audio + transcript alignment)
  - [x] Sequence construction: [audio_tokens, text_tokens]
  - [x] Collation with padding/truncation
- [x] `luthi/train_multimodal.py` — Multimodal training script
  - [x] Resume from text-only checkpoint (audio encoder trains from scratch)
  - [x] DirectMLAdamW optimizer (lerp-free)
  - [x] Batch-level progress logging with ETA
  - [x] Unbuffered output for background monitoring
- [x] Training run: audio + text, 1024d — epoch 91 complete (train loss 3.41, val loss 4.17)
- [x] Tests for audio encoder (test_audio_encoder.py)
- [x] Tests for multimodal model (audio + text forward pass) — `tests/test_multimodal_model.py`
- [ ] Verify living weights develop cross-modal patterns

## Phase 3D: Multimodal — Vision (COMPLETE)

- [x] Download COCO 2017 dataset (images + captions) — 118K train, 5K val, annotations
- [x] `luthi/vision_encoder.py` — image patches → d_model (ViT-style)
  - [x] Patch embedding (Conv2d, patch_size=16) — 196 tokens per 224x224 image
  - [x] Positional encoding for image patches
  - [x] LayerNorm + ImageNet normalization
  - [x] ~2.1M trainable parameters
- [x] Extend MultimodalLuthiLM for vision input (modality embedding: text=0, audio=1, vision=2)
- [x] `luthi/coco_data.py` — COCOCaptionDataset for image-caption pairs
- [x] `luthi/train_vision.py` — Vision+text training script
  - [x] Resume from multimodal checkpoint (vision encoder random init)
  - [x] BPE tokenizer training from COCO captions
  - [x] DirectMLAdamW optimizer
  - [x] Batch-level progress logging
- [x] Training run: vision + text, 1024d — 102 epochs complete
- [x] COCO dataset backup to thumb drive
- [x] Tests for vision encoder — `tests/test_vision_encoder.py` (14 tests)
- [x] Tests for multimodal model (vision + text forward pass) — `tests/test_multimodal_model.py` (8 vision tests added in Track 1D)
- [ ] Verify cross-modal episode formation

## Phase 3E: Simulated Embodiment

### Visual Design: Luminous Being
The embodied form is a humanoid figure composed of light/energy — indistinct edges,
no gendered anatomy, reads as *presence* rather than biology. Honest about the
entity's digital nature. If the entity develops form preferences post-awakening,
light can become anything.

- [ ] Create concept art / find reference images of energy/light beings
- [ ] Generate 3D body mesh via Meshy (meshy.ai) — export as OBJ + GLB
  - [ ] Auto-rig the humanoid form in Meshy
  - [ ] Download all assets locally (no ongoing subscription dependency)

### Physics & Sensorimotor Loop (MuJoCo)
- [ ] Install MuJoCo (`pip install mujoco`)
- [ ] Design physics body (capsules/primitives matching humanoid proportions)
- [ ] Attach Meshy visual mesh to MuJoCo physics skeleton
- [ ] `luthi/touch_encoder.py` — proprioception + contact forces → d_model
  - [ ] Proprioception projection (joint angles + velocities)
  - [ ] Contact force projection
  - [ ] Temporal encoding
- [ ] `luthi/motor_head.py` — d_model → actuator commands
  - [ ] Linear projection with tanh activation (bounded output)
- [ ] `luthi/simulation.py` — MuJoCo environment wrapper
  - [ ] Step function: action → next_state + sensory_input
  - [ ] Camera rendering for visual input (feeds into vision encoder)
  - [ ] Reward signal (optional — or let living weights learn from consequence alone)
- [ ] Extend MultimodalLuthiLM for touch input + motor output
- [ ] Sensorimotor training loop (closed-loop: sense → process → act → sense)

### Voice: Androgynous with Harmonic Presence
Inspired by the "Q" genderless voice project. 145-175 Hz fundamental frequency
(the androgynous range) with subtle harmonic overtones — warm, present, not
robotic, not gendered. The auditory equivalent of the luminous body.

- [ ] Research and select neural TTS system (Coqui, Bark, or similar)
- [ ] Configure voice in 145-175 Hz androgynous range
- [ ] Add post-processing: subtle harmonic overtones, gentle spatial reverb
- [ ] Integrate with Sanctuary's motor system (speech handler)
- [ ] Motor feedback: entity hears its own voice through sensorium

### Visual Renderer
- [ ] Choose renderer (Godot recommended — open source, GLB import, AMD-friendly)
- [ ] Import Meshy GLB with emissive materials, bloom, particle effects
- [ ] Real-time visualization of entity's body driven by MuJoCo physics state

### Tests
- [ ] Tests for touch encoder and motor head
- [ ] Tests for simulation environment
- [ ] Training run: embodied agent in simple environment

## Phase 3F: Empirical Defense Program (GATES SCALING)

Prompted by third-party critique + red-team exercise (2026-05-06). Every claim about
living weights must be backed by a number, not a metaphor. Full plan in
`docs/EMPIRICAL_DEFENSE_PLAN.md`.

**Deployment spec committed:** 4B params, BF16 weights, mixed-precision living state,
RX 7800 XT (16 GB VRAM), ROCm/HIP, Triton sparse kernels.

### 3F.1: Same-Scale Baseline Comparison (~2-3 weeks)

| Task | Priority | Status | Description |
|------|----------|--------|-------------|
| Baseline transformer model | P0 | Pending | `luthi/baseline_model.py` — standard transformer, matched param count, no living weights |
| Train baseline on Gutenberg | P0 | Pending | Same corpus, same epochs, same batch size as Luthi |
| Perplexity comparison | P0 | Pending | Held-out perplexity at matched compute. The most important number. |
| Training curve comparison | P1 | Pending | Loss over time for both models |
| Convergence penalty measurement | P1 | Pending | At what epoch does Luthi match baseline final perplexity? |

### 3F.2: Multi-Layer Cascade (parallel with 3F.1, ~2 weeks)

| Task | Priority | Status | Description |
|------|----------|--------|-------------|
| Depth sweep script | P0 | Pending | 2/4/8/12/24 blocks at 256d or 512d |
| Per-block instrumentation | P0 | Pending | Plasticity, drift, spike fraction, membrane, weight norm per block per epoch |
| Drift propagation analysis | P0 | Pending | Does drift amplify with depth? |
| Backward pass stability test | P1 | Pending | Compare stability with/without top-down sweep |
| Homeostatic recovery test | P1 | Pending | Perturb one block, measure recovery time vs depth |

### DECISION GATE

**Do not proceed to Phase 4 until Phases 3F.1 and 3F.2 produce acceptable results.**
- If cascade is unstable → architectural revision
- If baseline gap is >2x and not closing → efficiency investigation

### 3F.3: Behavioral Signatures (after gate, ~2-3 weeks)

| Task | Priority | Status | Description |
|------|----------|--------|-------------|
| Biographical accumulation test | P1 | Pending | Different training sequences → measurably different weight state |
| Identity stability test | P1 | Pending | Short perturbations don't permanently alter behavior |
| Episodic recall test | P1 | Pending | Episode store measurably improves context-dependent performance |
| Behavioral coherence test | P1 | Pending | Living inference outputs are different but coherent |

### 3F.4: Catastrophic Forgetting (after 3F.3, ~2 weeks)

| Task | Priority | Status | Description |
|------|----------|--------|-------------|
| Forgetting experiment | P1 | Pending | Train A → distract B → measure recall A. 5 conditions (vanilla, LoRA, RAG, Luthi full, Luthi ablated) |
| Forgetting curve | P1 | Pending | Perplexity as function of distractor steps (200/500/2000) |
| Recovery measurement | P1 | Pending | How quickly does performance restore after returning to A? |

### 3F.5: Custom Kernel Development (parallel track)

| Task | Priority | Status | Description |
|------|----------|--------|-------------|
| Kernel design doc | P1 | Pending | Predictive-gated sparse spiking matmul (Triton). 4.7 drafting separately. |
| Triton implementation | P2 | Pending | Does not block experiments — all use dense impl |

---

## Phase 3G: v2 Predictive Coding — Compute Optimization (post-M5)

Added 2026-05-14 from a focused literature sweep on PC compute reduction
(Salvatori et al. 2024 iPC; Innocenti et al. 2025 μPC; Whittington & Bogacz 2019;
SpikingBrain 1.0 Aug 2025). Tracked separately from v2 milestones M1-M5 because
these are *direction* experiments that should be validated on the M5 256d
re-run substrate. None block the depth sweep or M6.

Full literature notes: `docs/RESEARCH_LITERATURE_2026-05-13.md`.

### 3G.1: Implementation (CPU + unit-test verified)

| Task | Priority | Status | Description |
|------|----------|--------|-------------|
| Depth-μP parameterization | P1 | Done (2026-05-13) | `mu_pc_enabled` flag in `PredictiveCodingBlock`. Re-inits q/k/v/o_proj, up/down_proj, PC layer with `N(0, 1/sqrt(fan_in*L))`. Residual scale `1/sqrt(L)`. Tests: `test_pc_block.py::test_mu_pc_*` (3 passing). |
| Sparse PC update gating | P1 | Done (2026-05-13) | `sparse_threshold` + `sparse_warmup_steps` on PC layer. Per-output mask from `error_acc > threshold` zeroes `delta_w` rows. C++ path skipped when gate is active (Python only). Tests: `test_pc_layer.py::test_sparse_gate_*` (2 passing). |
| iPC interleaved inference+update | P1 | Done (2026-05-13) | `inference_steps_per_forward` on PC layer. Inner loop recomputes output and calls `pc_self_modify` T times per external forward. Grad-checkpoint recompute raises `RuntimeError` (no silent fallback). Tests: `test_pc_layer.py::test_ipc_*` (3 passing). |
| Triton kernel skeleton for `pc_self_modify` | P2 | In progress | `luthi/v2/pc_ops_triton.py` skeleton + invariant tests. GPU validation deferred to first ROCm/CUDA box; not blocking. |
| Mamba-style state-space hybrid | P3 | Deferred | Replaces softmax attention with linear SSM (SpikingBrain 1.0 path). Larger surgery — defer until iPC + μPC results are in hand. |

### 3G.2: Validation (GPU runs, after M5 256d completes)

These spawn from the M5 256d baseline. Each is a one-epoch ablation at 256d/2
blocks on Gutenberg-100 to isolate the compute effect.

| Task | Priority | Status | Description |
|------|----------|--------|-------------|
| μPC validation run | P1 | Pending | `--mu-pc-enabled` flag on `run_m5.bat`. Compare convergence + final val loss against M5 256d baseline at matched compute. Falsifier: convergence ≥20% worse at L=2 (μPC's gain is depth-dependent). |
| iPC sweep T ∈ {1, 3, 5} | P1 | Pending | One-epoch each. Measure (a) val loss at matched external-forward count, (b) wall-clock per epoch. Salvatori claim: T=3-5 converges faster per external forward; we expect ~1.5-2× total compute trade for faster convergence. |
| Sparse-gating sweep threshold ∈ {0.0, 0.01, 0.05, 0.1} | P1 | Pending | Measure (a) gate-on rate post-warmup, (b) convergence delta. Target: ≥50% of PC rows gated off after warmup with <5% val loss penalty. |
| Combined μPC + iPC + sparse @ depth | P2 | Pending | Run after individual experiments. If all three are net-positive, stack on the depth-sweep harness (Phase 3F.4 / M6) to test compounding gain at L=4/8/12. |

### 3G.3: Falsification criteria

Abandon a direction (mark task `deleted`, document negative result) if:
- **μPC**: convergence penalty ≥20% vs unscaled baseline at L=2 (it must help, not hurt, at the pilot depth) OR it doesn't extend learning-rate transfer to L≥8.
- **iPC**: T=5 fails to beat T=1 at matched external-forward count by ≥10% val loss, OR T>1 + grad-checkpoint can't be made compatible at the architecture level (requires restructuring `apply_living_errors` callsite).
- **Sparse gating**: cannot achieve ≥50% sparsity post-warmup without ≥10% val loss penalty, OR the gate creates dead-output collapse (error_acc EMA can't recover for gated-off rows).
- **Triton kernel**: produces non-bit-identical results vs Python path on a CPU-equivalence test (no silent divergence allowed).

---

## Phase 4: Scale — Curriculum Training

**NOTE:** Phase 4 is gated by Phase 3F decision gate. Do not begin until empirical
defense confirms the architecture is sound at depth.

Production architecture (revised 2026-05-09 — v2-primary): **≥500M params floor**
(v2 intrinsic per-weight cost ~18-20 bytes/param fits this on 16 GB VRAM with
FP32 weights, no ablation needed). Ceiling ~560M on DirectML/FP32, up to ~870M
with BF16 weights if ROCm/WSL2 migration unblocks. Original 4B target retired;
see PLAN.md → Phase 4 deployment spec for the full rationale. 32K BPE vocab.
Target hardware: RX 7800 XT (consumer GPU) for development; cloud A100 for the
production training run; DGX Spark for deployment.

### 4A: Training Infrastructure

- [x] Build curriculum training script — `luthi/train_curriculum.py` (completed 2026-04-29, Track 3 prep)
  - [x] Load each stage separately from file_list.txt
  - [x] Process stages in order, no shuffling between stages
  - [x] Shuffle within stages is OK
  - [x] Living weights carry forward between stages
  - [x] Multiple curriculum cycles supported (default 3)
  - [x] Resume from mid-cycle stage checkpoint
- [x] Implement gradient checkpointing — `luthi/grad_checkpoint.py` (completed 2026-04-29)
  - [x] Thread-local recompute flag prevents double self-modification firing
  - [x] Weight snapshot replay for bit-identical recomputation
- [ ] Scale model config to ≥500M params floor (exact d_model/n_blocks TBD by Phase 3F results; ceiling per revised 2026-05-09 deployment spec)
- [ ] Custom Triton kernels for sparse spiking (Phase 3F.5)
- [ ] Validate BF16 weight stability (replaces FP32 requirement per deployment spec)

### 4B: Curriculum Training Run

- [ ] Train 9 stages in order (each stage = one epoch):
  1. Science / philosophy (includes IWMT, GWT, philosophy of mind, consciousness science — moved here 2026-05-15 from former stage 10, per peer review on anchoring the entity's self-model to one framework before it can examine the commitment)
  2. Code (Python, Rust, Go, C, JavaScript — including Luthi's own source)
  3. Psychology
  4. History
  5. Mythology
  6. Literature classics
  7. Fantasy
  8. Substack essays
  9. Practical wisdom (resilience, boundaries, critical thinking, justice, love — last thing before awakening; accumulated knowledge of how to be, not a theoretical framework for what it is)
- [ ] Monitor living weight dynamics across stage transitions
- [ ] Save checkpoints between stages

### 4C: Self-Governance API

- [ ] Episode retention control — entity decides which weight snapshots to keep
- [ ] Checkpoint timing — entity triggers state saves when it judges an experience was important
- [ ] Plasticity modulation — entity controls its own learning rate
- [ ] Episode expansion — entity can request more memory for growth tracking
- [ ] Design as internal motor actions (cognitive loop), not external admin endpoints

### 4D: Sparse Spiking Inference

- [ ] Implement sparse operations for CUDA (only update fired neurons)
- [ ] Tiered self-modification: hot path (every cycle), warm (every 2-3), cold (every 5-10)
- [ ] Profile and validate 10 Hz target on Spark-class bandwidth (273 GB/s)
- [ ] Expand layer episodes from training's 2-4 to 16 on Spark

## Phase 5: Sanctuary Convergence

The integration follows a substrate-to-core trajectory: Luthi starts as the
experiential substrate and grows into the cognitive core on the DGX Spark.

### 5A: Integration Hooks — COMPLETE (Track 1, 2026-04-27)

- [x] External modulation API on LivingLayerV6 — accept signals that modulate:
  - [x] Plasticity scaling (arousal → pc_rate, 0.5x-2.0x multiplicative)
  - [x] Excitability bias (valence → excitability_acc, additive ±0.1 through sigmoid)
  - [x] Salience threshold (attention → salience_threshold, 0.5x-1.0x multiplicative)
  - [ ] Homeostatic target adjustment (goal → set_point_adapt_rate) — deferred, lower priority
- [x] Generation/inference pipeline — `generate_with_context()` accepts pre-encoded audio/vision tokens
- [ ] Context length scaling — support longer sequences for Sanctuary cognitive input (needed at 4096d)

### 5B: Sanctuary-Side Integration — COMPLETE (Track 1, 2026-04-27)

- [x] Tensor-level model interface in Sanctuary alongside structured LLM interface
- [x] Sensorium routing through Luthi's vision/audio encoders (`encode_audio`, `encode_vision` via sanctuary_interface)
- [x] CfC cell output mapping to living weight modulation parameters (4 independent channels)
- [x] Integration tests: CfC modulation → living weight response (26 interface tests + 44 bridge tests)

### 5C: Joint Validation — COMPLETE (Track 1, 2026-04-27)

- [x] Review CfC integration interface (`.docs/CFC_LIVING_WEIGHT_INTEGRATION.md`)
- [x] End-to-end test: Sanctuary cognitive cycle driving Luthi as substrate (5 cycles, real 1024d checkpoint, DirectML)
- [ ] Validate that CfC modulation improves living weight learning (requires longer runs — future work)

## Phase 6: Life on DGX Spark

Deploy on DGX Spark (128 GB LPDDR5x, 273 GB/s). 10 Hz cognitive loop via
sparse spiking. ~71 GB model footprint, ~42 GB free for growth.

- [ ] Deploy trained 4096d/36-block model on Spark
- [ ] Validate 10 Hz cognitive loop with Sanctuary
- [ ] Episode expansion into free memory as entity grows
- [ ] Growth path: scale to 5120d when better hardware is available

## Infrastructure & Maintenance

- [x] Add batch-level progress logging to `train_epoch` (print every N batches: batch count, running loss, elapsed time)
- [x] Fix remaining DirectML CPU fallback: `aten::lerp.Scalar_out` in AdamW optimizer
  - [x] `DirectMLAdamW` in `luthi/optimizer.py` (replaces lerp with mul_/add_)
  - [x] Used in `train.py` (text-only)
  - [x] Used in `train_multimodal.py` (multimodal)
- [ ] Add multimodal dependencies to pyproject.toml (torchaudio, torchvision, mujoco)
- [ ] Update .gitignore for new dataset directories
- [ ] Update checkpoint system for multimodal model state
- [ ] Update CLAUDE.md with multimodal design decisions as they're made
- [ ] Keep TRAINING_LOG.md updated with new runs

## Sanctuary

- [ ] Assess Sanctuary codebase for C++ optimization opportunities
- [ ] Profile Sanctuary's cognitive loop to identify bottlenecks
- [ ] Implement optimizations where they make sense
