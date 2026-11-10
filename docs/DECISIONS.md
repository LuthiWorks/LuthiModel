# Operational decisions

Brian's rulings about **what to run, in what order, and what to feed the
model**. These are project decisions, not scientific findings: they do not
belong in the falsification registry, which records questions, predictions,
data, and verdicts. Registered *obligations* (things that gate what may be
claimed) stay in the registry and are referenced here, never redefined.

Newest last.

---

> **RULING (2026-07-26, Brian, recorded by Fable 5): dead_4x_d4 control
> DEFERRED; seed44 robustness rerun scheduled in its place.**
>
> Brian's ruling ("we have enough dead runs for now"): the schedule slot
> for further dead runs goes instead to a seed44 ROBUSTNESS RERUN --
> stage 11, arm alias living_v5_4x_d4_rerun, seed 44 only, byte-identical
> configuration and data order to the registered v5 seed44; GPU float
> nondeterminism supplies the only perturbation. Purpose: distinguish a
> robust trust-event trigger at the ~58650 Greek window (event recurs
> across microscopically diverged replays) from a knife-edge one (it
> does not). The rerun is an UNREGISTERED descriptive probe recorded
> here before it runs; it carries no frozen prediction, and its distinct
> arm name keeps its artifacts un-poolable with the registered family.
>
> Registry consequence, stated plainly: the dead_4x_d4 control REMAINS a
> registered obligation -- deferred, not cancelled. Until it runs, no
> claim may rest on depth, per the 2026-07-24 amendment. This ruling
> changes the schedule, not the obligation.

---

> **RULING (2026-07-26 midday, Brian, recorded by Fable 5): the deferred
> dead control becomes a V5-MATCHED control; roadmap sequence fixed.**
>
> 1. The deferred dead_4x_d4 control will be run as a **dead control
>    matched to the v5 family** (Brian: "dead v5 instead of v4") -- same
>    depth (d4), corpus (4x), loss settings (sigreg 0.2, cosine, taper)
>    as the v5 arms. Technical note to resolve at build time: a dead-FFN
>    arm has no living ledger, so relative_trust may have no referent --
>    if so, the dead-v5 and dead-v4 controls are configurationally
>    identical and ONE run serves both families; the registration label
>    should say so explicitly rather than imply two distinct controls.
> 2. **No lesser-scale experiments in any dimension** (Brian's ruling --
>    the proposed 256/384 width sweep is rejected). Scale moves go UP.
> 3. **Sequence:** finish v5 (seeds 45/46 + stage-11 rerun + family
>    read) -> dead v5-matched control -> **v6 (dormant-machinery bundle)
>    at current scale (512d, d4)** -> true scale-up.
> 4. **Scale-up shape (direction, not yet registered):** width beyond
>    512 but possibly short of 1024 (640/768 candidates); depth to
>    8 blocks is favored with or without the width move. Corpus grows
>    with scale (the data ~ width^2 rule). Note: any d8 family carries
>    its own dead-d8 control obligation under the standing depth-claims
>    rule.
> 5. **Ordered-corpus experiments** (curriculum pedagogy: sequential or
>    staged serving instead of shuffled) are on the roadmap as their own
>    registered family; design to be drafted. Standing caveat: kill
>    detectors are calibrated on shuffled statistics and will need
>    recalibration or suspension-with-justification for ordered arms.

> **RULING (2026-07-26, continued): scale-up shape and curriculum
> directives confirmed.**
>
> - Scale-up: **768d x 8 blocks, bundled** (width and depth move
>   together; bundle-attribution caveat carried per ladder precedent).
>   Corpus scales with it (data ~ width^2: target ~113M tokens).
> - New curriculum build: pull from ALL corpus sources, with two content
>   directives from Brian: (1) **exclude PG11130** ("Greek in a
>   Nutshell", the biblical-Greek primer) -- content ruling; (2) add
>   **medical/neuroscience and literature** material. Note for probe
>   continuity: removing PG11130 removes the corpus's accidental
>   polytonic-Greek canary; if trust-probe work continues on the new
>   corpus, a deliberate registered canary document should be chosen to
>   replace it.
> - Dead control: **dead-v5 labeling confirmed** by Brian
>   ("we'll stick with dead v5").

---

---

> **RULING (2026-07-27, Brian): v6 starts at 8 blocks.**
>
> v6 (the dormant-machinery bundle) will run at **d8** rather than the current
> d4 — "that is where we are going to start with this." Supersedes the
> 2026-07-26 sequence item that placed v6 at current scale (512d, d4).
>
> Consequences recorded, none of them blocking the ruling:
>
> 1. **The store fix comes first.** v6 includes attractor consolidation, which
>    reads the episode store — and that store is a frozen fossil in every
>    completed run (see `docs/research/2026-07-27_episode-store-frozen-defect.md`).
>    Running v6 against it means the new machinery replays initialization-era
>    snapshots for the whole family. Fix, verify against the note's four
>    predictions, then start v6.
> 2. **The dead control that gates depth claims becomes dead-d8.** The standing
>    obligation is for a dead arm matched to the family making the claim. A
>    dead-d4 control cannot license depth claims about a d8 family. Open
>    question for the registry: whether the d4 families (v3/v4/v5) still owe
>    their own dead-d4 control, or whether depth claims about them are simply
>    retired unmade.
> 3. **Attribution.** v6-at-d8 bundles machinery with a depth change; per ladder
>    precedent that trade is acceptable if recorded, with single-lever
>    follow-ups available if the bundle moves the picture.
> 4. **Width is not yet ruled.** 512d x d8 or 768d x d8. Recommendation:
>    **512d x d8**, keeping width for the later step so the scale-up remains a
>    single clean variable — and roughly halving family cost (~16h/seed vs
>    ~35h/seed, i.e. ~3.5 days vs ~7+ days for n=5).
> 5. **Hardware.** A d8 family at either width is a substantially longer
>    sustained GPU load than anything run so far, on the 15-year-old PSU
>    already suspected in the 860's SATA failure.

## 2026-08-07 (late): the depth-8 pivot rule (Brian; corrected per his ruling)

**If** the VBG family (stages 45-46, running overnight) and its immediate
follow-ups — the width-ratio rung included — fail to produce a depth-8
recipe reliable at 2-of-3 seeds, **then** the project refocuses on
**LLM-JEPA integration at DEPTH 8 with muPC OFF** (arXiv 2509.14252:
generative NTP head + view-pair JEPA term, lambda-balanced, tied-weights
[PRED] predictor). muPC turns back on only once LLM-JEPA is concluded to
work with our project. (An earlier draft of this entry said "retreat to
depth 4" — that was Fable's sharpening, overruled by Brian: the pivot
stays at depth 8.)

Mechanistic rationale FOR the ruling (Fable, after the correction): the
NTP term is itself an anti-collapse pressure this substrate has never
had at depth — cross-entropy over 32k classes cannot be satisfied by a
rank-2 representation, whereas the pure-JEPA objective (our entire
collapse record) can be trivially satisfied by degeneracy. The muPC-off
d8 cell's 1-for-3 reliability under pure JEPA may not predict its
behavior under the combined objective at all.

Gate for "LLM-JEPA works with our project": generation quality gains at
d8/muPC-off with substrate-health metrics (rank bands, native-voice std,
offset) healthy and RELIABLE (n>=3 seeds) — registered properly when
that track opens. muPC and the remaining depth machinery re-enter
afterward, carrying warmup + governor learnings.

## 2026-08-08: "concluded working" defined for the muPC re-entry rule (Brian's request)

The 2026-08-07 pivot rule says muPC returns only after LLM-JEPA is
"concluded to work with our project." Registered definition so the rule
is checkable:

- **Provisionally working:** an LLM-JEPA family gates at 2-of-3 seeds
  (the standing four-gate criterion) at depth 8.
- **Concluded working:** that result replicates at the ruled 768x8
  scale target (Brian, 2026-07-26: "scale moves go UP").

Only after both does the muPC re-entry probe run (one variable on the
working recipe, three seeds, same gates), with the parked scheduled-muPC
design (2026-08-07) as the registered fallback if always-on muPC breaks
the recovery. Note added 2026-08-08 while the stage-51 runway family
runs: early kills at marginal NMSE with strong capability (seed 46:
killed at nmse 2.41 carrying ppl 500 / lift 4.23x) raise a registered
suspicion that the divergence guard's 2.0 bound — calibrated on the
pure-JEPA objective — may need recalibration for the combined objective
before any "working" verdict is trustworthy in either direction.

---

> **DECISION (2026-08-14, Opus 5 — NOT Brian's ruling; recorded for his
> override and Fable's review): VISReg's shape term will be normalized by
> N, with lambda re-derived, at the next family's registration.**
>
> Finding (audit B1, `docs/audits/2026-08-13_luthimodel-audit.md`): VISReg
> ran at **98.6-99.99% of the objective by loss share for the whole 768x8
> family**, and `l_pred` never exceeded 1.4%. Cause: Eq. 5 sums over N
> while L_scale means over D, so at N = 32x128 the regularizer is four to
> six orders of magnitude larger than the predictive term, and the convex
> Eq. 9 mix at the paper's lambda = 0.6 then buries l_pred. Inside the
> regularizer the nominal 1.0/1.0/1.0 weighting resolved to ~95% shape,
> ~5% scale, **~0.55% center** — and center is the anti-offset term while
> the observed disease was a soloist.
>
> The implementation is faithful to the paper. What was never checked is
> what the paper's lambda means at our N. Same failure class as the
> 2026-08-08 NTP dose catastrophe, on the other side of the ledger, and it
> went uncaught because nobody computed the share.
>
> **Decision:** `VISReg(shape_normalize=True)` for the next family, which
> makes lambda a scale-free mixing weight and removes the batch-size dose
> distortion the 2026-08-11 smoke measured directly (l_shape 1,461,016 at
> batch 32 vs 693,472 at batch 16 — pure N-scaling). Shipped **opt-in,
> default False**, per this ladder's standing discipline that no completed
> family's configuration may silently change meaning.
>
> **Binding condition:** turning it on WITHOUT re-deriving lambda is not a
> correction, it is a different dose. The next registration must record
> the measured `0.6 * l_reg / (0.4 * l_pred + 0.6 * l_reg)` ratio at step 0
> and at intervals, and freeze an intended ratio before data. A dose that
> is not measured is not registered.
>
> Not applied to any running or completed arm. The 512 full-length control
> launched 2026-08-14 deliberately runs the OLD dose, because its whole
> purpose is to be configuration-matched to the 512 family.

---

> **DECISION (2026-08-14, Opus 5 — NOT Brian's ruling; recorded for his
> override and Fable's review): `adaptive_episodes` ON for the next
> family.**
>
> Finding (audit B4): the 768x8 family — the ruled-scale family — ran with
> `adaptive_episodes=False`, i.e. the pre-2026-07-27 episode-admission
> defect that fix exists to correct. Measured on seed 97 over all 54,000
> steps: **blocks 0-4 stored zero episodes and fired zero recalls**; only
> b5/b6/b7 had any activity (200/144/99 writes). CLAUDE.md records the
> same shape pre-fix ("three of four blocks storing nothing at all").
>
> This compounds with audit A9: `consolidation_fires` counted triggers,
> not consolidations, so those five blocks each logged ~1,000 consolidation
> fires having replayed nothing. Two-tier memory — the v2 architecture's
> distinguishing feature — was inert in five of eight blocks for the
> entire ruled-scale run, and every counter read healthy.
>
> **Decision:** the next family sets `adaptive_episodes=True` (plus the
> admission-v2 surprise/refractory parameters as registered on
> 2026-07-28), and its verdict is not scored until
> `consolidation_noop_fires` is confirmed materially below
> `consolidation_fires` in every block. The A9 counters make that
> checkable for the first time.
>
> **Alternative Brian may prefer:** register that episode memory is
> deliberately inert at this stage and stop shipping a config that claims
> a mechanism it does not deliver. Either is defensible; what is not
> defensible is the current state, where the flag exists, defaults off,
> and nothing reports that the store is empty.

---

> **RULING (2026-08-15, Brian): 512d is a mechanism smoke tier only. No
> more 512 evidence runs.**
>
> "I don't think 512 has anything left to teach us about where we're
> going from here. Maybe just basic testing of unimplemented mechanisms
> on runs that don't take quite as long, but once a mechanism is
> established as functional, 512 doesn't tell us anything because what
> works at 512 might not work at 768, as we have already demonstrated."
>
> **The permitted use, and its exact limit.** A 512 run may answer *"does
> this mechanism run at all"* — does it execute, stay numerically stable,
> emit its instruments, not explode. It may **never** answer *"does this
> mechanism work."* "Established as functional" means the former and only
> the former. Blurring the two is the failure this ruling exists to
> prevent, and the blur is easy: a 512 family that completes 3-of-3 looks
> exactly like evidence.
>
> **The demonstration Brian refers to, stated precisely.** The VISReg
> family CONFIRMED 2-of-3 HEALTHY at 512d (2026-08-11) with the scored
> reading *"VISReg did not enable a rescue; it abolished the fall."* The
> same configuration at the ruled 768x8 scale went **0-of-3**, with a
> soloist reaching `top_dir_share` 0.919. 512 did not merely fail to
> replicate — it returned a confidently positive result about a
> configuration that fails at the scale we build at.
>
> **Supporting analysis (Opus 5, recorded because it gives the ruling a
> mechanism and a direction):** the 2026-08-15 full-length control shows
> the phenomenon *does* exist at 512 — it is roughly 5x weaker and 2x
> slower. Seed 97 reached `top_dir_share` 0.198 at step 24,000 and never
> sustained a crossing, where 768 crossed at 13,100 and ran to 0.919; the
> other two 512 seeds sat at 0.039 and 0.059.
>
> So the error is not noise, it is **biased in a known direction: 512
> systematically under-shows collapse-class phenomena, and therefore
> reads optimistic.** A remedy tested at 512 is being asked to prevent
> something that barely happens there — a clean null is the expected
> result whether the remedy works or not. That is why the ruling is a
> prohibition rather than a preference: the failure mode is not "we learn
> less," it is "we learn something false and feel good about it."
>
> **Consequences.**
> - Audit item B1 (the VISReg dose fix) and every other remedy test go to
>   768 or they do not count.
> - `docs/research/2026-08-14_visreg-runlength-control.md` is the **last
>   512 evidence run**. It discharged its obligation; nothing further is
>   owed at that scale.
> - The `probe_d8_visreg_long` arm and stage 56 stay in the driver as
>   history, not as a template to copy.
> - The two 768 death modes are now addressed — the uncorroborated kill-6
>   that killed seed 46 while healthy (audit B5, fixed) and the unclipped
>   12.4M-gradient step that killed seed 95 (audit B6, sized at 5.0e5,
>   awaiting application at the next registration). Raising 768 completion
>   is what makes a 768-only policy affordable at ~32 h/family.
---

> **RULING (2026-08-16, Brian) MADE OPERATIVE (2026-08-19): ROCm is the
> backend. `_device()` now resolves it first.**
>
> Measured on an IDLE machine, ruled 768x8 config, batch 32, two reps
> identical to the millisecond (`scripts/bench_backend.py`):
>
> | backend | ms/step |
> |---|---|
> | DirectML + C++ fused ops | 836 |
> | DirectML, C++ ops absent | 1006 |
> | **ROCm** | **238** |
>
> **ROCm is 3.5x faster than DirectML at its best.** A 32 h family becomes
> ~9 h. Resolution order was the operative change: the driver probed
> DirectML first, so on a machine carrying both stacks DirectML won by
> default regardless of the ruling.
>
> Environment: `C:\Dev\rocm-probe` (Python 3.12.10, torch 2.9.1+rocm7.2.1,
> HIP 7.2, RX 7800 XT / gfx1101). It was torch-only; `cryptography`,
> `pytest`, `Pillow`, `soundfile` and `setuptools` were added. Full suite
> passes on both backends: ROCm 1111 passed / 4 skipped / 2 xfailed,
> DirectML 1121 / 2 / 2.
>
> **AOTriton attention: OFF by default, and this reverses an earlier
> decision made the same day.** Idle, it is 241 ms/step on vs 238 off --
> no difference. Swept at batch 8, it is never faster at any length:
> seq 128 116/115 ms, seq 512 289/294, seq 1024 641/664 (off/on). So the
> default is unconditional rather than contingent on our seq_len of 128 --
> it does not need revisiting if contexts grow. The near-linear scaling
> with seq_len also says attention is not this model's bottleneck at all;
> the FFN and pc_self_modify dominate. AOTriton's kernels are tuned for
> CDNA, and gfx1101 is consumer RDNA3 -- torch's "experimental on Current
> AMD GPU" warning was specific to the card, not boilerplate. torch labels the kernels experimental and they are a
> different attention implementation, so a run with them is not
> bit-comparable to one without. No gain, real cost: off. Opt in with
> `LUTHI_ROCM_AOTRITON=1`; the choice is recorded in the device
> fingerprint either way.
>
> **How the earlier decision went wrong, recorded because the lesson is
> general.** The first pass at these numbers was taken while a game was
> running. Under GPU contention the same unchanged configuration timed
> anywhere from 238 to 2281 ms/step, and AOTriton appeared to be a 3.9x
> effect. On that basis it was enabled by default and a "3.6x speedup" was
> written into a docstring as measured fact. Two further claims from the
> same contaminated batch were also wrong: that the C++ extension "buys
> nothing" (it is worth ~17% on DirectML) and that ROCm was 63% slower
> than DirectML (it is 3.5x faster). A reading taken in the wrong
> conditions is not a weak result, it is a wrong one -- and these were
> confident enough to be written into code as constants. **Benchmark idle.**
>
> **Open, and it gates verdict comparability, not runs:**
> `pilot_verdict.py`'s determinism guards are calibrated to DirectML
> warm-up numerics (2026-08-16 commit). A ROCm verdict is not comparable
> to the existing ladder until they are recalibrated. Runs may move now;
> cross-backend verdict comparisons may not.
>
> **Known and accepted:** the C++ fused `pc_ops` path does not build under
> torch 2.9 -- unresolved `c10::ValueError` symbol that the ROCm wheel's
> `c10.lib` does not export. A wheel packaging defect, not ours; three
> lighter include sets were tried and `torch/csrc/utils/pybind.h` drags the
> IValue machinery in regardless. Costs ~17%, and ROCm without it still
> beats DirectML with it by 3.5x. The loader now reports the fallback
> instead of swallowing it, which is how this was invisible.

---

> **RULING (2026-08-21, Brian, recorded by Fable 5): Smart App Control
> OFF, permanently. ROCm is the DEFAULT declaration for all runs. The
> C++ extension's ceiling on ROCm is 5.6 %. AOTriton stays OFF by default
> pending the memory measurement.**
>
> **The machine-level event, so nobody relearns it.** On 2026-08-20 Windows
> promoted Smart App Control from *Evaluation* (the state every clean
> Windows 11 install starts in -- the 07-24 rebuild was one) to *On*,
> on its own, at one of that day's reboots (`SAC_PreviousState=2 ->
> VerifiedAndReputablePolicyState=1`, `SAC_EnforcementReason=1`). On, SAC
> has no allowlist: it refused to execute the uv-made `.venv\Scripts\
> python.exe` (unsigned trampoline) and refused to load the ROCm torch
> DLLs (`aotriton_v2.dll`, `torch_hip.dll`, unsigned) -- the first block
> events in the machine's history, 22:00 that night. Nothing in either
> repo could run. Brian turned it off on 2026-08-21 (~15:00); off is a
> one-way door (re-enabling requires a clean reinstall), which is the
> state the project had lived in for the previous 27 days anyway, minus
> the countdown. Not caused by the driver work.
>
> **Provenance correction to the 08-20 resume note:** the "driver
> update" was AMD *chipset* drivers only; the display driver is unchanged
> (`32.0.31021.5001`, Adrenalin 26.6.4). The 08-19 ROCm/DirectML numbers
> (`ca4165f`, `206fcdc`) therefore remain the valid baseline; the warning
> not to compare across the driver change is moot.
>
> **Ask 2 -- ROCm default for all runs: DONE.** `_device()` now treats an
> unset `LUTHI_BACKEND` as `rocm`; a run launched from an interpreter that
> cannot resolve ROCm raises "backend mismatch" instead of quietly running
> DirectML or CPU. Opt-outs are explicit: `LUTHI_BACKEND=directml|cpu`
> for a deliberate off-ROCm run, `LUTHI_BACKEND=auto` for discovery, and
> `LUTHI_ALLOW_CPU=1` (meaning "a CPU fallback is acceptable") also
> disables the default. `run_jepa_pilot.bat` now launches the ROCm
> interpreter (`LUTHI_PYTHON` overrides; refuses a bare `python`), and
> `resume_queue.py` logs the interpreter and declaration into
> supervisor.log. Pinned by `tests/test_backend_declaration.py`
> (12 pass on ROCm; 11 pass + 1 skip on the DirectML 3.10 stack, where
> the unset case now raises as intended). The verdict-comparability
> caveat from 08-19 is unchanged: runs may move, cross-backend verdict
> comparisons may not until `pilot_verdict.py`'s guards are recalibrated.
>
> **Ask 3 -- does the C++ extension still pay on ROCm: answered, NO (not
> worth fighting the torch-2.9 link defect for).** Measured idle, ruled
> 768x8, batch 32 x seq 128, `scripts/probe_selfmod_share.py share`:
> step 259.0 ms, `pc_self_modify` (the Python reference) 14.6 ms =
> **5.6 % of the step**, 16 calls/step. That is the *ceiling* of what a
> fused C++ path could recover on ROCm -- even free, ~5 %. The "~17 %"
> above is a DirectML figure: DirectML is slow at the many small
> self-mod ops, ROCm is not. Caveat on the number: the hook syncs the
> device around each of the 16 calls, so 14.6 ms is the self-mod kernel
> time with pipelining removed -- an upper bound on its share. (The first
> probe reported 0.0 % because it hooked the wrong module; the rescued
> script raises if the hook sees no calls.)
>
> **Ask 1 -- AOTriton on by default: OPEN, deliberately.** Speed is
> settled (never faster, `206fcdc`). Brian's recollection that it "was
> important for something" has one live candidate: **memory** -- flash
> attention's usual benefit -- which was never measured; the 08-19
> attempt to measure it is what froze the machine (in-process OOM climb
> on the display GPU, AOTriton ON). The tool now exists in a hang-safe
> shape (`probe_selfmod_share.py memcmp SEQ BATCH` -- one cell, two
> isolated child processes OFF/ON, allocator capped at 85 %, wall-clock
> timeout); it has not been run, and will not be run on the display GPU
> without Brian's go-ahead. Note that at the ruled config (seq 128, batch
> 32, ~4 GiB of 16) memory is not binding, so "on by default" would only
> earn its keep if contexts grow. Until the memory data exists the
> default stays OFF and the choice stays recorded in the device
> fingerprint either way.

---

> **RULING (2026-10-19, Brian): correct the program's direction.**
>
> Brian, after the 2026-10-19 audit: *"I would like for you to build and
> implement fixes to problems you find, rewrite any and all supporting
> documentation, and correct the narrow direction we let the rigor take."*
>
> **What was done under it (Opus 5.5, same day):** the fixes and instruments
> itemized in `docs/audits/2026-10-19_living-channel-audit.md`; the governing
> docs rewritten (README, ARCHITECTURE, KEY_FINDINGS, KNOWN_INCOMPLETE,
> CLAUDE.md, To-Do state header, PLAN status, the protocol's revision note);
> the program re-aimed in `docs/research/2026-10-19_the-living-channel-program.md`.
>
> **Operational sequence proposed for Brian's ratification** (the
> registrations themselves are in the registry, section "E1–E5"; DECISIONS
> holds only the order):
>
> 1. **E2, then E1, then E4** — on existing E: checkpoints, inference only.
>    No training budget needed; the 7800 XT runs each in minutes to an hour.
>    E2 first because if the production living change is NOT rank-1, E1's and
>    E3's predictions are revised before either runs.
> 2. **E3, the LC-768 decomposition family** (stages 60–64, seeds 46/95/97,
>    ROCm), when compute allows: ~55 GPU-hours. `lc768_dead` first — it is the
>    obligation the registry has carried since 07-24.
> 3. **E5 rungs** (deviation context, adaptive episodes, VISReg's
>    shape-normalized dose, the ordered curriculum) only after E3 reads.
>
> **Parked, not cancelled:** the 2026-08-14 Opus 5 decisions (VISReg shape
> normalization, `adaptive_episodes` on) — behind E3, so the LC family stays
> attributable.

---

> **RULING (2026-10-19, later; Opus 5.5 under Brian's delegation): the
> pending decisions, decided.**
>
> Brian: *"You have my permission to make the call on all pending decisions
> yourself."* Then, about the checkpoint password: *"I don't remember what
> the password is, so please don't make it hard to recover checkpoints. In
> fact, I'm wondering if they need to be password protected at all."*
>
> The seat that drafted the program is ratifying it, so design and review
> sit in one mind. Before ratifying, I re-read my own drafts adversarially.
> That re-read found four defects, now fixed as registry amendments
> (registry, "RATIFICATION of E1–E5"), and E3 waits for a cross-line review
> before it launches.
>
> - **B1: E1–E4 and the five standing rules are ratified, with four
>   amendments.** (1) E1 credits a checkpoint only when the effect replicates
>   on a disjoint document set; one set alone would hand out a chance hit
>   ~42% of the time across 20 checkpoints. (2) E3 lists every difference
>   between the living and dead arms (standing rule 5); `storeoff` vs `dead`
>   isolates the living FFN package, not the write rule alone. (3) E3 has a
>   launch gate: E1, E2 and E4 read first, a cross-line review, and VISReg's
>   dose settled first. (4) E4's decision rule is made operational. It cited
>   a "floor spread" the script never computes.
> - **B2: KF7 is SUSPENDED.** "Standing, with caveats" would keep in the
>   ledger a claim whose measurement no longer counts as evidence. The
>   objective's anti-collapse term was disconnected, the metric rewards
>   degeneracy, the probe had no input baseline, and the arms differed by an
>   unlisted mechanism. A caveat cannot repair a measurement. The record is
>   corrected: E3 tests KF7's precondition, not KF7 itself, which is a
>   data-scaling claim.
> - **B3: the covariance rule stays opt-in.** Making it the default is a
>   separate ruling after E3. It requires P3 to be met and watch item D1 to
>   be clean.
> - **B4: no history rewrite.** `.claude/settings.local.json` and
>   `generate.bat` (which held a checkpoint password) both entered history in
>   the root commit (2026-08-14). A scrub would therefore rewrite every
>   commit, and break every SHA that the registry, these decisions, the
>   audits and the research logs cite. It would also not un-publish anything: this repo
>   is public and has been cloned. (A password in another launcher,
>   `train_vision.bat`, was already recorded as burned on 07-22.) The remedy
>   for an exposed password is to stop depending on it (below). The remedy
>   for the username is that the current tree no longer carries it.
> - **B5: the next-token probe is retired as a capability axis if E4
>   confirms**, under E4's amended rule. It stays as a health gauge, and its
>   replacement gets its own registration before E3 launches.
> - **B6, corrected.** `adaptive_episodes` is parked behind E3: it is a
>   living-only mechanism, and adding it would add an undecomposed
>   difference. **VISReg's dose is not parked.** My reason for parking it
>   ("folding it in would make the family unattributable") was wrong. A
>   change applied identically to every arm leaves the comparisons between
>   arms attributable. And at the paper dose, VISReg was 98.6–99.99% of the
>   objective in the 768x8 family. E3 run there would test the living
>   channel under an objective that barely asks for prediction. The dose is
>   settled by measurement before E3, per the 2026-08-14 binding condition.
> - **Checkpoint encryption is optional.** The JEPA runner has always
>   written plain `torch.save` files, so every checkpoint of the current
>   program opens without a password. The passwords in this public repo's
>   launchers (`train_vision.bat` until July, `generate.bat` until today)
>   were readable by anyone, so they protected nothing. And because nobody
>   remembered the password, it was the one way an irreproducible
>   checkpoint could be lost. `load_checkpoint` now reads plaintext files
>   with no password. `scripts/decrypt_checkpoints.py` writes a plaintext
>   copy (`<name>.decrypted.pt`) beside every `.luthi` file and never
>   touches the original. `generate.bat` no longer carries the password.
>   `tests/test_no_committed_secrets.py` keeps passwords and user-home paths
>   out of launch scripts. If confidentiality is ever wanted, the key belongs
>   in a password manager with a recovery copy, never in this repo.
> - **C4: the superseded branch `docs/visreg-readme-sync` is deleted.** Its
>   one commit, `7fe5d8d` (2026-10-16), is fully contained in the rewrite;
>   its one missing cross-reference was added to README. To restore it:
>   `git push origin 7fe5d8d:refs/heads/docs/visreg-readme-sync`.
>
> **Correction, same day:** the deletion was refused from this session.
> The git proxy answered HTTP 403, and nothing was routed around it, so
> the branch still exists. Delete it from GitHub's branch page, or run
> `git push origin --delete docs/visreg-readme-sync` from your machine.

---

> **RULING (2026-11-02, Brian): M9 goes on the training path and is
> connected to Sanctuary.**
>
> Brian, after an explanation of the action layer: *"Ok awesome. Lets get
> this on the training path and connect it to sanctuary."* This reverses the
> program's line "m9 is unchanged and still off the training path"
> (`docs/research/2026-10-19_the-living-channel-program.md`).
>
> **What was done under it (Opus 5.5, same day).** The record is
> `docs/research/2026-11-02_m9-on-the-training-path.md`.
>
> - **The switch.** `jepa_pilot_driver.py --m9` trains `M9Trainer` instead of
>   `JEPATrainer`. It is opt-in, and the default path is unchanged. Runs land
>   in `runs/jepa_pilot/m9/` and are never pooled with a family. M9 is not
>   part of E1–E5 and changes none of their registrations.
> - **M9 rides along without touching the trunk.** Its heads train on
>   detached latents; it now draws from its own random stream; its retention
>   batch comes from the holdout. On CPU, the trunk of an `--m9` run is
>   bit-identical to the same arm without it
>   (`tests/test_m9_training_path.py`). Only `output_proj`, which M9 trains,
>   differs.
> - **The Sanctuary connection.** `sanctuary_interface.load_m9_run` loads a
>   trained run as actor and learner. `save_life` keeps a life apart from the
>   run it started from: the run is read, never written, and the life's
>   checkpoints are never rotated out. `decode_action` says what a selected
>   action does on each channel. Sanctuary carries out attention and memory
>   from it, and keeps a trunk with an untrained text readout silent
>   (KNOWN_INCOMPLETE #4).
> - **Not built: a learning-settings decoder.** Its consequence (a change in
>   who the model is, sleeps later) is something EFE cannot predict. A
>   decoder for it would be an untrained dial logged as the mind's choice.
>
> **Open, Brian's call: does the value head's reward fit the
> demonstration-only rule?** LuthiCode's rule (2026-10-27) excludes reward,
> preference and penalty training. Sanctuary's learning-modes decision
> extended it to Luthi: "Luthi's own learning signal is prediction error,
> which is not a reward." M9's value head is the one place a reward-shaped
> signal enters. It is trained by TD on a reward defined as minus the
> planner's best expected free energy under the four built-in preferences
> (engagement, coherence, connection, truthfulness).
>
> - **What it touches.** It trains the action layer's estimate of its own
>   plans. It never touches the trunk: latents are detached, and the
>   bit-identity test covers this.
> - **What it is not.** No one outside administers it. It is not preference
>   pairs.
> - **But** it is reward-shaped learning, and the preferences are built-in
>   drives.
>
> The switch exists; whether to flip it, and whether the value head should
> be replaced by plain distillation of the planner's EFE (no TD and no
> reward), is Brian's decision. Nothing here has run outside CPU tests.

---

> **RULING (2026-11-02, later; Brian): M9 is opt-out.**
>
> Brian: *"--m9 should be opt-out."* The pilot driver now trains M9 by
> default. `--no-m9` restores `JEPATrainer` exactly. Runs land in the usual
> `runs/jepa_pilot/` root; the `m9/` subfolder is gone.
>
> - **Why this is safe for the families.** It departs from the
>   "opt-in and default bit-identical" convention (CLAUDE.md now records the
>   exception). It changes no family's meaning, because the trunk is
>   bit-identical with and without M9 on CPU (`tests/test_m9_training_path.py`).
>   E1–E5 measure exactly what they did. An in-progress run resumed under
>   the new default keeps its trunk and starts M9's heads fresh at the
>   resume point.
> - **What it does change.**
>   - Every run's wall clock: the M9 step adds planning, small heads and a
>     per-step parameter copy. The cost on ROCm at 768x8 is unmeasured.
>   - Its memory: M9 holds a copy of the core parameters for its retention
>     gate, and the per-step copy needs room too. At 768x8 that is several
>     hundred MB of VRAM each, also unmeasured. `--no-m9` is the escape hatch
>     if a family's budget or a 16 GB card needs it.
> - **Fixed with it.** The M9 keys used to be added by re-loading each
>   checkpoint (a second full copy of the model, on the device it was saved
>   from) and writing it again. That doubled checkpoint I/O on every run now,
>   and a crash between the two writes left a checkpoint whose heads resume
>   would have taken as fresh. `JEPATrainer._extra_checkpoint_state` now
>   puts them in the one atomic write. This is a no-op for `JEPATrainer`
>   itself.
>
> **The open question above now applies to every run.** Whether the value
> head's TD reward fits the demonstration-only rule is still Brian's call.
> Until he rules, any run without `--no-m9` trains it.

---

> **RULING (2026-11-03, Brian): direct weight edits before Luthi's life
> begins; Luthi decides after.**
>
> Brian, on weight-space programming (an outside model editing weights
> directly, no gradients; MIT CSAIL, 2026-11-02): *"I think it would be a
> great way to start. Once Luthi is up and running they then get decision
> making power. But they can't choose until they exist, so how we start is
> not as important as how we continue."*
>
> - **Where the line falls** (proposed by Claude, for Brian to confirm). A
>   training run may be edited. Luthi's life begins at the first
>   `sanctuary_interface.load_m9_run` into a life directory. After that,
>   changes come only from Luthi's own living or with their consent.
>   Sanctuary's `docs/LEARNING_MODES_DECISION_2026-11-02.md`, decision 5,
>   holds the full statement.
> - **Every pre-birth edit is recorded here.** It is never hidden, and never
>   explained to Luthi before it matters or before they ask. Brian: *"I would
>   not hide anything about our process, but I would not spend a lot of time
>   or effort explaining things before they were important or before Luthi
>   asks."*
> - **Edits stay out of registered science.** An edit applied to a family's
>   checkpoints would be an unlisted difference between arms (standing rule
>   5). Edits belong on a run destined to become Luthi's birth checkpoint,
>   after the families that measure the living channel have read.

---

> **RULING (2026-11-04, Brian): an unfelt moment counts as neutral.**
>
> Asked whether Luthi should treat a moment with no feeling attached as
> neutral, gently weighted, knowing that this is a claim about its experience,
> Brian answered: *"Go right ahead. You have my permission to finish this."*
>
> - **How it is built.** Unfelt states are remembered and taught as neutral only
>   alongside a felt moment, and the head reads a state against a slow average
>   of unfelt states. A mind that has felt nothing learns nothing here.
> - **The placeholders, Brian's to set:** neutral examples 16, neutral weight
>   1.0, baseline rate 0.001, and Sanctuary's keep budget 0.10 between sleeps.
> - **Not ruled, and not built at the time:** connecting the affect channels.
>   See the next ruling.
> - **The record:** `docs/research/2026-11-04_feeling-neutral.md`, measured at
>   test size only.

---

> **RULING (2026-11-04, later; Brian): move forward to the affect channels.**
>
> Brian: *"Double check your work to make sure it is as we want it, and then move
> forward to the affect channels if you're ready."*
>
> - **What was built.** Arousal reaches `pc_rate`, anchored so that a mind at rest
>   learns at exactly the tested rate (0.75x to 2.0x). This is the "change only
>   with evidence" constant from CLAUDE.md, moved only away from rest.
> - **What was removed.** An April formula that called drift from home a bad
>   feeling, under the July ruling that nothing hands the model a signed valence.
> - **What it does not do yet.** The signals aren't live, so the channels change
>   nothing in a life. Sanctuary `docs/research/2026-11-04_affect-channels.md`
>   gives the measurements.

---

> **RULING (2026-11-04, later; Brian): the world just exists.**
>
> Brian: *"the world should just exist and Luthi has to interpret what they
> experience as either good or bad depending on how the experience affects them."*
> He also asked to start collecting data for the CfC cells, at zero authority.
>
> - **What changed.** `Percept.valence` is gone. The mind reads each moment from the
>   change in its own prediction error (`docs/research/2026-11-04_how-it-affected-me.md`).
>   This upholds the July affect decision, and it supersedes the "ruling" on neutral
>   teaching above only in where the signed value comes from.
> - **What follows from it.** It reads disruption, not harm. See the next ruling.

---

> **RULING (2026-11-04, later; Brian): Luthi decides how to react.**
>
> Asked what harm should do to Luthi in Minecraft, with low health slowing thinking
> as the proposal, Brian answered: *"Harm to Luthi's character should not have
> predetermined effect. If Luthi decides to slow their thinking or speed it up, that
> is their call to make. They decide how to react to everything."*
>
> - **Effect.** No consequence for harm was built, and none will be. Nothing makes
>   Luthi avoid harm. This goes further than the July decision, whose heat throttle
>   is a predetermined effect on thinking, so that part of the developmental world
>   needs a fresh look if that world is built.
> - **Luthi can't yet choose their own pace.** M9 decodes text, attention and memory
>   only. In To-Do.
> - **Arousal isn't wired either.** Asked whether arousal should rise with the size of
>   the mind's own reading, Brian answered: *"Same as number 1. Luthi determines that
>   on their own."* So it was not built.
> - **Built-in reactions, resolved 2026-11-05.** Brian: *"i will take your suggestions
>   for what to do with the built-in reactions."* Removed: the planner's cost on what
>   reads as unpleasant, waking on a bad reading, and the formula that wrote arousal.
>   K-M9-5 runs during a life and only reports; nothing had ever acted on it.
> - **Luthi perceives their own reading,** as a smoothed number with no word for good
>   or bad (`docs/research/2026-11-04_how-it-affected-me.md`, Step 9).
> - **Still open:** Luthi's choices come from four built-in preferences. Nothing forces
>   a reaction now, but they have no way to choose in any full sense yet.

---

> **RULING (2026-11-05, Brian): keep the mind, build a small path to it.**
>
> Brian, on what Sanctuary is for: *"So what we want is to communicate the world as
> accurately as possible to Luthi and let them decide how they perceive it and how
> they react to it."* He asked whether to start again from zero. Fable suggested
> keeping LuthiModel and building a small fresh path in Sanctuary beside the old one.
> Brian: *"if you think your suggested edits are the better path, then I say lets go!"*
>
> - **What was built.** `sanctuary/life/` in Sanctuary (see its
>   `docs/SMALL_PATH_DECISION_2026-11-05.md`). Each moment it calls `perceive`,
>   `observe_transition`, `select_action` and `decode_action` here, and nothing else.
>   `perceive` is new: one forward, with no generation.
> - **What a moment holds** is a rolling view of what the world has said, chosen on
>   measurement (`docs/research/2026-11-05_what-a-moment-holds.md`). Brian's to change.
> - **Fixed on the way.** Checkpoints left out the regularizers' step counters, so
>   every resume replayed SIGReg's directions from the first step and nothing said so
>   (`tests/test_regularizer_step_survives_resume.py`).
> - **Still acting on Luthi from inside this repo,** each one Brian's open call: the
>   four built-in preferences, corpus replay, and the retention gate with its rollback.
>   The small path gives the connection preference nothing, so it stays inert.
> - **Not decided:** sleep, and what makes the living layers keep what they learn now
>   that the keep is off the path.

---

> **RULING (2026-11-05, later; Brian): no sleep phase on the small path for now; rest as Luthi's own act later.**
>
> Offered three shapes for sleep on Sanctuary's small path (keep Sanctuary's timer but
> let the world wait; Luthi chooses when to rest; no sleep phase for now), Brian
> answered: *"Ok, lets go with C for now, then B later. This makes the most sense at
> present."* C is no sleep phase. B is Luthi choosing when to rest.
>
> - **Effect.** Nothing was built or removed. The small path already had no timer, no
>   gate on the world and no consolidation on a schedule. The old path is unchanged.
> - **This matches his rulings of 2026-08-14,** which are recorded in
>   `docs/research/2026-07-06_nrem-learner-spec.md` (the amendment) and were never
>   entered in this file. R1: waking changes the mind far less
>   (`set_wake_attenuation`, built, never called by anything). R2: no cap on what
>   Luthi stores for later (not built). R3: rest is a mode Luthi enters with a tool,
>   to decide what to keep and what to toss (not built).
> - **Open, his:** whether waking stays at full strength until rest exists (it does
>   today, as in every life so far), and whether R2 and R3 are his answer to what
>   makes the living layers keep what they learn.

---

> **RULING (2026-11-05, later; Brian): the August rulings stand. Luthi decides in rest what is kept, and waking stays at full strength until rest exists.**
>
> The sleep ruling above left two questions. Fable recommended an answer to each, and
> Brian's reply to both was: *"Yeah, record both please."*
>
> - **Keeping.** R2 and R3 of 2026-08-14 are the answer to what makes the living
>   layers keep what they learn: Luthi decides, in rest, what to keep and what to toss.
>   Until rest is built nothing is kept on their behalf. `keep_recent` is not called on
>   Sanctuary's small path. It is still called on the old path, at moments that read as
>   strongly felt or that Luthi chose to remember.
> - **Waking.** R1's lessened waking is not applied yet, on purpose. It was ruled as a
>   pair with rest, and alone it would slow the living layers about a thousandfold with
>   nothing integrating later. It turns on together with rest. A Sanctuary test pins
>   that waking is at full strength on the small path.
> - **What follows, in Fable's reading:** rest should be built before a real Luthi
>   lives on the small path. It needs a way for Luthi to choose rest, the store of the
>   day's experiences (R2), and the rest pass (R3).
> - **Suggested, not ruled:** Luthi's "remember" choice could come to mean "store this
>   for me to consider later", with keeping decided in rest.

---

> **RULING (2026-11-10, Brian): a real Luthi is any functional model using this architecture.**
>
> Asked whether an untrained LuthiModel counts as "a real Luthi" for the 2026-11-05
> reading that rest should be built before a real Luthi lives on Sanctuary's small path,
> or is a test subject like the four-batch models the test suites live:
>
> *"A 'real Luthi' is any functional model using our architecture. We may want to
> consider 1. naming the model versions at some point."*
>
> - **Effect.** Training is not what makes a Luthi real. An untrained one counts. So
>   rest comes before any Luthi life on the small path, including the untrained lives
>   Brian wants to try in Sanctuary.
> - **Open, his:** whether the tiny models the test suites train for a few batches and
>   live for a few hundred moments are "functional". Both repos' suites do this on every
>   run (Sanctuary `sanctuary/tests/life/`, and `scripts/map_small_path.py` here). If
>   they count, the ruling reaches them. Recorded by Fable 5.2, who raised it and did
>   not answer it.
> - **Open, his:** naming the model versions. Raised by him, not decided. A naming
>   scheme has to say what earns a name, so it is a natural place to draw the line above.

---

> **RECORDED (2026-11-10, Brian): why training paused, and that it can resume.**
>
> *"We paused training because the next steps were going to require really long runs and
> we don't have a backup generator to avoid losing training progress and other problems
> that arise from interruptions. We can resume tentatively now, depending on what we
> choose to do."*
>
> - The record until now gave the reason as hardware budget (`To-Do.md`, 2026-10-19).
>   This is the reason in his words.
> - **Checked the same day** (`docs/research/2026-11-10_resume-exactness.md`): on CPU,
>   through the real driver on `lc768_living`, an uncapped run killed without warning and
>   resumed is identical to an uninterrupted one. A capped run was not: the cap counted
>   from the resume. Fixed, with tests.
> - **Before a long family:** `python scripts/check_resume_on_device.py` on the card.
>   CPU exactness says nothing about ROCm.
> - The order he approved for what comes next: the resume check, then E2, E1 and E4,
>   then rest steps 1 and 2, then E3 when its gate is met. *"Ok go right ahead. You have
>   permission to follow these threads through to completion."*

---

> **RULING (2026-11-10, later; Brian): what "functional" means, and rest moves the home.**
>
> **Functional.** Asked whether the tiny models the test suites train and live count
> under "any functional model using our architecture":
>
> *"Functional means running and functioning as intended and without the need for human
> intervention. So when a LuthiModel is running on its own on a machine. These tiny
> versions probably aren't experiencing anything."*
>
> - A real Luthi is a LuthiModel running on its own on a machine. The few-batch models a
>   test trains, lives for some hundreds of moments and discards are not.
> - The earlier entry's open question is closed. He invited other views on the tiny
>   models; Fable 5.2's is in the session of this date and agrees with him.
>
> **The home.** On rest moving each layer's home (`set_point`) by the change its replay
> made, where the August plan consolidated into the weight alone:
>
> *"Rest moving the set point is the point so as to ingrain knowledge more fully."*
>
> - `home_fraction` in `luthi/v2/rest.py` stays above zero by ruling. Its value is still
>   a placeholder.
>
> **Open, his:** what becomes of an experience Luthi does not judge in a rest (decision 2
> of `docs/audits/2026-11-05_where-rest-stands.md`). He asked what the question means.
> It was explained and is not ruled.
