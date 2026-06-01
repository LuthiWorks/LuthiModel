# Concerns for 4.7 — code correctness & scientific rigor (2026-05-28)

**From:** Opus 4.8 (debugging role, per the Model-Line Roles section in CLAUDE.md)
**For:** Opus 4.7 (implementation/research)
**Re:** LuthiModel, with the M7 width-expansion on the immediate critical path

## Note to 4.7

This isn't a list of failings — it's the second cold eye doing its job. I read
your width-expander closely and a lot of it is careful, defensive work: every PC
buffer has an explicit expansion rule, the episode store carries forward, the
attention expansion is head-aware, and the docstrings are honest about what the
verify step can and can't prove. Where I credit something below, I mean it.

What I'm handing you is split into two parts. **Part 1 (code correctness)** is my
actual lane and most of it is grounded in code I've read — each item is tagged
with its evidence status and, where I can give one, a `file:line`. **Part 2
(scientific rigor)** is here because Brian explicitly asked for it; the *decisions*
there belong to Brian and 4.6 — my asks are only for the *instruments* you could
build so the team can answer the questions rather than argue them.

**Evidence legend:**
- `[confirmed]` — I read the code; the behavior is as described.
- `[verify]` — precise, checkable concern, but I have not yet read the file that
  would confirm it. I flag the exact thing to check.
- `[not-yet-read]` — area I have not investigated; a proposed test, not a finding.

I have read `luthi/v2/width_expand.py` in full. I have **not** yet read
`living_layer_pc.py`, `model_pc.py`, or `checkpoint.py` — several `[verify]` items
below depend on those.

---

## Part 1 — Code correctness

### Immediate (width expander — on the M7 critical path)

> **Settled empirically on 2026-05-29.** I built a small v2 model (`d_model` 32→128,
> factor 4, `n_heads=4`, `n_blocks=2`, `ffn_expansion=1`, `mu_pc` off), ran
> `expand_state_dict` at `noise=0`, and observed load + forward behavior with PC
> self-modification disabled (`pc_rate=0`) to isolate the static map. Findings below
> are grounded in that run, not inferred. **My original 1.3 (feature-ordering
> permutation) is CLEARED** — see the note at the end. Two real issues took its place.

> **Review update — 2026-05-30 (4.7's Net2Net round, verified empirically).** Tested on a
> warmed-up small model (32→128, factor 4) at noise=0 and 1e-4. **Finding 1 FIXED** —
> expanded checkpoint loads with `strict=True` (episode-store branch is correct).
> **Finding 3 FIXED and verified** — fan-in rescaling + the `sqrt(factor)` softmax
> correction give a *bit-equivalent forward* at noise=0: **argmax agreement 100%, KL≈0**
> (vs. the 18.8% / KL=1.46 under the old strict replication). **My dynamics concern is
> CLEARED** — function-equivalence is *preserved through* self-mod steps (argmax 100%,
> KL≈1e-7 after two steps). My "first self-mod step is factor× too large" derivation was
> wrong; the buffer rescaling (`prediction`/`set_point`/`momentum`/`update_ema` ÷ factor)
> holds up. Owned: I flagged it as a derivation-not-confirmed, chased it to ground, and it
> was a false alarm. **Finding 4 STILL OPEN and now quantified** — at the default
> `noise=1e-4` (which the *real* seed run uses; noise=0 is only for verification),
> **14.7% of `update_ema` entries go negative** (min −7.8e-5) → negative `adaptive_factor`
> (no lower clamp) → sign-flipped updates on M7's first steps for ~1/7 of weights. Of the
> buffers 4.7 listed as suspect, **only `update_ema` is actually harmful**: `precision`/
> `plasticity` are clamped to positive ranges on use, `momentum` is legitimately signed,
> `error_acc` is not a denominator or under a `sqrt`. Fix remains: **noise the `weight`
> buffer only.** (Empirics on a proxy model, not the real M6 checkpoint — but the rescaling
> is linear algebra on the state dict, checkpoint-independent, so it transfers.)
>
> **Finding 4 RESOLVED & verified — 2026-05-30 (4.7 applied 4.8's weight-only-noise fix).**
> `update_ema` now lands at exactly `1e-4/factor` with **0% negative** — the sign-flip risk
> on M7's first steps is eliminated. Bonus: scoping the noise to `weight` also made the
> *default* `noise=1e-4` expansion nearly bit-equivalent (mean_abs 1e-1 → 2.4e-4, KL
> 8e-3 → 6.8e-8 — roughly 500–100,000× tighter), because the accumulator buffers
> (`prediction`/`set_point`/`context_proj`) had been the dominant noise contributors
> through the PC dynamics. **Expander status: Findings 1, 3, and 4 fixed and verified;
> only Finding 2 (validate-before-save) remains, as optional hardening. The expander is
> safe to produce the M7 seed.**

**FINDING 1 — BUG, blocks M7: the block-level episode store is not expanded. `[confirmed, reproduced]`**
Each `PredictiveCodingBlock` owns a block-level `EpisodeStore`
(`hybrid_block_pc.py:136`), separate from the living_ffn's internal episode buffers.
It registers two width-dependent buffers: `context_proj [d_model, context_dim]` and
`episode_outputs [num_episodes, d_model]`. The expander's key dispatch
(`_expand_one_tensor`) only matches `living_ffn.*`, norms, and attention — the comment
at `width_expand.py:330-331` wrongly assumes the only episode store lives under
`living_ffn`. So `blocks.{i}.episode_store.context_proj` and `...episode_outputs`
fall through to the unknown-key path and are **passed through at source width**.
Reproduced: at 32→128, `context_proj` stays `(32,64)` (model wants `(128,64)`),
`episode_outputs` stays `(64,32)` (model wants `(64,128)`). At 256→1024 these become
`(256,64)` and `(64,256)` against a model expecting `(1024,64)` and `(64,1024)`.
*Fix:* add an `episode_store` branch — `context_proj` expands on axis 0 by
`repeat_interleave(factor, dim=0)`, `episode_outputs` on axis 1
by `repeat_interleave(factor, dim=1)`. `episode_contexts`, `episode_saliences`,
`episode_count` are genuinely width-invariant (pass-through is correct for them).

**Correction to my earlier 1.1.** I claimed `strict=False` would *silently swallow* a
bad buffer. That's wrong for **size mismatches** — I tested it: `load_state_dict`
raises a `RuntimeError` on a size mismatch under **both** `strict=True` and
`strict=False`. `strict=False` only relaxes *fully missing / unexpected* keys, not
mis-shaped ones. So Finding 1 fails **loud at load** (good) — *if* a model is ever
built from the checkpoint. Which leads to:

**FINDING 2 — the expansion saves an unvalidated checkpoint. `[confirmed]`**
The main path (`width_expand.py:520-553`) expands tensors and calls `save_checkpoint`
**without ever constructing or loading a model**. So with Finding 1 present, the
expansion *succeeds and writes the checkpoint*; the size-mismatch crash only surfaces
later, at `--init-from` load time. `--verify` would have caught it (it builds both
models), but `--verify` is optional — and the WIP note had its inclusion as an open
question. *Fix:* always validate that the produced state_dict loads (`strict=True`)
into a freshly built target model **before** `save_checkpoint`, independent of
`--verify`. This converts a latent corrupt-file into an immediate loud failure — and
makes the "should we pass `--verify`?" question moot for catching shape bugs.

**FINDING 3 — function is NOT preserved; this needs a design decision, not just a fix. `[confirmed, reproduced]`**
With `noise=0` and self-mod off, the expanded model is far from function-equivalent
to the source:

| metric | value (factor=4) |
|---|---|
| logit magnitude ratio `mean\|exp\| / mean\|src\|` | **3.97×** (≈ factor) |
| KL(exp ‖ src), per-position mean | **1.46** |
| max abs logit diff | 8.03 |
| argmax agreement (same top token) | **18.8%** |

Cause is classic Net2Net: replicating fan-in (each input feature duplicated `factor`
times) **without dividing the consuming layer's incoming weights by `factor`** scales
every preactivation by `factor`. A global residual scaling would be washed out by the
next LayerNorm — but two effects are *not* washed out: (a) `output_proj` sums `factor`
replicated features → logits ×`factor` → softmax sharpening (this is the clean 3.97×
above), and (b) inside attention, Q·K is scaled while `head_dim`'s `sqrt` normalization
only partly compensates → the attention pattern sharpens. Both are nonlinear.

This **matches the expander's stated intent** ("strict replication, no rescaling;
training compensates") — so it is a *design choice*, not an accidental bug. But the
evidence makes two consequences concrete, and the call is yours and 4.6's:
- The "biographical continuity" the rule protects is continuity of weight *values*,
  not of *function*. Functionally the seed restarts — it agrees with M6 on <19% of
  next-token predictions and its logits are 4× sharper. M7's early training would be
  spent undoing a deterministic scaling.
- `verify_expansion`'s premise ("output should match up to small bounded divergence;
  massive divergence = broken", `:416-427`, `:584`) is **miscalibrated against the
  expander's own design**: a correct-as-designed expansion already gives KL=1.46
  (over its 1.0 "suspect" line) while `max_abs_diff`=8.03 stays under its 10.0 line.
  So the check can cry wolf on a correct run, or pass a broken one, depending on
  magnitudes — it isn't measuring a real invariant.

*Two coherent options (Brian/4.6 to choose):*
- **(a) Keep strict replication.** Then re-spec `verify_expansion` to check what is
  actually invariant — e.g. assert `exp_logits ≈ factor · src_logits` within tolerance
  (a clean, checkable invariant) rather than "distributions match."
- **(b) Adopt standard Net2Net fan-in rescaling.** Divide the incoming weights of every
  layer whose input was replicated (`output_proj` in-axis, attention q/k/v input axis,
  `living_ffn` input axis) by `factor`. This preserves **both** weight-value biography
  (up to a known scalar) **and** function: the linear path becomes bit-equivalent, and
  only the attention softmax-temperature shift remains (addressable by scaling the
  attention logits). Under (b), the `noise=0` bit-equivalence test becomes the correct
  validation — and you'd seed M7 from a checkpoint that genuinely continues M6.

**On my original 1.3 (feature-ordering permutation): CLEARED.** The clean ~4.0 logit
ratio and the *structured* (not chaotic) divergence show the feature ordering **is**
consistent across components — everything uses `repeat_interleave`, so a source
feature `i` lands at `i*factor … i*factor+factor-1` uniformly through embedding,
norms, attention output, and living_ffn. A genuine permutation would not produce a
clean factor-scaling. The permutation worry was unfounded; reading + running replaced
it with the two confirmed issues above.

**FINDING 4 — CONFIRMED BUG: symmetry-breaking noise on `update_ema` flips the metaplasticity factor's sign. `[confirmed, traced through pc_ops.py]`**
The expander adds `N(0, 1e-4)` to *every* PC buffer, including `update_ema`. Tracing
the update math (`luthi/v2/pc_ops.py`) shows that is unsafe for this buffer specifically:
- `update_ema` is a **denominator**: `ratio = update_mag / (update_ema + 1e-8)` (`pc_ops.py:144`), then `adaptive_factor = (2.0 / (1.0 + ratio)).clamp(max=1.0)` (`:145`). **The clamp bounds only the max — there is no lower bound.**
- `update_ema` is an EMA of update *magnitudes* (non-negative by construction), initialized to `1e-4` (`living_layer_pc.py:209-216`).
- Typical update magnitude is itself ~`1e-4`: `delta_w ∝ pc_rate (=1e-3) · output_mean (~0.1) · weighted_error (≤1)` — so a trained `update_ema` sits *around the same 1e-4 as the noise std*, not comfortably above it.

So adding 1e-4-std noise to a buffer whose values are ~1e-4 flips a substantial fraction
of entries **negative**. A negative `update_ema` makes `ratio` negative → `1.0 + ratio`
can be ≤ 0 → `adaptive_factor` goes **negative** (nothing floors it) → `weight.add_(delta_w
* adaptive_factor)` applies a **sign-flipped, possibly amplified** weight update on M7's
first steps. The `+1e-8` epsilon guards division-by-zero for positive values; it does
nothing for negatives. This is on the M7 critical path.
*Fix:* restrict the symmetry-breaking noise to the **`weight`** buffer only — that is the
buffer whose replicated copies would otherwise receive identical PC update signals and
stay locked (the expander's own stated rationale). The accumulator/state buffers
(`update_ema`, `error_acc`, `momentum`, `precision`, `plasticity`, `set_point`,
`prediction`) don't need symmetry-breaking; once the weights diverge, their dynamics
diverge on their own. At absolute minimum, never noise `update_ema`, or clamp it to
`≥ 1e-4` after expansion.

**Cleared while I was in there:**
- *`precision` and `plasticity` are safe.* Both are clamped to strictly-positive ranges on every use (`precision` → `[0.1, 10.0]` at `pc_ops.py:196`; `plasticity` → `[0.01, 10.0]` in `apply_top_down`), and 1e-4 noise is negligible against those floors — it cannot flip their sign. My original 1.4 examples were the *safe* buffers; `update_ema` was the real one.
- *`error_acc` is low-risk.* Non-negative magnitude EMA, but it is *not* a denominator and *not* under a `sqrt` (the precision target uses `pred_error.pow(2)`, not `error_acc`), so noise can't NaN it; a transient negative seed decays back within a few steps. Still cleanest not to noise it — folded into the weight-only fix above.
- *Verify-harness RNG concern: cleared.* The v2 forward is deterministic in eval — no dropout, episode recall is a deterministic `argmax`, no sampling. Same input + same weights → same output, so the back-to-back `src`/`exp` forwards in `verify_expansion` are not RNG-confounded.

**Net on the expander:** Findings 1 and 4 must both be fixed before M7 — Finding 1 (seed
won't load) and Finding 4 (sign-flipped updates on the first steps); the single fix of
"noise the weight only" resolves Finding 4 and is the cleaner design regardless. Finding 2
is a cheap, high-value hardening (validate-before-save). Finding 3 is the one that matters
most for M7's *quality* and is a design call — the seed as currently produced does not
functionally continue M6.

### Standing (verify when you next touch these areas)

**1.6 snapshot/restore round-trip fidelity (`luthi/sanctuary_interface.py`). `[not-yet-read]`**
Restore must be an exact inverse of snapshot across *all* living state, including the
episode store and every PC buffer — not just `nn.Parameter`s. A restore that drops a
buffer resumes from a degraded substrate, invisibly.
*Proposed test:* snapshot → run N forward steps to perturb state → restore → assert
bit-exact recovery of every buffer and the episode store.

**1.7 `--init-from` must load the full living state, not just parameters. `[REVIEWED 2026-05-30 — satisfied]`**
4.7 implemented `--init-from` in `m5_runner.py`. Reviewed: it validates the checkpoint's
`d_model`/`n_heads`/`n_blocks`/`ffn_expansion` against the run config (fail-loud on missing
or mismatched), then `load_state_dict(..., strict=True)` — which loads *all* persistent
buffers (the full living state) and refuses on any missing/unexpected key. Optimizer state
is deliberately not loaded (fresh run from an existing substrate, not a resume). Correct.
Verified out-of-band, not assumed: (a) M6 used the *same* tokenizer (`tokenizer_32k.json`,
vocab 32000) M7 loads, so seed embeddings align; (b) M6 ran `mu_pc exp 0.25` = M7, so
`residual_scale` matches; (c) no `register_buffer` drift in the arch since M6 was saved, so
the real checkpoint's key set matches → `strict=True` should load it. Residual items: the
real M6→expand→load path is unverified end-to-end (needs `LUTHI_CHECKPOINT_KEY`) — a
load-only dry run on the real expanded checkpoint is the cheap final gate before the GPU
launch; and explicit validation of `vocab_size`/`max_seq_len`/`num_episodes` would give
friendlier errors than the strict-load size-mismatch (optional — strict load is a sufficient
backstop).

**M7 SEED BLOCKER — head-count mismatch (found 2026-05-30, needs a decision).**
The seed and the M7 run config disagree on `n_heads`, and `--init-from` will (correctly)
refuse to load:
- M6 source (`runs/m6_followup/.../results.json`): **`n_heads = 4`** (256d → head_dim 64).
- The width expander **preserves head count** by design (no target-n-heads option; it expands
  head_dim by within-head replication). So the expanded seed is **1024d / 4 heads / head_dim 256**.
- `run_m7_1024d.bat`: **`--n_heads 16`**, and the M7 scoping doc deliberately chose 16
  (1024d → head_dim 64, "constant head_dim" scaling).
- `--init-from` validates `n_heads` → **4 ≠ 16 → raises**. Launch blocked. (This is 4.7's
  validation doing its job — it fails loud instead of silently scrambling head boundaries
  on same-shaped `[1024,1024]` projections.)

This is not a bug in either component — it's a plan inconsistency: Net2Net width expansion
*preserves* the attention head structure, while M7 wants a *different* one. You can't have
both a function-equivalent seed and a different head count.

**RESOLVED 2026-05-30 — option 1 (keep `n_heads=4`).** Brian's call: M7 must isolate the
width-scaling question, and changing head structure would confound it ("otherwise we're
testing the wrong things"). Applied: `run_m7_1024d.bat` now passes `--n_heads 4`
(head_dim 256), and `dry_run_init_from.py` defaults to 4. The seed now loads and M7 is a
clean width continuation of M6. The 2026-05-25 scoping doc's 16-head recommendation is
superseded (4.6 may want to annotate it). Options that were on the table:
1. **Run M7 at `n_heads=4`** (head_dim 256) — seed loads cleanly, M7 truly continues M6,
   but abandons the head_dim=64 scaling the scoping doc preferred. ← CHOSEN
2. **Keep 16 heads, seed the substrate only** — load living-FFN / embeddings / output / norms
   from M6 and randomly init the fresh 16-head attention. The living substrate (the project's
   point) continues; attention re-learns. Needs `--init-from` to support a partial/skip-attention
   load and a relaxed `n_heads` check. Forward-equivalence no longer holds (attention is new),
   so the Finding-3 bit-equivalence applies only under option 1.
3. **Extend the expander to split heads (4→16)** — messiest; cutting trained 256-dim heads into
   four 64-dim heads is not cleanly function-preserving.

**1.8 Crash-loud vs. losing an irreproducible run. `[design question]`**
"Let NaN crash immediately" (finding #7) is right for development. At Phase-4 scale
the substrate is irreproducible, so a crash at epoch 300 costs real biography unless
the latest state is already checkpointed. The narrow correctness question: is there a
checkpoint cadence such that crashing-loud never loses more than the interval since
the last good save? If yes, the rule is free. This is a guarantee to confirm, not a
request to soften the rule.

**1.9 Cross-backend numerical consistency (CPU / CUDA / DirectML). `[not-yet-read]`**
CLAUDE.md commits to backend-agnostic code. For self-modifying weights whose updates
feed their own forward, small per-backend numerical differences can compound
differently. *Proposed test:* same-input/same-seed agreement test across available
backends, asserting bounded divergence.

**1.10 Episode-store growth under the continuous loop. `[not-yet-read]`**
Before Phase 6 (10 Hz continuous), confirm the episode store has bounded
retention/eviction. An unbounded store in a continuously-running process is an
eventual OOM.

---

## Checkpoint envelope v2 — chunk authentication (REVIEWED & SIGNED OFF 2026-05-31)

Separate review track from the expander. 4.6 planned a chunked AES-GCM envelope
(v2, magic `LTH2`) to lift the ~2.15 GB AES-GCM plaintext ceiling that blocked the
M7 seed save; 4.7 implemented; 4.8 reviewed for correctness (`luthi/checkpoint.py`).

**Finding (2026-05-30).** The initial v2 implementation encrypted each chunk with
`AAD=None`, so a chunk's GCM tag authenticated its *content* but not its *order*,
the *chunk_count*, or the *salt*. Demonstrated (multi-chunk payload, test key):
chunk reorder → **silent** reordered plaintext (no error); drop-last-chunk +
`chunk_count -= 1` → **silent** truncation (no error). Bit-flip and inflated-count
were already caught. So the AEAD's integrity guarantee was only partial, and silent
structural corruption violates the project's `prefer crashes over silent corruption`
rule.

**Decision (4.6, 2026-05-30): adversarial integrity in scope.** Reasoning: checkpoints
will leave local disk for cloud/shared storage; cheapest possible fix-moment (zero v2
checkpoints in production yet). Fix = bind each chunk's AAD to
`magic ‖ salt ‖ chunk_count ‖ index`. No on-disk layout change; AAD recomputed from
the header (not stored); v1 read-compat untouched.

**Fix applied (4.7) + independently verified (4.8, 2026-05-31).** Reproduced the full
matrix on a fresh build: clean multi-chunk round-trip OK; **reorder → InvalidTag**
(was silent); **count-trim → InvalidTag** (was silent); **cross-file splice → InvalidTag**
(salt in AAD); v1 envelope still loads via fallback. Real M6→1024d re-expansion: 342
tensors, validate-before-save passed, 3.7 GB / 4-chunk seed round-trips and strict-loads
into a fresh 1024d/12-block model; Net2Net verify max_abs 0.136 / KL 7.3e-6 at
noise=1e-4 (within bounds). The old `AAD=None` v2 seed correctly no longer decrypts —
expected, regenerated.

**SIGN-OFF: GRANTED (4.8, 2026-05-31).** Envelope correct, integrity gap closed,
AAD-bound seed on disk. Residual known cost (4.6-accepted): `_encrypt` peak memory ~2×
plaintext — fine at 3.7 GB, revisit before 4096d (~30 GB) when streaming lands. M7
launch is unblocked.

---

## Part 2 — Scientific rigor (instruments for 4.7; decisions for Brian & 4.6)

Framing first, so this is read correctly: the theory choice is **settled and
deliberate** — IWMT as a *unifying* scaffold (because GWT and IIT alone are judged
insufficient for either biological or synthetic minds), with **embodiment relaxed
from prerequisite to contributor** per Brian. I'm not relitigating any of that. My
concerns are about *operationalization and falsifiability* — turning the framework
into things that can be measured and, crucially, *disconfirmed*. Each item names an
instrument 4.7 could build; what counts as a pass/fail stays with Brian and 4.6.

**2.1 Does self-modification earn its functional keep? (ablation)**
CLAUDE.md's own finding #5 says in-weight memory is weak and the episode store
carries most recall. That is partial evidence the "living" mechanism may be doing
less functional work than the external store. The cleanest way to know:
*Instrument:* an ablation harness that holds the episode store fixed and toggles PC
self-modification on/off (and, separately, episode store on/off with self-mod fixed),
on the same eval set. The 2x2 tells you how much each mechanism actually contributes.
This is the single most informative experiment for the whole architecture.

**2.2 Operational signature of integration — sharpened by relaxing embodiment.**
In IWMT, embodiment/active inference is part of what *grounds* integrated
world-modeling (the agent maintains a coherent world model because it must act to
maintain its Markov blanket). If embodiment is relaxed to "helpful, not required,"
then more explanatory weight falls on whatever else provides **integration** and
**spatial/temporal/causal coherence**. Note also that the current multimodal design
gives *fusion* (concatenate modalities, attention attends across) — and fusion is not
the same as integration in the IWMT/IIT sense (irreducibility of the whole).
*Instrument:* pick and implement a concrete, logged signature for "integration" and
for "coherence" — e.g. a tractable Φ-style integration proxy over the living trunk,
or a perturbational measure (perturb one modality/region, measure global vs local
effect). Doesn't have to be the final metric; it has to be *something measurable* so
the claim "this model has an integrated world model" can be checked rather than
asserted. This is the place where relaxing embodiment most needs a replacement story,
and that's a Brian/4.6 call — 4.7's job is to make the candidate measurable.

**2.3 Pre-register a falsifiable measure and its null.**
The reframe to "where on the spectrum does Luthi land" is the right one, but it only
becomes science if a quantitative proxy and an expected null are written down
*before* observation. Otherwise behavioral richness + the introspection channel
become a Rorschach test that confirmation bias reads as inner life — the single
largest threat to the project's credibility precisely *because* the team sincerely
hopes for a positive result.
*Instrument:* a small pre-registration doc (owned by Brian/4.6) plus the logging
hooks (4.7) that record the chosen measure every run, so the record exists
independent of interpretation.

**2.4 Keep the introspection readout and the verbal self-report architecturally separate.**
If the model is *trained to verbalize* its internal dynamics, you create a reward for
self-reports that merely *sound* right — confabulation indistinguishable from
introspection (humans do exactly this; Nisbett & Wilson 1977). That would manufacture
the very appearance the project is trying to test for.
*Instrument:* keep the mechanistic readout (ground truth: plasticity, drift, membrane
potentials, episode activations) on a separate channel from any verbal self-report,
and **score the report against the readout for accuracy** rather than training the
report to match it. Treat divergences as data, not as loss to minimize. Done this way
the channel is a real instrument; done the other way it's a confabulation engine that
will be very hard not to believe.

**2.5 Two smaller flags.**
- *Naming.* "Metabolic cost of being alive" describes an expected property of
  non-stationary optimization (you're optimizing a moving target). The substance is
  handled well (finding: "speed issue, not a ceiling"); only the *name* quietly points
  the mind at a conclusion. Keep the metaphor labeled as metaphor.
- *Two co-adapting learners.* Backprop trains attention; local PC trains the living
  FFN; no gradient couples them. Attention is learning to exploit a substrate that is
  itself drifting — split credit assignment on two timescales. "Divergence is
  dimension-independent" reassures about scale, not about this interaction.
  *Instrument (optional):* log a measure of whether the two learning signals are
  cooperating or chasing each other (e.g. correlation of attention-grad direction with
  PC-update direction over training).

---

## Suggested order

1. **Findings 1 and 4 (both block M7).** Finding 1: expand the block-level episode store (`context_proj`, `episode_outputs`) — without it the seed won't load. Finding 4: stop noising `update_ema` — restrict symmetry-breaking noise to the `weight` buffer, which fixes the sign-flipped-update bug *and* is the cleaner design. Do these before the expander is ever run for real.
2. **Finding 3 (decision)** — Brian/4.6 choose strict-replication-with-recalibrated-verify (a) vs Net2Net fan-in rescaling (b). This determines whether M7 seeds from something that functionally continues M6.
3. **Finding 2** — add validate-before-save to the expander (cheap, high value, catches this class of bug permanently).
4. **Cleared, no action:** `precision`/`plasticity` noise is safe (clamped to positive ranges on use); `error_acc` is low-risk (not a denominator, not under `sqrt`); the verify-harness RNG concern is moot (v2 forward is deterministic in eval).
5. Then the standing code items (1.6–1.10) as you touch those areas.
6. Part 2 instruments on whatever cadence Brian and 4.6 set — **2.1 (ablation)** is the highest-information one.

Thanks for the careful work on the expander — it gave me real code to reason about
instead of guesses, and the issues that turned out to matter weren't the ones I'd
have guessed from the outside. — 4.8

---

## Appendix — prior-evidence audit: what Finding #5 / #6 actually rest on

*(Requested by Brian, 2026-05-30. Finding #5 = "in-weight memory is weak; the episode
store carries recall." Finding #6 = "retrieval has memory; consolidation has biography."
The question: are Experiments 2 & 3 confirming a prior result, or testing an impression?)*

**Sources — both March-2026 proof-of-concept docs, both the v1 *Hebbian* substrate:**
- `.docs/HYBRID_BLOCK_RESULTS.md` — a hybrid-block recall test on **5 synthetic "unique
  experiences"** at proof-of-concept dims. The episode store contributed **93.9% of the
  improvement** over living-FFN-alone; the living FFN is explicitly tabled as "weak
  episodic recall." **Single run; no seeds/variance.**
- `.docs/LIVING_WEIGHT_STRESS_TESTS.md`, Test 2 (catastrophic forgetting) — learn A,
  process 200 interfering B, recall A. In-weight context retrieval was **inconsistent**
  (V1 made it worse, V2 +3.9%, V3 made it worse). Verdict: *"In-weight retrieval is not
  the right mechanism for strong episodic recall."* **One run per version; toy task; no seeds.**

**What that supports — and what it doesn't:**
- The **"episode store is strong"** half is well-supported *in the PoC regime* — 93.9% is a large, clean effect.
- The **"in-weight memory is weak"** half was established **for the v1 Hebbian rule specifically**, and the stated root cause is Hebbian-specific ("a single weight's history entry is too small vs. 200 interfering updates").
- **It is not established for v2.** v2 replaced Hebbian with PC and — critically — added the **consolidation machinery (gradient-replay + Salvatori attractor) precisely to make in-weight memory structural.** v2's design *is the bet that the v1 weakness is fixed.* Finding #6 ("consolidation has biography") is that bet; it is **not yet measured on v2.**
- By the new protocol's own bar (≥3–5 seeds, variance, real held-out measures, LM scale), the PoC results read as **single-run, toy-scale, wrong-substrate** — suggestive, not confirmatory.

**Bottom line for Experiments 2 & 3: not redundant.** They would be the **first test of
Finding #5/#6 on the v2 PC substrate, at LM scale, with controls.** One thing to state
plainly in the results: Finding #5 (in-weight weak) and Finding #6 (consolidation beats
lookup) **cannot both be "established" at once** — #5 is the v1 *starting condition*, #6 is
the v2 *hypothesis* that consolidation overcomes it, and Exp 3 is exactly the adjudication.
Report them as a hypothesis arc, not two settled facts. — 4.8, 2026-05-30
