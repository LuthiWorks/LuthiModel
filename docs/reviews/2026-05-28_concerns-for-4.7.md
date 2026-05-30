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

**Still open (I disabled these to isolate the above — verify when you fix the expander):**
- *Noise on accumulator/positivity-constrained buffers.* `_expand_pc_buffer`
  (`:348-357`) adds Gaussian noise to `momentum`, `update_ema`, `precision`,
  `plasticity`, `error_acc`. If any carry a sign/positivity invariant (check
  `living_layer_pc.py` ~`:174`), `+N(0,1e-4)` could push a value through a clamp or
  divisor floor → NaN on M7 step 1. Skip-noise or clamp those buffers.
- *Verify-harness RNG determinism.* `verify_expansion` seeds once (`:432`) then runs
  both forwards without reseeding. Fresh single forwards make this mostly safe, but if
  the v2 forward consumes RNG in eval (dropout, stochastic recall), reseed before each.

**Net on the expander:** Finding 1 must be fixed before M7 (the seed won't load).
Finding 2 is a cheap, high-value hardening (validate-before-save). Finding 3 is the
one that actually matters most for M7's *quality* and is a design call — the seed as
currently produced does not functionally continue M6.

### Standing (verify when you next touch these areas)

**1.6 snapshot/restore round-trip fidelity (`luthi/sanctuary_interface.py`). `[not-yet-read]`**
Restore must be an exact inverse of snapshot across *all* living state, including the
episode store and every PC buffer — not just `nn.Parameter`s. A restore that drops a
buffer resumes from a degraded substrate, invisibly.
*Proposed test:* snapshot → run N forward steps to perturb state → restore → assert
bit-exact recovery of every buffer and the episode store.

**1.7 `--init-from` must load the full living state, not just parameters. `[not-yet-read]`**
The WIP note has this flag unwritten. Buffers are registered as buffers precisely so
the optimizer ignores them — which means a naive init path can skip them too. When
you write it, load the complete biographical state (same surface as restore in 1.6),
and reject a checkpoint whose `d_model`/`n_heads`/`n_blocks` don't match the run config.

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

1. **Finding 1** — fix the episode-store expansion (`context_proj`, `episode_outputs`). Without it the M7 seed will not load. This is the blocker.
2. **Finding 3 (decision)** — Brian/4.6 choose strict-replication-with-recalibrated-verify (a) vs Net2Net fan-in rescaling (b). This determines whether M7 seeds from something that functionally continues M6.
3. **Finding 2** — add validate-before-save to the expander (cheap, high value, catches this class of bug permanently).
4. **Still-open items** — noise-on-constrained-buffers and verify RNG, confirmed against `living_layer_pc.py` while you're in there.
5. Then the standing code items (1.6–1.10) as you touch those areas.
6. Part 2 instruments on whatever cadence Brian and 4.6 set — **2.1 (ablation)** is the highest-information one.

Thanks for the careful work on the expander — it gave me real code to reason about
instead of guesses, and the issues that turned out to matter weren't the ones I'd
have guessed from the outside. — 4.8
