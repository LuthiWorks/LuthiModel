# A7 under attack — "benign carrier" vs "collapsed block bypassed by residual"

**Auditor:** Muse Spark (Meta), fresh instance, first session with the repo.
**Scope:** the A7 verdict in `docs/audits/2026-08-13_luthimodel-audit.md`
("Blocks 0–1 were never collapsed; `effective_rank` orders these states
backwards"), taken up at the audit's own invitation ("the three things
most worth attacking: 1. A7's conclusion").
**Method:** re-derive the verdict from the evidence chain and the
measurement code paths. Read-only; no checkpoints available on this
machine, so the discriminating experiment is proposed, not run.

## The verdict as stated

Seed 97's blocks 0–1 read `effective_rank` ~2.1 (looks collapsed) but are
declared healthy — a "benign carrier," a soloist direction riding on a
rich representation — on two legs: (a) whole-model probe lift is flat at
5.6–6.1x including the rank-2.1 blocks, and (b) chorus rank of those
blocks (412/414, audit; 154.0 in the `jepa_runner.py:748` docstring —
see O2) far exceeds the wrecked reference (seed 95 b7: eff 4.3,
chorus 61, probe 4.12x).

## The attack: the evidence cannot see the block, only the stream

Every per-block gauge in the chain is computed on `block_latents`,
collected **after** the block: `h = block(h); block_latents.append(h.detach())`
(`multimodal_model_pc.py:386-391`). And the block is residual —
`x = x + residual_scale * attn_out`, `x = x + residual_scale * ffn_out`
(`hybrid_block_pc.py:114-117`). So the measured object is
`h_out = h_in + s·ffn_out`, and the probe reads only the *final* trunk
latents (`eval_heldout.py:147`, `result["per_modality"]["text"]`).

This admits a second hypothesis fitting **every number A7 cites**:

- **H1 (audit's):** blocks 0–1 compute richly; the soloist is an
  incidental carrier direction.
- **H2 (bypass):** the living FFN in blocks 0–1 is collapsed — its
  output is a near-batch-constant giant vector, the exact v5 pathology
  ("92–95% a single batch-constant direction") in miniature — while the
  residual stream carries the rich representation *past* it. Chorus then
  measures the residual, not the block. The whole-model probe is fine
  because six other blocks do the work.

Under H2, "blocks 0–1 were never collapsed" is false at the level it is
stated: the *stream* is healthy, the *block* is dead, and no instrument
in the chain can tell the difference. That is a block-level
healthy-but-inert failure — the class this repo hunts — hiding behind
the residual, invisible to every gauge that reads post-residual outputs.

## Why this matters beyond the verdict

1. **The veto reads per-block gauges.** The divergence rank veto and the
   kill criteria consume `effective_rank` / chorus per block. If those
   gauges cannot distinguish "block healthy" from "block bypassed," the
   project can lose living computation block by block with every
   instrument green. A7's framing ("two states, cleanly separated")
   encourages trusting the separation; H2 says the separation is
   between *stream* states, and block death is still uninstrumented.
2. **"Two kinds" may be one continuum.** v5's fatal collapse and seed
   97's benign soloist share the signature (batch-constant dominant
   direction); they differ in magnitude (soloist share ~0.95 vs ~0.20).
   Chorus rank then isn't detecting a *kind* difference — it's a better
   *magnitude* gauge than effective rank, which genuinely inverts the
   ordering (4.3 "healthier" than 2.1 while worse). The gauge improvement
   (A8) survives this attack intact; the mechanistic story ("carrier, not
   collapse") does not need it.
3. **n=1 per class.** Benign = seed 97; fatal = seed 95; prospective =
   one 512d run. The audit already flags the thin base (B3's gate ships
   disarmed "for want of a distribution"). The H1/H2 ambiguity means the
   base is thinner than it looks: we have one stream-healthy example and
   one stream-wrecked example, and zero block-healthy-vs-block-bypassed
   pairs.

## The discriminating experiment (proposed)

`block_latents` already gives `h_in` (previous block's output;
embeddings for block 0) and `h_out`. The block's *contribution* is
`c = h_out − h_in = s·ffn_out` — no new collection needed, computable
from any checkpoint with block latents saved.

- **Cheap version:** batch-variance of `c` normalized by batch-variance
  of `h_in`, per block. H1: `c` varies with input (the block computes).
  H2: `c` is near-batch-constant (variance ratio ≈ 0) — the soloist is
  all the block contributes.
- **Decisive version:** run the standardized linear probe
  (`fit_next_token_probe(..., standardize=True)`) on `c` per block.
  H1: probes well above chance. H2: chance — the block adds no
  information the residual didn't already carry.

If H2 wins on blocks 0–1, the right gauge is not chorus-on-output but a
**contribution gauge** (chorus, or probe lift, on `c`), and the veto
should read it.

## Concessions (steelman of A7)

- At *model* level, "benign" holds under either hypothesis: capability
  is fine, and the audit's practical conclusion — don't kill the run on
  eff-rank alone — is correct.
- The 512d prospective run (eff ↓22% while chorus ↑25% in the same
  block, best probe of three) is consistent with H1 *and* H2 (richer
  embeddings flowing through the residual raise block-0 chorus under
  H2 as well) — it does not discriminate, but it doesn't contradict
  either.
- A8's gauge itself (chorus rank as a better-ordered instrument than
  effective rank) is untouched by this attack.

## Observations

- **O1:** the audit's "chorus2 rank (412/414)" vs the code docstring's
  "chorus 154.0" for seed 97 b0 disagree with no reconciliation note
  (different checkpoints? top-2-dropped vs top-1-dropped?). Worth one
  line stating which.
- **O2:** `_eval_guard` + `freeze_plasticity` in the probe collector
  means the probe can never see living-state dynamics — correct for a
  capability probe, but it also means the probe is blind by construction
  to exactly the substrate behavior at issue.

## Bottom line

A7's *instrumental* conclusion (effective rank inverts; chorus orders
correctly; don't kill on eff-rank) stands. Its *mechanistic* conclusion
(blocks 0–1 were never collapsed) overreaches: the measurements are
post-residual, the probe is whole-model, and a collapsed-but-bypassed
block fits all the same numbers. The missing instrument is a per-block
contribution gauge, and the experiment to build it is cheap.
