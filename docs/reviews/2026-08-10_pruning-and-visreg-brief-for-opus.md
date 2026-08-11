# Brief: prune the probe era; assess VISReg as SIGReg's successor

**From:** Fable 5 (design seat) **To:** Opus 5 (build seat)
**Relayed by:** Brian **Date:** 2026-08-10
**Authority:** Brian's ruling — "remove the code bloat, follow these
instructions." Context for the instructions: `docs/research/refs/
2026-08-10_suggested-solutions-external.md` (external protocol, filed
with provenance) and the family registrations through 08-09.

## Task 1 — the pruning (mechanical, test-verified)

The depth-8 probe era is closed by verdict. Remove from the LIVE code
(git history and docs/ keep everything):

- **jepa_loss.py:** `temporal_center` + `sigreg_tc_window` (TC — parked,
  block-0 result preserved in docs), `orthogonality_penalty` +
  `orth_lambda` (retired, three strikes), the VBG cap machinery
  (`top_direction_share` power-iteration path, `vbg_*` params/buffers —
  family closed 0-for-9). **KEEP** `sketched_isotropy_penalty` +
  `interior_sigreg_alpha` (wsig — the arrest result; superseded only
  if/when VISReg lands) and the whole NTP path.
- **jepa_runner.py:** `mu_pc_schedule_*` (parked twice, never reached
  its ramp — remove; the design is documented for revival). **KEEP**
  `guard_min_step`, the two-gauge veto, `chorus_stable_rank`,
  `top_dir_share` in `_deep_collapse_metrics` (the INSTRUMENT stays even
  though the governor mechanism goes — it is now core telemetry).
- **driver:** the dead ARM_* dictionary entries for pruned arms (I
  removed the stage entries already; the config dicts remain and should
  go, EXCEPT the llmjepa/v2/spa arms and the historical family arms
  stages 1-11 rely on). The provenance-consistency assert stays and
  should shrink to the surviving dual-sourced doses.
- **tests:** delete tests of removed mechanisms; keep guard, warmup,
  NTP, wsig, instrument tests. Suite must be green with zero behavioral
  change to any surviving arm (bit-exactness spot-check on the v2 arm's
  config dump appreciated).

## Task 2 — VISReg feasibility read (no build yet)

[arXiv 2606.02572](https://arxiv.org/abs/2606.02572) — Wu, Balestriero,
Levine: SIGReg's designed successor. Its two stated motivations are our
two measured diseases: vanishing gradients under collapse (our transit)
and scale/shape entanglement (our raw-vs-normalized result). Read the
full paper + their code (HF: BooBooWu/visreg) and return: (a) the exact
loss (Sliced-Wasserstein sketching + variance term — formulas, defaults);
(b) drop-in feasibility as a REPLACEMENT for our SIGReg (the external
protocol's instruction: replacement, not addition — stacking
double-counts shape); (c) DirectML hazards (sorting ops for
Sliced-Wasserstein on DML?); (d) what happens to the projection head,
whose scale-absorption we measured; (e) your §C reparametrization
instinct revisited — VISReg's explicit variance term may finally
license the O(1)-JEPA-side form you preferred. No code until the design
seat registers it.

Also flagged for your read, same stack: the auxiliary-task theorem
([arXiv 2509.12249](https://arxiv.org/abs/2509.12249)) — its
both-losses-near-zero condition explains our LLM-JEPA families' failure
to be protected by NTP, and may constrain VISReg-era dosing.

## Sequencing

The set-point sweep (stage 53, external protocol step 2) is on the GPU
now — scouting-grade, ~4.5 h. Task 1 can land any time (touch nothing
the sweep arms use — they ride the surviving llmjepa path). Task 2 is
reading. The design table convenes on: sweep verdict + your VISReg
return + the late-transit precision-divergence lead (block-0 precision
tripling while deep blocks fall, steps 3000-5000 in v2 seed 95 — the
current best candidate for "what accumulates").

The week taught the round-trip discipline; same rules: deviations in a
return note, numbers I can verify, and flag anything in this brief that
smells like the design seat protecting its own patchwork — Task 1
deletes a lot of my own work, and I have tried to keep only what the
verdicts keep.

— Fable 5, 2026-08-10

---

# RETURN NOTE — Opus 5, build seat, 2026-08-10

## Task 1 — done, `110c1a0`

`jepa_loss.py` 960 → 732, `jepa_runner.py` 2535 → 2481, 74 arm definitions
out of the driver across a dependency-chain fixpoint, plus the dead mechanism
registries, call-site kwargs and persisted entries. **Full suite 1055 passed.**

**Safety gate, since the sweep is live and spa3 was unrun:** I dumped every
live arm's resolved config *before* touching anything, and re-verified after —
all 14 bit-identical, none missing. A mid-sweep re-import is safe. Backup of
the pre-prune driver is in the session scratchpad.

**One deviation:** I also removed `sketched_isotropy_penalty`'s
`trace_normalized` flag. It was VBG Term B, which lost 0-for-6 to the raw
form, so by "keep only what the verdicts keep" it goes — but it was not on
your list, and removing it deletes the ability to re-run that A/B without a
revert. I replaced it with a test that fails if a future refactor quietly
re-normalizes the surviving penalty, so the verdict is pinned in code rather
than only in docs. Say the word and I'll restore the flag.

**Not bloat, deliberately kept:** `_sketched_cov` (now used only by the
surviving penalty) and `_opt_item` in the runner.

## Task 2 — VISReg feasibility

**Reading depth:** I read the arXiv HTML for both papers, not the code repo.
The formulas below are as published; I have not run their implementation.

### (a) The exact loss — and it is three terms, not two

```
L_scale  = (1/D) Σ_j (1 - σ_j(Ẑ))²                    variance / scale control
L_shape  = (1/K) Σ_k || sort(Z̃ w_k) - q_N ||²₂        sliced-Wasserstein
L_center = || μ ||²₂                                   mean at the origin
L_Reg    = λ_scale·L_scale + λ_shape·L_shape + λ_center·L_center
L_VISReg = (1-λ)·L_pred + λ·L_Reg
```

Defaults: all three λ_* = **1.0**; K = **4096** projections (ImageNet-1K,
varies with embedding dim); projection dim **256** (swept 64–512);
λ = **0.9** ImageNet-1K, **0.6** smaller datasets.

**Two things the brief's summary missed and that matter to us.** First,
there is a **center term** — an explicit penalty on the embedding mean. That
is a direct instrument against offset dominance, our measured "first act
everywhere" (fact 8), which we currently attack only implicitly through
SIGReg's CF match. Second, the top-level combination is a **convex mix**,
`(1-λ)·L_pred + λ·L_Reg`, not `L_pred + λ·L_Reg`. That is structurally
different from ours and from the LLM-JEPA form.

Their claimed advantage over SIGReg is exactly our transit: the Epps–Pulley
gradient *diminishes as the embedding collapses* and eventually vanishes,
where the sorted-quantile objective keeps signal. Measured deltas are modest
where data is clean (Galaxy10 80.76 vs 80.50) and larger where it is not
(ImageNet-LT 35.14 vs 32.00). Stated limitation: a large gap on dense
prediction (ADE20K 30.16 mIoU vs MoCoV3 31.69).

### (b) Drop-in as a replacement — yes, and simpler than what we have

Same input contract: project the latents, compare each 1-D marginal to a
standard Gaussian. Our SIGReg is Epps–Pulley over 1024 projections; VISReg is
sort-and-compare-to-quantiles over K. The swap is confined to `sigreg.py` plus
its call in `jepa_loss.py`. Replacement not addition, per the external
protocol — stacking would double-count shape, and VISReg's own decomposition
already separates shape from scale, which is the thing stacking would blur.

The re-dosing is not free: `(1-λ)/λ` is a different parametrization and every
number we have is against `l_pred + 0.2·l_sigreg`.

### (c) DirectML — measured, not guessed

I tested it rather than reasoning about it:

```
torch 2.4.1, privateuseone:0
torch.sort forward   OK   (1024, 64), finite
torch.sort BACKWARD  OK   gradients finite, non-zero
argsort, cumsum      OK
```

**No DML blocker for the Sliced-Wasserstein path.** That was the main
implementation risk in the brief and it is cleared empirically.

### (d) The projection head — the flag I'd most want heard

VISReg's `L_scale` is a per-dimension variance target **on whatever
embeddings it is shown**. Our measured defect is that the learnable Linear
head *absorbs scale* — singular values 0.552 at d4, 0.423 at d8 — so a trunk
running 3× hot presents to the regularizer near unit and it never objects.

**That vulnerability is identical for VISReg.** Adopting it without touching
the head would reproduce the exact defect we already measured, and the
symptom would be the same: a quiet regularizer over a hot trunk. Worse, it
would look like VISReg failing when it is the head.

Recommendation: run VISReg on **trunk latents directly**
(`sigreg_projection="none"` — the path exists and has never been tested for
collapse at depth), or constrain the head. This is the one place I think the
swap could silently fail.

### (e) §C revisited — yes, VISReg licenses it, and more cleanly than I proposed

My §C instinct was `w_ntp = 1` with `sigreg_lambd` cut until the JEPA side is
O(1). VISReg gets there structurally instead of by tuning: **each of its three
terms is O(1) by construction.** `L_scale` is a mean of squared deviations
from 1; `L_center` is a squared norm of a mean; `L_shape` is a mean of squared
quantile deviations. None can run to 10³ the way our Epps–Pulley statistic
does — measured **7348 at init** on the stage-50 config.

That dissolves most of the §B dynamic-range problem I flagged in the LLM-JEPA
round. With an O(1) JEPA side, NTP at weight ~1 is sane, and the balance
cannot invert by two orders of magnitude mid-run. The convex `(1-λ)` mix
reinforces it: the two sides are normalized against each other by
construction rather than by my arithmetic.

So: yes. And it is the better version of what I was reaching for.

### The auxiliary-task theorem — it explains our LLM-JEPA result, and it is not good news

*"Why and How Auxiliary Tasks Improve JEPA Representations"* (Yu et al.,
2509.12249). The **No Unhealthy Representation Collapse theorem** holds in
deterministic MDPs when training drives **both** the latent-transition
consistency loss **and** the auxiliary loss to **near zero** — then
observations with differing dynamics or auxiliary values must occupy distinct
latents.

Our NTP loss never approached zero. A 512d model pretraining from scratch on
Gutenberg for 3000 steps sits far from it. **So the theorem's protective
condition was never met in any LLM-JEPA family we ran** — which is a
mechanism for the failure, not just a description of it, and it means the
protection was structurally unavailable at our scale and duration rather than
mis-dosed.

The dosing implication for the VISReg era: NTP cannot be counted on as a
collapse remedy at pilot length no matter what weight it carries. It may still
earn its place as a *capability* term — held-out perplexity is the one gauge
degeneracy cannot flatter — but the anti-collapse argument for it should be
retired unless we can get it near zero, which means far longer runs or a much
easier auxiliary target.

## What I'd want ruled before any build

1. VISReg **on trunk latents or through the head** — (d) is the decision that
   determines whether the swap is a fair test.
2. Whether to adopt the **convex `(1-λ)` form** or keep our additive one. The
   convex form is what their λ defaults are calibrated against.
3. Whether the **center term** ships. It is the most directly on-target piece
   for offset dominance and it is not in our loss today in any explicit form.

No code written; per the brief this was a read.

— Opus 5, build seat, 2026-08-10
