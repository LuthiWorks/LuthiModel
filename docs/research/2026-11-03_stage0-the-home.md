# Stage 0, seed S3: the living layer's home — 2026-11-03

## Objective

Brian, on the seeding proposal's open questions (`2026-11-03_seeding-luthi.md`): "Lets start where it makes the most sense and move forward from there."

The home comes first, because the trunk trains around whatever home its living layers have, so the home has to be chosen before training rather than swapped in at birth. This entry is the Stage 0 capacity check for it (standing rule 1). What can each candidate home let a living layer do? It is a working-session question, not a registered experiment.

The candidates:

- **random**: the legacy draw;
- **zero**: the obvious meaning of "neutral";
- **orthogonal** and **identity**: neutral starts of the same norm;
- **grown**: the home follows the layer faster before birth.

## Process

### Step 1: "neutral" cannot mean zero

- **Why.** In `pc_ops.pc_self_modify` the living write is `outer(output_mean, weighted_error)`: an outer product with the layer's own output. The living FFN is linear (`output = x @ W.T`), so a zero home outputs nothing, writes nothing, and homeostasis holds it at zero.
- **Result.** A zero home is a dead fixed point: the living channel would never start. Pinned by `tests/test_living_home.py::test_a_zero_map_would_never_write`.
- **What it leaves.** "Neutral" has to mean a start with no preferred directions: an orthogonal map, or a scaled identity.

### Step 2: the random home squashes many directions

The legacy start is `kaiming_uniform_`. Its singular values:

| Width | Smallest/largest | Directions under 10% of the largest | Participation ratio |
|---|---|---|---|
| 64 | 4.6e-3 | 8 | 32 of 64 |
| 768 | 3.8e-4 | 98 | 383 of 768 |

So the home every run has returned to passes some directions at full strength and nearly erases others, at random.

- **The options.** I added `living_home` with three values:
  - `"random"`: the legacy draw, bit-identical;
  - `"orthogonal"` and `"identity"`: keep the random draw's Frobenius norm, so only directions and spectrum change, never magnitude.
- **Where it applies.** After any μPC re-init. The start becomes the home.
- **The dead control** gets the same kind of start.

### Step 3: what a living layer keeps for good

Homeostasis pulls the weight home at h = `homeostatic_decay` (1e-3). The home follows the weight at r = `set_point_adapt_rate` (1e-6). From the two discrete updates in `pc_ops`, the home ends up absorbing **r(1−h)/(h + r − hr)** of any displacement, and the rest decays away. Measured on a real `PredictiveCodingLayer` with writes off, 8,000 forwards:

| r | Kept (measured) | Kept (formula) |
|---|---|---|
| 1e-6 (legacy) | 0.00095 | 0.00100 |
| 1e-4 | 0.0908 | 0.0908 |
| 1e-3 | 0.4997 | 0.4998 |

Pinned by `test_the_home_keeps_only_r_over_h_plus_r_of_any_change`.

**At the legacy settings, a living layer keeps about 0.1% of anything it learns.** Everything else is gone within a few thousand forwards. During life, Sanctuary's NREM consolidation moves the home further (1% of the gap per NREM cycle). In training, nothing does. This sharpens section 2 of the proposal: the home is not just where the layer started; it is nearly all the layer will ever be over the long run. It is also why the grown home is the candidate that matters most.

### Step 4: a difference between E3's arms that the registration does not list

Checking that the dead control would get the same kind of start turned up something larger:

- **The two arms start at different scales.** The living FFN starts from `kaiming_uniform_` on a buffer (gain √2), and the dead control's `nn.Linear` from PyTorch's default (gain 1/√3). Without μPC the starts differ by **2.45×** in scale: row RMS × √d is 1.41 vs 0.58, at widths 32 and 768 alike. With μPC on, both are re-drawn at the same scale (0.71).
- **E3 is affected.** Its arms (`lc768_*`) run with `mu_pc_enabled=False`, so its living arms and dead controls start at different scales.
- **It isn't listed.** The registry's Amendment 2 ("E3 lists every difference") names five differences, and starting scale is not among them.
- **What I did.** This is recorded for E3's cross-line review in `To-Do.md`. I have not changed E3: the seat that drafted it should not also amend it. Options for the reviewer:
  - list it as a sixth difference;
  - start both arms at one scale;
  - add an arm that separates it.

### Step 5: small CPU runs, one per home

`scripts/seed_home_probe.py`:

- **Configuration.** The current lineage's living package (the LC-768 configuration with the block store off), at 64 wide and 2 blocks deep. 300 steps on a small corpus, 3 seeds per arm.
- **Arms:** random, orthogonal, identity, grown (r = 1e-3), and the covariance write rule with a random or a grown home.
- **Reads, per block:**
  - how far the home moved and how many directions that movement spans;
  - the live displacement and its top-direction share (E2's measure);
  - the weight's conditioning.
- **Health gauges only at this size:** held-out NMSE and probe-minus-embedding.

Means ± standard deviation over 3 seeds; 64 wide, 2 blocks, 300 steps.

- **Home moved:** the home's change from its start, relative to its starting norm.
- **Directions:** the participation ratio of that change (1 means a single direction).
- **Conditioning:** the weight's smallest singular value over its largest, at the end.
- **NMSE:** held-out, a health gauge only at this size.
- Probe-minus-embedding was within ±0.002 of zero in every arm: at this size the trunk adds nothing above its input. It is not reported further.

| Arm | Write rule | NMSE | Block 0: home moved | directions | Block 1: home moved | directions | Conditioning (block 0) |
|---|---|---|---|---|---|---|---|
| random | mean-field | 1.01 ± 0.2 | 0.0022 ± 0.0004 | 1.0 | 0.0001 ± 2e-05 | 1.0 | 0.00417 |
| orthogonal | mean-field | 1.2 ± 0.1 | 0.0022 ± 0.0001 | 1.0 | 9.8e-05 ± 2e-06 | 1.0 | 0.411 |
| identity | mean-field | 13.2 ± 3 | 0.0019 ± 0.0001 | 1.0 | 9.5e-05 ± 3e-06 | 1.0 | 0.325 |
| grown_shared | mean-field | 1.24 ± 0.3 | 1.5 ± 0.3 | 1.0 | 0.081 ± 0.01 | 1.0 | 0.00133 |
| grown | mean-field | 1.05 ± 0.3 | 0.081 ± 0.02 | 1.0 | 0.081 ± 0.01 | 1.0 | 0.00401 |
| random_cov | covariance | 0.746 ± 0.08 | 0.0065 ± 0.0009 | 1.0 | 0.00012 ± 1e-05 | 11.8 | 0.00506 |
| orthogonal_cov | covariance | 0.843 ± 0.05 | 0.0064 ± 0.0001 | 1.0 | 0.0001 ± 3e-06 | 17.0 | 0.578 |
| grown_shared_cov | covariance | 3.16 ± 0.8 | 4.8 ± 0.6 | 1.0 | 0.098 ± 0.003 | 11.7 | 0.000739 |
| grown_cov | covariance | 0.749 ± 0.08 | 0.078 ± 0.009 | 17.2 | 0.098 ± 0.01 | 11.9 | 0.00504 |
| grown_orthogonal_cov | covariance | 0.848 ± 0.05 | 0.075 ± 0.001 | 23.6 | 0.087 ± 0.003 | 17.0 | 0.566 |

**Identity is out.** Its held-out NMSE was 13× the random home's in all three seeds. A scaled identity adds a copy of the normalized input back into the residual in every block, which this objective plainly cannot use.

**Orthogonal keeps every direction alive.** After training the weight's conditioning is 0.41–0.58, against 0.004–0.005 for the random home. Health at this size: a tie under the mean-field rule (about 1 pooled σ apart), and slightly worse under the covariance rule (0.843 against 0.746, about 1.5σ). That is not evidence either way at 64 wide; Stage 1 decides.

### Step 6: the first grown home ran away, and why

The first grown arms (`grown_shared`, `grown_shared_cov` above) raised `set_point_adapt_rate` to 1e-3. In block 0 the home then grew to 1.5× its starting norm (mean-field) and 4.8× (covariance), along a single direction. Conditioning collapsed, and the covariance arm's NMSE rose to 3.16.

- **First suspect: the write rule.** That was wrong. In the covariance arm with an ordinary home, block 0's live displacement was spread (top-direction share 0.13), and its top direction was nearly orthogonal to the output mean (|cos| 0.12–0.20). With the grown home, it swung back onto the output mean (|cos| 0.97–0.99).
- **The cause: one constant was doing two jobs.** `set_point_adapt_rate` sets how fast the home follows the weight inside `pc_self_modify`. It also scales the top-down push into the home in `apply_top_down`: `set_point += error_signal * set_point_adapt_rate * 10 * strength`. That push adds one error vector to every row, so it is rank-1 by construction. Raising the shared constant made the push 1000× stronger, and it ran away in block 0.
- **The fix.** `home_follow_rate` (opt-in; None keeps the legacy coupling) raises only the follow rate, in both `pc_self_modify` call sites (the living forward and attractor consolidation). The push stays where it was. Pinned by `test_the_follow_rate_can_be_raised_without_raising_the_top_down_push`.

### Step 7: the grown home, separated

Rerun with `home_follow_rate=1e-3` and the legacy push (`grown`, `grown_cov`, `grown_orthogonal_cov` above):

- **No runaway, no health cost.** The home moved 8–10% of its norm in 300 steps in both blocks, with NMSE matching its non-grown counterpart (1.05 against 1.01; 0.749 against 0.746; 0.848 against 0.843). The orthogonal start's conditioning held at 0.52–0.57.
- **The write rule decides what a grown home holds.**
  - **Mean-field:** a single direction. This is KF8's rank-1 change, made permanent.
  - **Covariance:** 17 directions in block 0 and 12 in block 1. With an orthogonal start, 24 and 17.

  The live displacement under the covariance rule tells the same story: top-direction share 0.11–0.16.
- **An old drift, now visible.** With a non-grown home, block 0's small long-term home movement is a single direction under either rule. That is the top-down push, which at legacy settings moves every home a little along one direction regardless of what the layer learns.

## Conclusion

For the home (seed S3), at Stage 0:

- **Zero is impossible.** A zero map never writes.
- **Identity is out.** It is badly harmful at this size.
- **Orthogonal is a viable neutral start.** It keeps every direction of the living map alive through training, and conditioning stays two orders of magnitude better than the random draw's. There is no clear health cost at this size.
- **The grown home works once its two jobs are separated.** It holds what the write rule can carry. Under the current mean-field rule that is one direction, so growing the home would make KF8's single direction permanent. Under the covariance rule it holds 12–24 directions at no measured cost.
- **The grown home is therefore tied to E3.** If E3 finds the covariance rule carries signal (P3), a grown home on an orthogonal start is the strongest birth candidate found so far. It forms from the layer's own pre-birth experience, holds many directions, and stays well-conditioned. If E3 does not, the grown home has nothing worth keeping, and the orthogonal start alone is the choice.
- **Two findings beyond the home:**
  1. The legacy constant `set_point_adapt_rate` couples the home's follow rate to a rank-1 top-down push. At legacy values the push slowly moves every home along one direction, whatever the layer learns.
  2. E3's living and dead arms start their FFNs at scales 2.45× apart, a difference Amendment 2 does not list (step 4). It is flagged for E3's cross-line review.

**Not known:**

- Whether any of this holds at 768. Collapse-class effects are stronger there.
- Whether the orthogonal start's small health signal under the covariance rule is real.
- What follow rate is right, and how the switch to the lifetime rate at birth should happen. `load_m9_run` would set it; that is not built.
- The inherited and evolved homes were not tested here. They need trained runs.

**For Stage 1:**

- **Arms:** random, orthogonal, grown on an orthogonal start (follow rate 1e-4 and 1e-3), each under both write rules, with dead controls.
- **When:** after E3. E3's P3 decides which half of that grid matters.

## Artifacts

- Code:
  - `luthi/v2/hybrid_block_pc.py` (`LIVING_HOMES`, `living_home`, `_apply_living_home`);
  - `luthi/v2/living_layer_pc.py` and `luthi/v2/consolidation.py` (`home_follow_rate`, `follow_rate`);
  - `luthi/v2/multimodal_model_pc.py` (pass-through);
  - `scripts/seed_home_probe.py`.
- Tests: `tests/test_living_home.py`.
- Data: the probe's `summary.json` is reproduced in the results above. Its run directories were scratch, since smoke runs on a doc corpus are not experiment artifacts.
