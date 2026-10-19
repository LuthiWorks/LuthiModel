# The living write is mean-field, and the block store blends a constant — 2026-10-19

## Objective

Brian asked for an audit of the repo and an honest read of the project. The
audit started from the question the August work never asked: **can the living
mechanism, as written, carry the information its claim requires?** This entry
records how that question was answered from the code and from measurements on
CPU smoke checkpoints. It records the two mechanism facts it turned up, the
covariance write rule built in response (including the instability found
along the way), and what the new instruments show at toy scale.

**Scale caveat, stated once and meant throughout.** Every training run below
is d128 × 4 blocks, 200–300 steps, on the 11 children's books in `data/`, on
a 2-core CPU. By Brian's 2026-08-15 ruling, a small run may answer "does this
mechanism run" and never "does it work". The mechanism facts in this entry
rest on the code's arithmetic. The small runs show the arithmetic holds when
the code runs, and they exercise the instruments end to end. Every "does it
work" question is registered for 768 as E1–E4.

## Process

### Step 1: read the write rule as arithmetic

`luthi/v2/pc_ops.py`, `_pc_self_modify_python`:

```
output_mean     = output.mean(dim=0)            # over ALL B*L positions
actual_input    = x_flat.mean(dim=0)
pred_error      = actual_input - output_mean @ prediction
weighted_error  = clamp(drive(pred_error) * trust, -1, 1)
delta_w         = outer(output_mean, weighted_error) * plasticity * pc_rate
```

This is an outer product of **batch means**. Two consequences follow without
any experiment:

- **Content blindness.** Any two batches with the same per-dimension means
  produce the same write, whatever their tokens.
  `tests/test_living_write_rule.py::test_CHARACTERIZATION_mean_field_write_sees_only_batch_means`
  constructs two such batches and gets bitwise-identical writes.
- **One output direction.** The write's output side is always
  `output_mean`. If that vector is stable across batches, every write lands
  in the same output direction, and the accumulated change is rank-1 there.
  The effect on any token is `Δoutput(x) = η · (w·x) · output_mean`: a fixed
  direction with per-token magnitude.

The layer-level episode context (`_compute_context`) and the stored
attractor pattern (`input_pattern`) are the same batch means.

The only open question was empirical: **how stable are the batch means?**

### Step 2: measure, near initialization (3 training steps)

Scratch probe on the real model code, VISReg objective, v5-style living
config (relative trust, consolidation, learning gain, recall gate 0.7,
top-down on), d128 × 4, batch 32 × 128.

| per block | value |
|---|---|
| cosine of batch-mean FFN input, independent B=32 batches | 0.9997–0.9999 |
| same, single sequences (B=1) | 0.990–0.998 |
| cosine of the actual weight writes from two different B=32 batches, same start state | 0.997–0.999 |
| accumulated change: top singular direction's energy share | 0.998 |
| \|cos\| of that direction with the mean FFN output | 0.91–0.96 |
| ‖per-token covariance discarded‖ / ‖mean term kept‖ | 0.06–0.22 |

At initialization the batch means are near-identical even for single
documents, so every write is effectively the same write.

### Step 3: re-measure after 200 training steps

| per block | value |
|---|---|
| batch-mean input cosine, B=32 | ≥ 0.9996 |
| episode context cosine, B=32 | ≥ 0.9995 |
| covariance discarded / mean term kept | **4.31, 2.07, 1.06, 0.62** (blocks 0–3) |
| accumulated change top-1 energy | 0.956–0.995 |
| \|cos\| with mean output | 0.977–0.995 |
| write cosine, two B=32 batches | 0.53–0.73 |
| write cosine, two single documents | 0.10–0.23 |

**A correction to my own first reading.** At initialization I would have
said "every batch writes the same thing". After training that is too strong.
Writes differ batch to batch on the input side, where the sign pattern of the
small residual moves. What every write shares is the **output side**, which
is why the accumulated change stays rank-1 along the mean output. After
training, the token-level covariance the rule throws away is up to 4.3× the
part it keeps. That is the information a content-sensitive rule would use.

**This reframes the 2026-07-29 drive-goes-quiet verdict.** That verdict
ruled out novelty exhaustion because `update_ema` fell three orders of
magnitude within the first 17% of epoch 1, while every document was still
new. But the rule never sees documents. It sees one average over 4,096
positions from 32 shuffled windows, and that average is near-stationary from
the first batch. "Quiet while everything is new" is exactly what a
batch-mean rule predicts. The drive did self-extinguish (candidate 3); the
reason it had nothing to respond to is this.

### Step 4: the block output store (found while checking what else reads batch means)

`luthi/episode_store.py`, used by every living `PredictiveCodingBlock`
(`x = self.episode_store(block_input, x)`, after the FFN):

- context = the block input pooled over **batch and sequence**;
- value = the block output pooled the same way;
- salience = mean `|output|`, which clears the 0.1 threshold on every call
  at the latent scales this trunk runs at (block-output RMS 0.3–9.1 in the
  768 family);
- recall when best cosine ≥ `episode_recall_threshold` (0.7 in the v3+
  arms); the stored contexts are near-identical, so it fires;
- blend: `output ← 0.7 · output + 0.3 · recalled`, broadcast to every
  token of every sequence.

So in a batched trunk this store does not work as an episodic memory. It
shrinks every token's deviation from the batch mean by 30% per block and
re-injects a batch-constant vector. It does this in training, and in frozen
eval too (`freeze_plasticity` skips its write, not its recall). The dead arm
removes it (`dead_ffn=True → episode_store = None`). So every living-vs-dead
comparison in the registry compared two mechanisms against none.

Measured on an untrained d32 model with random tokens, then on the trained
smoke checkpoint, with the counters added in this pass:

| | recall duty | last similarity | blend share of token norm |
|---|---|---|---|
| untrained, random tokens, 80 forwards | 0.9875 (79/80; the first had nothing stored) | 0.9946 | 0.24 |
| trained 300 steps, frozen reads, blocks 0–3 | **1.000** in every block | 0.91–0.9999 | 0.23–0.28 |

A search of `docs/` found no analysis of this store anywhere: no audit,
review, or log. `scripts/localize_offset_in_block.py` reads its third column
after the store, so the "ffn delta" it reports includes the store's blend. The
"x2 post-ffn" label hid it.

### Step 5: a content-sensitive rule — and the instability that taught the cap

The direct fix keeps everything about the rule except its sufficient
statistic. `write_rule="covariance"` computes the write from per-token
deviations, `mean_t (out_t − ō) ⊗ clamp(drive · (e_t − ē) · trust)`. The
prediction matrix learns from the per-token error, which is its true
gradient. Precision tracks per-token error energy. The drive, trust, sparse
gate, metaplasticity, gain, homeostasis, error_acc, and the returned mean
`pred_error` are all shared with the legacy rule.

**First version: uncapped. It ran away.** Under 20×-scaled structured
input at 10× the production `pc_rate`, the weight norm went from 3.0 to
**1.3e10** in 3,000 steps, while the legacy rule on the same stream peaked at
154. The mechanism is the classic Hebbian loop: the centered output grows
with the weight it is writing, while the prediction matrix, learning
10× slower, lags behind the error it should be explaining.

**Fix: same budget, different direction.** The covariance write is capped
at the Frobenius norm of the mean-field write the same batch would have
produced. Two reasons, both load-bearing. First, attribution: a difference
between arms cannot then be "the covariance arm simply writes more". Second,
stability: on the same stress stream the capped rule peaked at 1,580 (legacy
154). At the production rates (`pc_rate` 1e-3, `pred_learning_rate` 1e-4),
over 5,000 steps it grows *less* than the legacy rule:

| input scale | legacy max ‖W‖ | covariance max ‖W‖ (from 3.01) |
|---|---|---|
| 1× (LayerNorm scale) | 3.68 | 3.09 |
| 20× | 9.92 | 5.84 |

The 10×-rate transient is recorded as watch item D1, not hidden: the cap
bounds each step, not consistent accumulation, and homeostasis is the brake.

### Step 6: the attractor replay bug (found while wiring the rule through consolidation)

`consolidate_layer_attractor` called `pc_self_modify` with every keyword
defaulted, so `relative_trust=False`. On a relative-trust layer, each replay
then ran the absolute-precision branch, which clamps the freed ledger (values
up to 1e12) into `[0.1, 10]`. That flattens exactly the differentiation the
v5 family exists to have. No registered training arm combined attractor
replay with relative trust; the 07-29 M4 stop-gate rerun applied it offline
to v5 checkpoints. The fix passes the layer's trust regime through and pins
`write_rule="mean_field"`: a stored episode is one batch-mean pattern, with
no per-token spread for the covariance rule to read.

### Step 7: run the new instruments end to end on four smoke arms

Same config as step 2, 300 steps each: dead; living (legacy rule, legacy
store); living with `block_episode_store="off"`; living with the covariance
rule and the store off. Each was converted to a jepa_pilot-shaped run dir and
read with the repo's own `rebuild_run`, `living_write_probe.py`, and
`adaptation_eval.py`, so the scripts Brian will run on E: have run end to end.

**Living-write probe** (per block 0–3):

| arm | living-change top_share | out_align | store recall duty |
|---|---|---|---|
| living (legacy) | 0.995 / 0.987 / 0.978 / 0.962 | 0.99 / 0.99 / 0.99 / 0.97 | 1.000 all |
| store off (legacy rule) | 0.992 / 0.991 / 0.987 / 0.865 | 0.99 / 0.99 / 0.99 / 0.90 | — |
| covariance, store off | **0.124 / 0.095 / 0.132 / 0.119** | 0.08 / 0.28 / 0.46 / 0.08 | — |

The covariance rule removes the rank-1 signature, as designed. Its living
change is also *larger*: `rel_norm` 0.44 in block 0, against 0.12 for the
legacy rule. That is consistent accumulation under the same per-step budget,
and it is the thing to watch at 768 (D1).

**Profile** (held-out NMSE; standardized probe for trunk and for the
embedding stream; offset dominance and effective rank, blocks 0–3):

| arm | NMSE | probe trunk / embedding | offset dominance | eff. rank |
|---|---|---|---|---|
| dead | 0.563 | **0.072** / 0.068 | 0.23 / 0.06 / 0.03 / 0.04 | 95 / 92 / 94 / 102 |
| living (legacy) | 0.741 | 0.062 / 0.068 | **0.78 / 0.90 / 0.93** / 0.07 | 76 / 62 / 55 / 50 |
| store off | 0.576 | 0.066 / 0.068 | 0.52 / 0.32 / 0.12 / 0.04 | 76 / 65 / 61 / 63 |
| covariance, store off | 0.608 | 0.063 / 0.068 | 0.37 / 0.21 / 0.13 / 0.04 | 62 / 49 / 51 / 61 |

Two things here are mechanism-level, and so legitimate at this scale:

- **The store drives token uniformity up through depth.** With the store,
  offset dominance climbs 0.78 → 0.93 through blocks 0–2 and drops at the
  last block, where VISReg's center term acts. Without it, offset dominance
  falls with depth, as it does in the dead arm. The 768x8 family's
  measured profile has the same shape: 0.79–0.999 through blocks 0–6 in
  every seed, then 0.03–0.07 at block 7 in the two seeds that did not
  diverge. Seed 95, killed by its gradient bomb, stays at 0.999. Result 6 of the 768 spec called it "architectural".
  This is consistent with it being the store. P1 tests that at 768.
- **Only the dead arm's trunk beats its own embedding baseline on the
  probe.** That is suggestive only, at this size.

The NMSE ordering is a toy-scale number and, per the 08-15 ruling, says
nothing about 768.

**Adaptation eval** (32 held-out documents; read 4 windows, score 2):

| arm | exact null | NMSE B−A (t) | C−A (t) | **B−C (t)** |
|---|---|---|---|---|
| dead | **true** | 0 | 0 | 0 |
| living (legacy) | false | −0.00004 (−0.5) | −0.00014 (−6.4) | +0.00010 (+1.1) |
| store off | false | +0.00001 (+1.4) | +0.00000 (+0.9) | +0.00000 (+0.6) |
| covariance, store off | false | −0.00003 (−3.3) | −0.00003 (−4.7) | +0.00001 (+1.7) |

The positive control holds: the dead arm is bitwise null. The harness is
sensitive: generic drift reaches |t| = 6 in NMSE, and for the covariance arm
`l_pred` C−A reaches t = +60. No arm shows document-specific adaptation at
this scale. Reading *anything* moves the scores in a consistent direction,
and reading *this* document buys nothing beyond that. For the legacy rule
this matches the diagnosis. For the covariance rule it is the toy-scale null
that E3 exists to test at 768.

## Conclusion

1. **The legacy living write cannot see content within a batch, and its
   accumulated change is ~rank-1 along the mean output.** This is
   established by the arithmetic and confirmed when the code runs. Whether it
   holds at 768 is E2, with thresholds frozen.
2. **Every living block blends a stored batch-mean output into ~every
   forward, and the dead control has no such block store.** Every
   living-vs-dead verdict in the registry carries this confound. KF7 is
   SUSPENDED (proposed here; ratified the same day, audit B2). The store is the leading candidate for the
   "architectural" offset profile of the 768 family (E3 P1).
3. **The covariance rule removes the rank-1 signature at the same per-step
   budget.** It is stable at production rates, and it is opt-in. Whether it
   produces document-specific change at scale is E3 P3. At toy scale it
   doesn't.
4. **Nothing in the repo before today could have produced conclusions 1–2,**
   because no instrument read the living change itself or the store's
   behavior, and no protocol step asked the capacity question. Both now
   exist: the instruments, and the standing rule in CLAUDE.md.

## Artifacts

- Code: `luthi/v2/pc_ops.py` (`write_rule`), `luthi/v2/living_layer_pc.py`
  (`write_rule`, `context_mode`, `_last_output_mean`),
  `luthi/v2/hybrid_block_pc.py` (`block_episode_store`, store reporting),
  `luthi/episode_store.py` (counters), `luthi/v2/consolidation.py` (A6),
  `luthi/v2/living_diagnostics.py`, `luthi/v2/adaptation_eval.py`,
  `luthi/v2/run_rebuild.py`, `luthi/v2/eval_heldout.py` (probe controls),
  `luthi/v2/jepa_runner.py` (deep-cadence `living_change_*`,
  `output_episode_*`), `scripts/living_write_probe.py`,
  `scripts/adaptation_eval.py`, `scripts/jepa_pilot_driver.py` (LC-768
  arms, C++ fallback, embedding baseline), six forensic scripts (freeze).
- Tests: `tests/test_living_write_rule.py`, `tests/test_adaptation_eval.py`,
  `tests/test_scripts_freeze_plasticity.py`,
  `tests/test_living_channel_program_arms.py`.
- Data: the CPU smoke checkpoints and probe JSON were scratch artifacts of
  this session and are not in the repo. The scripts above regenerate the
  tables from any run dir (`--run-dir`). The production reads are E1/E2/E4.
- Commits: listed in the audit's closing note once pushed.
