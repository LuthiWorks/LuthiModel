# VBG family registration: two anchors, three seeds each

**Date:** 2026-08-07, 22:30
**Design/registration:** Fable 5. **Build:** Opus 5 (`bb44fc8`, return
note in the spec doc — read §A-§F there; every number verified firsthand
against `scripts/calibrate_vbg.py` before this registration, including
the floor spot-check: dk5000 share 0.790 sketch / 0.833 full, ratio 64.8.
Opus's §A catch of the spec's floor/arrest mislabel is CONFIRMED and
ratified — the spec's w_cap arithmetic would have overdosed ~112x.)
**Registered BEFORE launch. Six runs, sequential, overnight.**

## Rulings on the return note

- **§A ratified:** w_cap = 18 (Opus's corrected arithmetic).
- **§B ruled by running both:** the anchor question — floor (w_share
  1.5, trace-normalized) vs arrest (10.3, raw) — is really "was the
  scale-fight component of the arrest dose load-bearing?" Trace
  normalization strips exactly the re-inflation pressure that may have
  done the lifting on a variance-starved floor. Two sub-families answer
  it: `probe_d8_vbg` (stage 45, normalized, 1.5) and `probe_d8_vbg_raw`
  (stage 46, raw, 10.3). Same cap term (18 @ 0.05) in both.
- **§C ratified at 0.05:** the cap exists to tax over-funding, not to
  squeeze partial recoveries; the sketch-space permissiveness toward the
  recovered state is acceptable for v1 and recorded.
- **§D deviation ACCEPTED, fence amended:** the four auxiliary loss
  fields stay. Opus's argument is the observability principle applied
  correctly — a regularizer whose magnitude cannot be read cannot be
  dosed; §B's own reconstruction-by-arithmetic proved the cost.
- **§E ratified** (non-persistent power buffer; full-space always-on
  instrument distinct from the sketch-space governor).

## Gates (frozen; Brian's goal is the gate)

Per sub-family: **CONFIRMED** = logged `deep.stable_rank >= 20` at two
consecutive deep firings at/after step 2000 AND at the final firing, in
**>= 2 of 3 seeds** (46, 95, 97). **FLOOR** = a seed killed, or final
pooled eff < 20 (per-seed). Sub-family with 0-1 of 3 confirming = NOT
CONFIRMED. Cross-family comparison (normalized vs raw) is read on the
tally plus the `top_dir_share` trajectories — the new instrument every
run now carries.

**Recorded, not scored:** per-block eff (floors at blocks 0/3/6 vs
ungoverned), probe lift (the flattery-proof capability read), l_vbg_cap
and l_vbg_share magnitudes over time (the first runs whose doses are
readable in their own record), offset dominance, SIGReg.

## Confounds, stated in advance

1. Warmup base: every result is warmup+governor; attribution is to the
   combination against warmup-only's 1/3 base rate.
2. The cap acts in sketch space, the gate in logged stable_rank —
   related but not identical observables; a governor that wins its own
   game but misses the gate is a NO VERDICT with a lesson, not a fail.
3. w_share values are Opus's calibration judgments ratified by me; a
   nll here retires the dose pair, not the governor.
4. Six sequential runs ≈ 4.5-5 h unattended; guards live after 1000 per
   arm; any kill is that seed's verdict, not a pipeline fault.

## Launch

```
python scripts/jepa_pilot_driver.py --stage 45 --seeds 46,95,97 --epochs 1 --max-batches-per-epoch 3000 --heldout-batches 5
python scripts/jepa_pilot_driver.py --stage 46 --seeds 46,95,97 --epochs 1 --max-batches-per-epoch 3000 --heldout-batches 5
```

---

# FAMILY VERDICT (2026-08-08, 00:30): 0-for-6 on the gate — with the diagnosis in hand

**Normalized (stage 45): 0/3** — all seeds killed at ~1000-1300. **Raw
(stage 46): 0/3 on the gate**, but seeds 46/95 completed with
recovery-class breadth (eff 106/119, min blocks 28.9/66.0) and
top_dir_share driven DOWN all run (95: 0.341→0.081; 46: 0.28→0.19).
Seed 97 died in both sub-families (share 0.776 at kill — a hostile path
throughout the record).

**§B answered: the scale-fight is load-bearing.** Shape-only pressure
dies early; raw pressure (re-inflation included) completes and recovers
breadth. Term B stays raw.

**The gate/cap incompatibility — design error #2, mine:** cap 0.05
mathematically cannot deliver stable_rank 20. With ~100 directions
sharing the tail, top share parked at 0.05-0.08 pins stable_rank at
~2-4 (measured: 1.2-3.0). Stable 20 requires share ≈ 0.02 — exactly the
§C tightening Opus offered and I declined. Overruled by measurement;
ratifying Opus's number.

# V2 REGISTRATION (cap 0.02, raw anchor) — one variable vs stage 46

Arm `probe_d8_vbg2` (stage 47): identical to `probe_d8_vbg_raw` except
`cap = 0.02`. Same gates (stable_rank >= 20, two consecutive >= 2000 +
final, 2-of-3 seeds 46/95/97), same recorded reads. Prediction, stated:
if the cap mechanism is what parks the share, v2 parks it near 0.02 and
stable_rank lands in the 8-25 range — the gate sits inside the
uncertainty, which is where an honest gate belongs.
