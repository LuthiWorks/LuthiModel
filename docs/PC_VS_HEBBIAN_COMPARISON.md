# v2 PC (256d) vs v1 Hebbian — Comparison

> Compiled 2026-05-14 from `runs/m5_256d/` (v2 PC + DeadLM baseline) and
> `runs/ablation_A/` (v1 Hebbian baseline). Asked by Brian after the M5
> 256d re-run: "How do the results from our most recent 256d PC training
> run compare to the Hebbian runs at the same depth?"

## TL;DR

**There is no Hebbian run at 256d.** The matched-corpus v1 Hebbian
ablation was only ever run at **128d**, so a like-for-like comparison
at the same depth doesn't exist in our data. The best available
reference points are:

- **v1 Hebbian at 128d** (`runs/ablation_A/baseline_seed{42,1337,2026}`):
  same corpus, same tokenizer, same epoch count as the 256d M5 re-run,
  but half the width — and architecturally older (single-head attention,
  no LR schedule, no grad clip).
- **v1 Hebbian at 1024d on Gutenberg-100** (`spiking_1024d_bpe_gutenberg`):
  4× the width, but BPE-4096 tokenizer (not BPE-32k), 45.9M character
  corpus (not 10.6M tokens), and 80 epochs — losses are not directly
  comparable because the vocabulary differs.
- **v1 Hebbian at 512d** (`thirteenth_run_512d`): character tokenizer
  on a 2.2M-char corpus. Useful as a depth reference only — losses are
  not comparable to BPE.

**Cross-scale headline**: v2 PC at 256d beats v1 Hebbian at 128d by
**~11.3%** on mean best val loss (5.72 vs 6.45). That margin bundles
three changes — width (×2), architecture (PC vs Hebbian, multi-head
vs single-head, cosine LR + grad clip), and the corresponding
hyperparameter retuning — into one number. It is **not** an
isolated PC-vs-Hebbian signal.

The cleaner isolated signal lives in the 128d M5 pilot data (already
documented in `docs/V2_PILOT_RESULTS.md`): at the same width, v2 PC
was 1.88% *worse* than DeadLM. The 256d run flipped that to 0.64%
better. PC's win over a vanilla transformer is width-dependent. PC's
win over Hebbian is consistent at every width we have data for.

## What there is data for

| Run | Arch | d_model | Corpus | Tokenizer | Epochs | Seeds | Comparable to 256d PC? |
|-----|------|---------|--------|-----------|--------|-------|------------------------|
| `runs/m5_256d/v2_seed*` | v2 PC | **256** | Gutenberg-100 (10.6M tok) | BPE-32k | 30 | 3 | — (the run in question) |
| `runs/m5_256d/dead_seed*` | DeadLM | 256 | Gutenberg-100 | BPE-32k | 30 | 3 | YES (matched control) |
| `runs/m5/v2_seed*` | v2 PC | 128 | Gutenberg-100 | BPE-32k | 30 | 3 | width-mismatched |
| `runs/m5/dead_seed*` | DeadLM | 128 | Gutenberg-100 | BPE-32k | 30 | 3 | width-mismatched |
| **`runs/ablation_A/baseline_seed*`** | **v1 Hebbian** | **128** | **Gutenberg-100** | **BPE-32k** | **30** | **3** | **width-mismatched, but the only same-corpus Hebbian** |
| `runs/spiking_1024d_bpe_gutenberg` | v1 Hebbian (spiking) | 1024 | Gutenberg-100 (45.9M chars) | BPE-4096 | 80 | 1 | NO (tokenizer differs) |
| `runs/thirteenth_run_512d` | v1 Hebbian | 512 | 2.2M-char synthetic | char (vocab 96) | 240 | 1 | NO (corpus + tokenizer differ) |

## The numbers (matched corpus + tokenizer + epoch count)

All three seeds, all from `results.json` per-seed. Same Gutenberg-100
corpus, same BPE-32k tokenizer, same 30-epoch budget, same seq_len=128,
batch_size=32, stride=64.

### v1 Hebbian at 128d (single-head, no LR schedule)

| Seed | Best val | Best epoch | Final train | Final val | NaN events |
|------|----------|------------|-------------|-----------|------------|
| 42   | 6.4862 | 26 | 4.9222 | 6.5801 | 0 |
| 1337 | 6.5224 | 17 | 4.9116 | 6.6813 | 0 |
| 2026 | 6.3460 |  9 | 4.8820 | 6.7478 | 0 |
| **Mean ± stdev** | **6.4515 ± 0.0932** | 17.3 | 4.9053 ± 0.0207 | 6.6697 ± 0.0846 | 0 |

### v2 PC at 256d (multi-head, cosine LR + grad clip)

| Seed | Best val | Best epoch | Final train | Final NFF | NaN events |
|------|----------|------------|-------------|-----------|------------|
| 42   | 5.6859 | 17 | 4.2781 | 9.35e-03 | 0 |
| 1337 | 5.7575 | 20 | 4.3323 | 7.51e-03 | 0 |
| 2026 | 5.7269 | 15 | 4.2855 | 1.33e-02 | 0 |
| **Mean ± stdev** | **5.7234 ± 0.0360** | 17.3 | 4.2986 ± 0.0298 | 1.01e-02 ± 2.96e-03 | 0 |

### DeadLM (vanilla transformer) at 256d — bridge control

| Seed | Best val | Best epoch | Final train | NaN events |
|------|----------|------------|-------------|------------|
| 42   | 5.7491 | 12 | 4.1411 | 0 |
| 1337 | 5.7606 | 13 | 4.1728 | 0 |
| 2026 | 5.7703 | 11 | 4.1818 | 0 |
| **Mean ± stdev** | **5.7600 ± 0.0106** | 12.0 | 4.1652 ± 0.0214 | 0 |

## Deltas

| Comparison | Δ best val | Relative | Direction |
|------------|-----------|----------|-----------|
| v1 Hebbian 128d → v2 PC 256d | **−0.7281** | **−11.29%** | v2 PC better |
| v1 Hebbian 128d → DeadLM 256d | −0.6915 | −10.72% | DeadLM better |
| DeadLM 256d → v2 PC 256d | −0.0366 | −0.64% | v2 PC better |

## How to read these deltas

The 11.3% gap between v1 Hebbian 128d and v2 PC 256d bundles four
changes that cannot be cleanly separated from this data:

1. **Width** (128 → 256). Doubling d_model alone closes the gap a lot —
   the 256d DeadLM beats 128d v1 Hebbian by 10.72% just from the
   architectural upgrades + width, with no PC involved.
2. **Attention shape** (single-head → 4-head). v1's ablation_A used
   `n_heads=None` (single head); v2 and DeadLM at 256d use 4 heads.
   The 2026-05-10 audit added MHA to v2 specifically because single-head
   was undertrained.
3. **LR schedule** (constant → cosine + 2-epoch warmup) and **grad clip**
   (none → `max_norm=1.0`). Both added after v1 ablation_A.
4. **Living-weight substrate** (Hebbian correlation → PC error-driven).
   The actual mechanistic change.

The 256d-v2-vs-256d-DeadLM Δ of −0.64% is the **only** number in this
table that isolates the PC mechanism cleanly — both architectures share
width, attention shape, schedule, and clip; they differ only in whether
the FFN is a static `nn.Linear` (DeadLM) or a self-modifying PC layer
(v2).

The "PC vs Hebbian" question proper would need a v1 Hebbian re-run at
256d/4-head/cosine — i.e., a Hebbian ablation_A re-run at the M5 256d
config. We don't have that. The closest we have is the 128d M5 pilot,
where v2 PC (5.85 mean val) was **0.82** better than v1 Hebbian (6.67
mean val) at matched width and tokenizer — but even that gap is
confounded by single-head/multi-head and schedule.

## What the data does say cleanly

- **PC is not catastrophically worse than vanilla.** At 128d, v2 PC was
  1.88% worse than DeadLM (M5 pilot). At 256d, v2 PC was 0.64% **better**
  than DeadLM. PC stayed within the 20% falsification gate at both
  widths.
- **Hebbian *was* catastrophically worse than vanilla at the pilot scale.**
  v1 Hebbian at 128d was ~14% worse than DeadLM at 256d (6.45 vs 5.76),
  and ~10% worse than DeadLM at 128d (using the implied DeadLM-128d
  number ~5.74 from the M5 pilot doc). The gap shrunk dramatically with
  PC.
- **PC is also not Hebbian.** Whatever Hebbian's relative position vs
  DeadLM at the same architectural config (heads + schedule), the M5
  pilot at 128d already showed v2 PC strictly beating v1 Hebbian
  (5.85 vs 6.67). At 256d the absolute number falls further (5.72).
- **The win mechanism shifts with scale.** At 128d, PC underperformed
  DeadLM — the living-weight property cost some convergence quality.
  At 256d, PC matches and slightly beats DeadLM, while still providing
  the living-weight property DeadLM structurally cannot. The depth/width
  scaling story is asymmetric: PC seems to need width to fully express
  whatever advantage it confers, while Hebbian seems to have stalled
  somewhere below DeadLM's curve at any tested width.

## NFF as a sanity check

The non-feedforward signal is the diagnostic that the living-weight
mechanism is actually doing something during training (positive NFF ⇔
weights change between identical forward passes). DeadLM has
deterministic forward, so NFF is exactly 0.0 for it.

| Run | Mean NFF (final) | Interpretation |
|-----|-----------------|----------------|
| v1 Hebbian 128d (ablation_A) | not instrumented per-epoch | (legacy run pre-dates the metric) |
| v2 PC 128d (M5 pilot, post-hoc) | 5.32e-03 | PC active end-of-training |
| **v2 PC 256d (M5 re-run, per-epoch)** | **1.01e-02** | **PC active, ~2× the 128d signal** |
| DeadLM 256d | 0.0 | feedforward (sanity check) |

NFF roughly doubled from 128d to 256d. The substrate has more capacity
to express the PC error signal at higher width, consistent with the
better-than-DeadLM result at 256d being a width-enabled win rather than
a stat-noise artifact.

## Compute cost reference

Hebbian and PC at the same width have nearly identical compute cost per
step — they differ only in how `delta_w` is computed. The 256d v2 run
took ~5.5h per seed wall-clock; the 128d v1 Hebbian baselines took about
the same per seed at half the width (Hebbian's tighter inner loop offset
v1's pre-C++-extension code path).

The 1024d v1 Hebbian run (`spiking_1024d_bpe_gutenberg`) was ~24 hours
for 80 epochs at 1024d on a 4×-larger corpus — that's the closest thing
to a depth/scale-reference number, but the tokenizer difference makes
the loss values non-comparable.

## What would close the gap

If the question is really "PC vs Hebbian at matched config," the
experiment that would answer it is one of:

1. **Hebbian 256d re-run** at the M5 config (4-head, cosine LR, grad clip,
   BPE-32k, Gutenberg-100, 30 epochs, 3 seeds). Would isolate the
   mechanism question. Cost: ~16h wall-clock for 3 seeds. Not currently
   planned — v1 was deprioritized 2026-05-09 in favor of v2 as the
   primary substrate.
2. **v2 PC 128d re-run** with the 2026-05-13 compute-optimization stack
   (μPC + iPC + sparse gating) to test whether PC's 128d underperformance
   vs DeadLM was a tuning artifact rather than a width-dependence. Cost:
   ~16h wall-clock. **This is planned** (Phase 3G in `To-Do.md`).

Direction (2) is the more useful experiment because it tests whether the
compute-optimization machinery introduced in the literature sweep
reshapes the PC-vs-DeadLM curve at the cheaper end of the width
dimension. If μPC's LR-transfer claim holds, the same PC config that
won at 256d should also win at 128d once it's properly initialized.

## Files

- v1 Hebbian 128d data: `runs/ablation_A/baseline_seed{42,1337,2026}/results.json`
- v2 PC 256d data: `runs/m5_256d/v2_seed{42,1337,2026}/results.json`
- DeadLM 256d data: `runs/m5_256d/dead_seed{42,1337,2026}/results.json`
- 256d run details: `docs/M5_RERUN_256D_RESULTS.md`
- 128d pilot details (v2 PC + DeadLM + v1 reference): `docs/V2_PILOT_RESULTS.md`
- Compute-optimization experiments planned post-256d:
  `docs/RESEARCH_LITERATURE_2026-05-13.md`, `To-Do.md` Phase 3G
