"""Two-arm JEPA pilot driver — Experiment 1, JEPA edition (2026-07-15).

The falsification-critical run, merged with the M8 256d de-risking pilot
(critical-path item 1). One instrumented sweep answers three
pre-registered questions (docs/research/2026-07-15_falsification-
preregistration.md; protocol living-weights-experiments.md, JEPA
edition):

  (a) the pilot's collapse-kill thresholds (pilot-set machinery derives
      them per-run; this sweep sanity-checks the derived values);
  (b) does representation collapse behave differently when the weights
      self-modify — the dead arm is the direct control;
  (c) matched capacity: does the living arm sit above the dead arm's
      effective-capacity curve on held-out latent prediction + probes?

Arms and stages (5 seeds per condition — Brian, 2026-07-15):

  stage 1   living@256 x5  +  dead@256 x5     the matched point
  stage 2   dead@192 x5  +  dead@384 x5       the curve's shape
  stage 3   dead@512 x5                       the upper bracket

Stage 1 decides half the outcomes alone: if the living arm loses or
ties at the matched point, KF2-strong dies and stages 2-3 are
unnecessary. Bracket {192,256,384,512} pends Brian's S1 ratification.

Per run: text-only JEPA (round-1 scope), leakage-gapped 2% holdout,
end-of-epoch held-out eval (built into JEPATrainer), then a final
held-out eval + next-token linear probe WITH its shuffled-label floor.
Results land in <run_dir>/pilot_result.json — the completion marker, so
the sweep is resumable (completed runs skipped; a failed run stops the
queue loudly: a condition with fewer seeds than its siblings corrupts
the variance estimate).

Device: DirectML -> CUDA -> CPU (the train_pc pick). NOTE: the JEPA path
has only ever run on CPU (the smokes); the first pilot run doubles as
the device shakeout — watch the first hundred steps.

Usage:
  python scripts/jepa_pilot_driver.py --stage 1 --dry-run
  python scripts/jepa_pilot_driver.py --stage 1
  python scripts/jepa_pilot_driver.py --aggregate
Smoke (CPU, minutes):
  python scripts/jepa_pilot_driver.py --stage 1 --smoke
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parent.parent
# The driver imports luthi in-process (unlike the retired LM driver,
# which shelled out); make it runnable from anywhere.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
OUTPUT_ROOT = REPO_ROOT / "runs" / "jepa_pilot"

SEEDS = (42, 43, 44, 45, 46)

# (arm, d_model) conditions per stage. Bracket amended by Brian
# 2026-07-16: a SINGLE overshoot point (dead@512, ~4x the living FFN's
# nominal weight count) replaces the {192, 384, 512} curve; stage 3 is
# reserved for the 384 fallback the pre-registered read requires if
# dead@512 wins or ties (see the pre-registration's bracket entry).
STAGES: dict[int, list[tuple[str, int]]] = {
    1: [("living", 256), ("dead", 256)],
    2: [("dead", 512)],
    3: [("dead", 384)],  # fallback only -- run on an inconclusive stage 2
    # Run 2 of the staged-configuration ladder (Brian, 2026-07-17): the
    # FULL living configuration -- backward pass ON (DNR 9b: the
    # task-salience -> plasticity channel) + consolidation ON (episodes
    # become structure). Stage-1's "living" arm ran the MINIMAL config
    # (both off, inherited from smoke defaults); the ladder turns
    # subsystems on stepwise so improvements stay attributable. Held for
    # run 3: plasticity taper, inverted-U gain, recall-gate tightening.
    4: [("living_full", 256)],
    # Run 3 (Brian, 2026-07-17 evening): the three held builds land --
    # plasticity taper (formative->mature, floor 0.2 from 50% progress),
    # inverted-U gain, recall gate 0.5 -> 0.7 -- AND the living arm moves
    # to 512 alongside the existing dead@512 control (no dead rerun: all
    # three changes are living-side only). Attribution caveat, recorded:
    # width and the three builds move together in this rung by Brian's
    # ruling; a bridge arm (living_full@512, no builds) can be run later
    # if attribution needs splitting.
    5: [("living_v3", 512)],
    # The BRIDGE arm (Brian, 2026-07-18): living_full's exact config at
    # 512 -- no taper, no gain, recall gate 0.5. Fills the attribution
    # cell run 3 confounded: vs living_v3@512 isolates the three builds
    # at fixed width; vs living_full@256 isolates width at fixed config.
    # Tracking arm -- no verdict force; reads frozen in the registry.
    6: [("living_full", 512)],
    # Run 5, the DATA-SCALING cells (Brian, 2026-07-18: "increase the
    # data by 4x" -- the param rule, data ~ width^2). Same 512 arms on a
    # 4x-token SUPERSET corpus (the 1x corpus + 382 more books, ~50.4M
    # tokens). Stage 7a = dead (the pure starvation test: does the dead
    # NMSE curve's worsening-with-width reverse with data?); stage 7b =
    # living_v3 (does the living advantage's magnitude recover?).
    7: [("dead_4x", 512)],
    8: [("living_v3_4x", 512)],
    # Run 6, the DEPTH rung -- AMENDED 2026-07-20 (Brian): the depth
    # family now BUNDLES the two cheap levers with the depth increase
    # (2 -> 4 blocks): cosine LR decay (the registered cosine rung,
    # folded in) and 2x SIGReg weight (the variance-floor lever from the
    # seed44 denominator-race finding). One-variable attribution vs the
    # d2 anchors is deliberately traded for speed; if the bundle moves
    # the picture, single-lever follow-ups split it (bridge precedent).
    # First JEPA-era outing of the muPC depth machinery (exponent 0.25).
    # The depth-only registration is superseded -- see the 2026-07-20
    # amendment in the falsification pre-registration doc.
    9: [("living_v4_4x_d4", 512)],
    # RUN v5 (Brian's 2026-07-21 resequencing): precision awakening as
    # its own family -- v4 + relative_trust, single variable. The
    # dormant-machinery bundle (sparse/iPC/attractor/lambda) is v6.
    10: [("living_v5_4x_d4", 512)],
    # Seed44 ROBUSTNESS RERUN (Brian's 2026-07-26 ruling): identical
    # config and data order as stage 10's seed44; GPU float
    # nondeterminism supplies the only perturbation. Distinguishes a
    # robust trigger (the ~58650 trust event recurs) from a knife-edge
    # one (it does not). Distinct arm name so run dirs and verdict
    # filters keep it out of the registered v5 family (never-pool).
    # Unregistered descriptive probe -- no frozen prediction. Replaces
    # the dead_4x_d4 control in the SCHEDULE only; that control remains
    # a registered obligation (deferred; no depth claims until it runs).
    11: [("living_v5_4x_d4_rerun", 512)],
    # VALIDATION PROBE (2026-07-27, not a registered family): v5's exact
    # configuration plus the episode-store admission fix and the homeostatic
    # band. Short run, one seed. Its only purpose is to check the four
    # predictions in docs/research/2026-07-27_episode-store-frozen-defect.md
    # against real training before v6 commits to a multi-day family -- the
    # fix is unit-tested but has never run at scale, and the defect it
    # repairs was invisible to every counter for five whole families.
    # Distinct arm name so its artifacts can never pool with a family.
    # --- Depth-8 probe era (stages 12-52, 2026-07-29 to 08-09): PRUNED ---
    # 2026-08-10, Brian's ruling: the probe arms and stages of the depth
    # investigation are removed from the live driver. Every arm
    # definition, dose, and verdict survives in git history and in
    # docs/research/ (start: 2026-08-08_depth-investigation-consolidated
    # .md). Active work continues from the LLM-JEPA pivot arm below.
    50: [("probe_d8_llmjepa", 512)],
    52: [("probe_d8_llmjepa_v2", 512)],
    # SET-POINT ADAPT-RATE SWEEP (2026-08-10, external protocol step 2,
    # Brian's go): the v2 arm with the homeostatic anchor made mobile.
    # Default is 1e-6 (glacial); the sweep asks whether a mobile anchor
    # changes the late collapse. One parameter, nothing else.
    53: [("probe_d8_v2_spa5", 512), ("probe_d8_v2_spa4", 512),
         ("probe_d8_v2_spa3", 512)],
    # VISREG REPLACEMENT (2026-08-11, external protocol step 3, Brian's
    # build order; rulings in docs/reviews/2026-08-10_pruning-and-visreg-
    # brief-for-opus.md). v5-at-d8 base, muPC off, SIGReg REPLACED by
    # VISReg (arXiv 2606.02572) on TRUNK latents, convex mix 0.6, paper
    # defaults. NTP is OUT of v1 (aux-theorem read: its protective
    # condition is structurally unavailable at pilot scale) -- the
    # cleanest possible one-variable swap against the v5-d8 record.
    54: [("probe_d8_visreg", 512)],
    # RULED-SCALE FAMILY (2026-08-11): 768x8, VISReg, full-corpus epoch.
    # See the arm block below and the spec's Amendment 2.
    55: [("probe_768_visreg", 768)],
}

# Per-arm model configuration -- single source of truth, shared with
# pilot_verdict.py's checkpoint-rebuild path (the eval must reconstruct
# the arm EXACTLY, including the recall gate, which is live at eval).
ARM_CONFIGS: dict[str, dict] = {
    "dead":        {"dead_ffn": True},
    "living":      {},  # minimal: smoke defaults (BP off, consolidation off)
    "living_full": {"backward_pass_enabled": True,
                    "consolidation_enabled": True},
    "living_v3":   {"backward_pass_enabled": True,
                    "consolidation_enabled": True,
                    "learning_gain_enabled": True,
                    "episode_recall_threshold": 0.7},
}
# Trainer-side (non-model) per-arm settings.
ARM_TAPER: dict[str, bool] = {"living_v3": True, "living_v3_4x": True}

# Data-scaled arms: model config identical to their base arm; the corpus
# is the 4x superset filelist. Registered as distinct arm names so run
# dirs, verdict filters, and the never-pool rules all keep them separate.
ARM_CONFIGS["dead_4x"] = dict(ARM_CONFIGS["dead"])
ARM_CONFIGS["living_v3_4x"] = dict(ARM_CONFIGS["living_v3"])
# The depth arm (run 6): v3's living config at 4 blocks with muPC depth
# scaling enabled -- its first outing under JEPA. n_blocks/mu_pc_* are
# model kwargs, so they ride in ARM_CONFIGS and override the driver's
# --n-blocks default via the arm-config merge in _run_one.
ARM_CONFIGS["living_v3_4x_d4"] = dict(
    ARM_CONFIGS["living_v3"],
    n_blocks=4,
    mu_pc_enabled=True,
    mu_pc_exponent=0.25,
)
ARM_TAPER["living_v3_4x_d4"] = True
# v4 (Brian's 2026-07-20 bundle ruling): the depth arm as actually run --
# same MODEL config as living_v3_4x_d4 (the sigreg weight and LR schedule
# are trainer/loss-side, not model kwargs; see ARM_SIGREG / ARM_COSINE).
# Registered as a distinct arm name so run dirs, verdict filters, and the
# never-pool rules keep it separate from any pure-depth run.
ARM_CONFIGS["living_v4_4x_d4"] = dict(ARM_CONFIGS["living_v3_4x_d4"])
ARM_TAPER["living_v4_4x_d4"] = True
# v5 (Brian's 2026-07-21 resequencing): the precision awakening as its
# own family -- v4's exact config + relative trust (ratio-to-median
# weighting, numerics-only eps, freed ledger). ONE change vs v4; the
# dormant-machinery bundle became v6. Registered in the pre-reg doc's
# corrected precision entries.
ARM_CONFIGS["living_v5_4x_d4"] = dict(
    ARM_CONFIGS["living_v4_4x_d4"], relative_trust=True,
)
ARM_TAPER["living_v5_4x_d4"] = True
ARM_FILELIST: dict[str, str] = {
    "dead_4x": str(REPO_ROOT / "corpus_build" / "gutenberg_4x_filelist.txt"),
    "living_v3_4x": str(REPO_ROOT / "corpus_build" / "gutenberg_4x_filelist.txt"),
    "living_v3_4x_d4": str(REPO_ROOT / "corpus_build" / "gutenberg_4x_filelist.txt"),
    "living_v4_4x_d4": str(REPO_ROOT / "corpus_build" / "gutenberg_4x_filelist.txt"),
    "living_v5_4x_d4": str(REPO_ROOT / "corpus_build" / "gutenberg_4x_filelist.txt"),
}
# Loss-side per-arm setting: SIGReg weight. v4 doubles the default 0.1 --
# the variance-floor lever. Rationale (2026-07-20): living spaces settle
# at per-dim std ~0.3 against SIGReg's implicit unit target, and the
# seed44 denominator race showed over-quieting destabilizes NMSE across
# seeds. A stronger pull toward the isotropic target should hold the
# space louder; registered prediction in the pre-reg amendment.
ARM_SIGREG: dict[str, float] = {"living_v4_4x_d4": 0.2, "living_v5_4x_d4": 0.2}
# Trainer-side per-arm setting: cosine LR decay to a 10% floor (the
# registered cosine rung, folded into v4 by the same ruling).
ARM_COSINE: dict[str, bool] = {"living_v4_4x_d4": True, "living_v5_4x_d4": True}
# Rerun alias (stage 11): byte-identical configuration to the v5 arm
# under a distinct name, so the rerun's artifacts can never pool with
# the registered family by accident.
ARM_CONFIGS["living_v5_4x_d4_rerun"] = dict(ARM_CONFIGS["living_v5_4x_d4"])
ARM_TAPER["living_v5_4x_d4_rerun"] = ARM_TAPER["living_v5_4x_d4"]
ARM_FILELIST["living_v5_4x_d4_rerun"] = ARM_FILELIST["living_v5_4x_d4"]
ARM_SIGREG["living_v5_4x_d4_rerun"] = ARM_SIGREG["living_v5_4x_d4"]
ARM_COSINE["living_v5_4x_d4_rerun"] = ARM_COSINE["living_v5_4x_d4"]
# Validation probe: v5 + the 2026-07-27 store fix + the homeostatic band.
# Identical to v5 in every other respect, so any change in store behaviour is
# attributable to the two flags and comparable to the measured v5 baselines.
# Surprise-drive probe (2026-07-29). probe_storefix + drive_mode="surprise".
# ONE change against probe_storefix, so the drive is attributable: the store
# fix, the band, and the fixed objective are all held constant and already
# measured (probe_storefix_512d_seed45: NMSE 0.597, probe lift 4.67x over its
# own shuffled floor, update_ema 2.6e-6..8.3e-6).
#
# `relative_trust` is already True here via the v5 base, which is what makes
# surprise mode legal at all -- absolute precision weighting saturates the
# +/-1 clamp at 100% and discards the drive magnitude (measured on the real
# code path). That dependency is enforced by a raise in the layer, not
# supplied silently, so this pairing is declared here where it can be
# attributed rather than inherited from a default.
#
# What to read first in the log: substrate_blocks[*].drive_duty. It is 0.0000
# on stationary input and on i.i.d. draws from a fixed distribution, and rises
# only at a genuine shift in the error scale (unit-tested). So on real corpus
# data a nonzero duty means the drive is finding structure the forecast did not
# expect, and a flat zero means it is not -- the discriminator between "quiet
# because familiar" and "quiet because broken" that the first five families
# could not make.
# The 8-block depth arm (Brian's call 2026-07-29, and the depth v6 was already
# registered to start at): probe_surprise at n_blocks=8. Everything else held
# identical to the 4-block probe so depth is the only difference against it.
#
# muPC note: mu_pc_exponent stays at 0.25, inherited from the d4 arms. muPC
# exists precisely so depth does not need per-depth rate retuning, so 0.25 is
# the right thing to carry forward rather than re-tune by hand -- but this is
# the first time that claim gets tested at 8 blocks in this project, and it is
# a claim, not a guarantee. If the deep blocks show a systematically different
# firing rate or a rate/stability problem the shallow ones do not, the exponent
# is the first suspect.
# Trainer-side (non-model) per-arm gradient clipping. Depth 8 diverged at step
# ~2250 without it: grad_norm median 1065 vs 28.4 at depth 4, ~37x larger, at an
# unchanged 3e-4 learning rate, and there was no clipping in the runner at all.
#
# Clip value 1000 is chosen against the measured distribution rather than the
# conventional 1.0 -- gradients here are O(1e3) because the loss itself is
# O(1e2) (SIGReg contributes hundreds), so clipping at 1.0 would reduce the
# effective step ~1000x and stall learning. 1000 sits at roughly the depth-8
# median, so typical steps pass close to untouched while the excursions that
# preceded divergence (2158, 2941, 5555, 8645) are bounded. Deliberately
# aggressive for a stability probe: over-damping shows up as poor learning,
# which is a diagnosable failure, where divergence is not.
#
# Depth-4 arms keep grad_clip_norm=0.0 (off), so every completed family stays
# bit-identical and comparable.
ARM_GRAD_CLIP: dict[str, float] = {
    "probe_surprise_d8": 1000.0,
    "probe_surprise_d8_noproj": 1000.0,
}
# Loss-side per-arm setting: the SIGReg projection head.
#
# `probe_surprise_d8_noproj` is `probe_surprise_d8` with sigreg_projection
# switched from "linear" to "none", and NOTHING else changed -- same 8 blocks,
# same clip of 1000, same seed discipline. It tests one hypothesis: that the
# per-modality nn.Linear head's BIAS absorbs the batch-mean offset, presenting
# SIGReg with centered latents while the trunk keeps the offset. That would be
# structurally the same defect as the BatchNorm removed on 2026-07-28 -- a
# learnable layer standing between SIGReg and the quantity it exists to
# constrain. "none" is nn.Identity(), so SIGReg sees trunk latents directly.
#
# The clip of 1000 is deliberately CARRIED OVER rather than fixed first, so
# this run differs from probe_surprise_d8_512d_seed96 by exactly one thing. It
# is known to be too aggressive (43% of steps clipped) and to kill capability
# on its own -- see the pre-registered readout note in
# docs/research/2026-07-30_sigreg-projection-hypothesis.md for why capability
# metrics therefore CANNOT be read from this run either way.
ARM_BACKPROP_LR_COMPENSATE: dict[str, int] = {
    "probe_surprise_d8_bplr": -1,   # all blocks -- REFUTED 2026-07-31, diverged
    "probe_surprise_d8_bplr0": 1,   # block 0 only
}
ARM_SIGREG_PROJ: dict[str, str] = {"probe_surprise_d8_noproj": "none"}
# `probe_surprise_d8_balanced` is `probe_surprise_d8` (muPC ON, exponent 0.25)
# plus mu_pc_balance_rates. One variable against stage 14, which collapsed.
#
# Why this and not "muPC off": the depth ladder shows muPC's attenuation is
# doing real work -- activation growth first-to-last block is flat at ~1.14
# from 4 blocks to 36 with muPC, and climbs 1.47 -> 3.92 without it over the
# same range. Disabling muPC trades a depth-8 collapse for unbounded growth at
# production depth (36 blocks). Balancing keeps the attenuation and removes the
# imbalance it creates.
# The OPPOSITE adjustment, same magnitude (Brian, 2026-07-30): amplify the PC
# rates by 1/residual_scale instead of attenuating by residual_scale.
# power=+1 made the collapse worse (offset dominance 0.5657 -> 0.8277), and the
# three-point ordering showed TOTAL attenuation tracks the offset rather than
# the PC/backprop ratio -- so more block signal, not less, is the direction the
# data points at.
# Double the knob again (Brian, 2026-07-30): power -1 -> -2, PC-rate multiplier
# 1.682x -> 2.829x at depth 8. Continues a monotonic endpoint ordering:
# power +1 0.9807, power 0 0.9704, power -1 0.6384 (within-batch cosine).
# power=-4 (Brian, 2026-07-30). A LANDMARK, not another rung: with exponent
# 0.25, residual_scale**-4 == n_blocks exactly, at every depth (4x at L=4, 8x at
# L=8, 36x at L=36). The attenuation and amplification cancel and PC rates end
# up scaling LINEARLY WITH DEPTH.
#
# Clip raised 1000 -> 20000 in the same run, deliberately. At power=-2 the clip
# engaged on 93% of steps with a gradient median of 3591, so the ladder had
# stopped being a test of PC rates and become a test of the clip. Amplification
# at 8x will push gradients higher still; leaving the clip at 1000 would make
# this run uninterpretable. 20000 is a catastrophic-runaway backstop that should
# not shape ordinary steps -- engagement rate will be reported, and if it binds
# often the result is confounded and will be reported as such.
#
# TWO variables move here (power and clip). That breaks one-variable discipline
# and is a deliberate trade: at power=-4 the old clip is not a control, it is a
# different experiment. The divergence guards (loss-vs-frozen-baseline, held-out
# NMSE) are the safety net that makes a loose clip affordable.
# power=-8 (Brian, 2026-07-30). Multiplier becomes n_blocks**2: 64x at depth 8,
# 1296x at depth 36. Clip held at 20000 -- ONE variable against stage 20, which
# engaged the clip on 0% of steps, so it is a genuine control here rather than
# the dominant term it had become at clip 1000.
# Embedding scaling (2026-07-31). probe_surprise_d8_amp4 + mu_pc_scale_embedding.
# ONE variable against stage 20, the best configuration found. Tests whether
# scale-matching the trunk input to its correctors restores block-0 offset
# stripping while keeping muPC's attenuation.
# Backprop-LR compensation (2026-07-31). Boosts ONLY the block parameters'
# learning rate by 1/residual_scale, restoring parity with the un-attenuated
# parameters outside the trunk. One variable against stage 20.
# Block-0-ONLY backprop-LR compensation (2026-07-31). Identical to stage 23
# except the boost is confined to block 0. One variable against stage 23.
# SURPRISE DRIVE OFF (Brian, 2026-07-31). Stage 20's configuration with
# drive_mode back to "raw". First isolation test of a recent mechanism against
# the depth-8 problem: the surprise drive (2026-07-29) changes how much the PC
# substrate moves per step, and substrate motion at depth is exactly the axis
# the whole muPC investigation turns on.
# BUNDLE OFF at depth 8 (2026-08-05). Stage 14 (`probe_surprise_d8`) minus
# exactly the seven bundle mechanisms; muPC stays ON at the standard exponent.
# Every flag is written out explicitly -- including the ones that match the
# model defaults -- so this arm reads as its own record rather than
# inheriting silently through the probe chain. Non-bundle machinery is held
# byte-identical to stage 14: episode_recall_threshold stays at the
# living_v3 value 0.7 (base episode store is pre-bundle machinery), the 4x
# filelist / sigreg 0.2 / cosine LR / taper ride below, and the clip of
# 1000 is carried per the stage-16 precedent (it engaged 3% there once the
# trunk was healthy; engagement rate is reported either way).
# Per-arm deep-metric cadence (2026-08-05). The rank-trajectory read showed
# depth-8 block-0 rank is already 9.95 at the first deep firing (step 1000,
# seed96) -- the destruction completes inside the window the default cadence
# never observes. 100 gives 30 observations over a 3000-step probe and makes
# the first 1000 steps visible. Applied per-arm so completed families stay
# bit-identical and comparable.
ARM_DEEP_CADENCE: dict[str, int] = {
    "probe_d8_bundleoff": 100,
    "probe_d8_naked": 100,
    "probe_v5_d8": 100,
    "probe_v5_d8_dk1000": 100,
}
# Per-arm kill-guard delay (2026-08-06, Brian's instruction). Suppresses
# every kill path until the given global step -- loudly; each would-have-
# fired trip is logged. Observation-side knob for short attended probes
# only; the value rides into run_config.json and pilot_result.json.
ARM_GUARD_MIN_STEP: dict[str, int] = {
    "probe_v5_d8_dk1000": 1000,
    "probe_v5_d8_dk5000": 5000,
    # Warmup arm: guards held through the ramp. Init-proximal NMSE has
    # never been measured (every prior run had 100+ full-LR steps of
    # damage before the first check), and the NMSE guard is documented to
    # misread degenerate states -- a barely-trained model tripping it at
    # step 100 would void the run for nothing. Guards live from 1000,
    # exactly when the ramp ends and full LR arrives.
    "probe_v5_d8_warmup": 1000,
    "probe_v5_d8_warmup15": 1500,
    # Surgery arm: resumed at global step 3000; grace to 4000 because the
    # perturbed projections transiently predict worse and the NMSE guard
    # would kill the patient on the table. Live from 4000.
    "probe_v5_d8_surgery": 4000,
}
# Per-arm LR warmup steps (2026-08-06). Rides into LRScheduleConfig;
# 0 (default) preserves every prior arm's schedule bit-exactly.
ARM_LR_WARMUP: dict[str, int] = {
    "probe_v5_d8_warmup": 1000,
    "probe_v5_d8_warmup15": 1500,
    "probe_v5_d8_surgery": 1000,  # schedule continuity with the parent run
    "probe_d8_tc": 1000,
    "probe_d8_wsig": 1000,
    "probe_d8_orth": 1000,
    "probe_d8_tc_wsig": 1000,
    "probe_d8_tc_orth": 1000,
    "probe_d8_wsig_orth": 1000,
    "probe_d8_wsig1": 1000,
    "probe_d8_wsig10": 1000,
    "probe_d8_orth1": 1000,
}
# Scheduled-muPC arm (2026-08-07, Brian's design): the stage-16 healthy
# cell (probe_surprise bundle, muPC OFF, clip 1000) run longer, with the
# runner annealing residual scale to the muPC value mid-run.
# Remedy-probe loss-side settings (2026-08-07). Values are the papers'
# defaults where papers exist (TC window 9 -- odd for exact centering,
# inside the paper's 4-32 ablation band; wsig alpha 0.1, sketch 64) and
# a registered first guess for orth lambda: 0.1, sized offline against
# real checkpoints (penalty mean ~10 per matrix at the measured floor,
# ~4 healthy -> term ~1.0 at floor, ~0.4 healthy, vs loss 4-500; still
# light during the transit window, and a lambda sweep is the cheap
# follow-up if the probes say the direction works).
ARM_WSIG_ALPHA: dict[str, float] = {
    "probe_d8_wsig": 0.1, "probe_d8_tc_wsig": 0.1, "probe_d8_wsig_orth": 0.1,
    "probe_d8_wsig1": 1.0, "probe_d8_wsig10": 10.0,
    "probe_d8_tc_wsig10": 10.0,
}
# (probe_v5_d8_dk1000's ARM_CONFIGS entry lives below, after probe_v5_d8
# itself is defined -- assigning it here raised a KeyError at import.)
# NAKED TRUNK at depth 8 (2026-08-06). ONE variable against
# probe_d8_bundleoff: mu_pc_enabled False. Per the stage-16 caveat this
# removes the residual scaling AND the depth-scaled init together, by
# design -- a clean result implicates muPC as a whole, and separating the
# halves is the mu_pc_exponent=0.0 run if it matters. The exponent key is
# carried (inert when disabled) so the two arms' records differ by exactly
# one value. Same seed as rung 1 (95): the loader is deterministic, so the
# early steps are directly comparable against the diverged run.
# V5 AT DEPTH 8 (2026-08-06, Brian's call). The registered v5 family
# config with exactly one model change: n_blocks 4 -> 8. Inherits the
# pre-07-27 bundle (backward pass, consolidation, learning gain, relative
# trust, episode_recall_threshold 0.7) and muPC at exponent 0.25. NO
# grad-clip entry on purpose: v5 ran unclipped, the clip was a depth-era
# addition, and the divergence guards (proven live twice on 2026-08-05/06)
# are the safety net. Deep cadence 100 -- instrument-side, not mechanism.
ARM_CONFIGS["probe_v5_d8"] = dict(
    ARM_CONFIGS["living_v5_4x_d4"],
    n_blocks=8,
)
ARM_TAPER["probe_v5_d8"] = ARM_TAPER["living_v5_4x_d4"]
ARM_FILELIST["probe_v5_d8"] = ARM_FILELIST["living_v5_4x_d4"]
ARM_SIGREG["probe_v5_d8"] = ARM_SIGREG["living_v5_4x_d4"]
ARM_COSINE["probe_v5_d8"] = ARM_COSINE["living_v5_4x_d4"]
# Delayed-kill twin of probe_v5_d8 (2026-08-06): byte-identical model
# config under a distinct arm name (never-pool discipline). The delta is
# observation-side only -- guard_min_step=1000 in ARM_GUARD_MIN_STEP.
# Extended-leash twin (2026-08-06, Brian's standing order): identical
# again, guard_min_step 5000, intended run length 6000 steps.
# Warmup twin (2026-08-06): probe_v5_d8 byte-identical in model config;
# the delta is schedule-side (ARM_LR_WARMUP) plus the guard hold above.
# Remedy-probe arms (2026-08-07): warmup-1000 base, model config =
# probe_v5_d8, plus interior_latent_blocks (0, 3, 6) for the wsig arms
# (block 0 is the measured collapse locus; 3 and 6 span the interior).
for _arm in ("probe_d8_tc", "probe_d8_orth", "probe_d8_tc_orth"):
    ARM_CONFIGS[_arm] = dict(ARM_CONFIGS["probe_v5_d8"])
for _arm in ("probe_d8_wsig", "probe_d8_tc_wsig", "probe_d8_wsig_orth"):
    ARM_CONFIGS[_arm] = dict(
        ARM_CONFIGS["probe_v5_d8"], interior_latent_blocks=(0, 3, 6),
    )
for _arm in ("probe_d8_wsig1", "probe_d8_wsig10", "probe_d8_tc_wsig10"):
    ARM_CONFIGS[_arm] = dict(
        ARM_CONFIGS["probe_v5_d8"], interior_latent_blocks=(0, 3, 6),
    )
for _arm in ("probe_d8_tc", "probe_d8_wsig", "probe_d8_orth",
             "probe_d8_tc_wsig", "probe_d8_tc_orth", "probe_d8_wsig_orth",
             "probe_d8_wsig1", "probe_d8_wsig10", "probe_d8_orth1",
             "probe_d8_tc_wsig10"):
    ARM_DEEP_CADENCE[_arm] = 100
    ARM_GUARD_MIN_STEP[_arm] = 1000
    ARM_TAPER[_arm] = ARM_TAPER["living_v5_4x_d4"]
    ARM_FILELIST[_arm] = ARM_FILELIST["living_v5_4x_d4"]
    ARM_SIGREG[_arm] = ARM_SIGREG["living_v5_4x_d4"]
    ARM_COSINE[_arm] = ARM_COSINE["living_v5_4x_d4"]

# ---------------------------------------------------------------------------
# Variance-budget governor (VBG), stage 45. Spec:
# docs/reviews/2026-08-07_variance-budget-governor-spec-for-opus.md
#
# Stage-31 base (warmup 1000, guard hold 1000, cadence 100, unclipped),
# interior_latent_blocks (0, 3, 6), marginal SIGReg untouched.
#
# DOSES ARE MEASURED, NOT ASSUMED -- scripts/calibrate_vbg.py, run against
# real checkpoints on 2026-08-07. They differ sharply from the spec's own
# arithmetic because the spec reads the floor's top-direction share as
# ~0.12; that figure is the wsig10 ARREST run (the successful
# intervention), not a floor. Measured:
#
#   probe_v5_d8_dk5000  seed46  share_sketch 0.790  (full 0.833)  floor
#   probe_v5_d8_warmup  seed97  share_sketch 0.667  (full 0.647)  floor
#   probe_d8_wsig10     seed46  share_sketch 0.093  (full 0.218)  arrest
#
# w_share: match the alpha=10 raw-scale contribution that arrested the
#   collapse. Backed out of the loss identity on probe_d8_wsig10 step 100
#   (l_wsig is unlogged -- see the runner deviation note):
#       451.11 = 3.738 + 0.2*1887.56 + 10*l_wsig  =>  10*l_wsig = 69.86
#   Trace-normalized penalty on floor checkpoints measures 44.7-51.1:
#       w_share = 69.86 / ~47 = 1.49  ->  1.5
#   (The same calc on the ARREST checkpoint gives 10.3, because wsig had
#   already pulled that state's scale toward unit and trace-normalization
#   has almost nothing left to remove there -- ratio raw/norm 1.17, vs
#   27-65 at the floor. The floor is the right anchor: it is the state the
#   governor actually has to act on.)
#
# w_cap: O(10) contribution at the measured floor share.
#       relu(0.79 - 0.05)^2 = 0.548;   10 / 0.548 = 18.2  ->  18
#   The spec's derivation from share 0.12 gives w_cap ~2041, which at the
#   measured floor share would contribute 2041 * 0.548 = 1118 -- two orders
#   above the entire loss late in the arrest run (~17). Flagged in the
#   return note rather than taken silently.
# ---------------------------------------------------------------------------
# LLM-JEPA (arXiv 2509.14252), stage 50. Spec:
# docs/reviews/2026-08-08_llm-jepa-integration-spec-for-opus.md
# Depth 8, muPC OFF, warmup 1000, guard hold 1000, cadence 100, unclipped.
#
# DOSE MEASURED, and it does not match the spec's estimate -- return note SA.
# Measured at init on a real gutenberg_100 batch with this exact config
# (scripts/calibrate_ntp.py):
#
#   L_NTP            10.47      (ln(32000) = 10.373 -- head is at chance)
#   l_pred           11.65
#   l_sigreg       7348.22      (x0.2 = 1469.64)
#   JEPA-side       1481.29      -- the spec estimated O(100-500)
#
# Loss-share dosing (the spec's stated target, NTP at 30-50% of total):
#   w_ntp = 60.6 (30%) / 94.3 (40%) / 141.4 (50%)   -- spec guessed 5-15
#
# Gradient-share on the SHARED TRUNK (LM head excluded), which is what
# actually steers optimization:
#   ||dL_NTP/dth|| 0.518   ||dL_JEPA/dth|| 466.27   ratio 900:1
#   w_ntp = 385.7 (NTP at 30% of gradient) / 899.9 (50%)
#
# Loss share and gradient share disagree by ~6x. Built against GRADIENT
# share at 30%: w_ntp = 400. This is the number I most expect to be
# overruled, and it is one dict entry.
ARM_W_NTP: dict[str, float] = {"probe_d8_llmjepa": 400.0,
                               "probe_d8_llmjepa6k": 400.0,
                               "probe_d8_llmjepa_v2": 3.0}
# Two-gauge execution rule (2026-08-08): NTP arms only. 8000 = a quarter
# of the 32k chance level -- generation this good vetoes a marginal NMSE
# trip; genuinely broken generation (ppl >= 8000) does not.
ARM_PPL_VETO: dict[str, float] = {"probe_d8_llmjepa": 8000.0,
                                  "probe_d8_llmjepa6k": 8000.0,
                                  "probe_d8_llmjepa_v2": 8000.0}
ARM_CONFIGS["probe_d8_llmjepa"] = dict(
    ARM_CONFIGS["probe_v5_d8"], mu_pc_enabled=False,
)
ARM_DEEP_CADENCE["probe_d8_llmjepa"] = 100
ARM_GUARD_MIN_STEP["probe_d8_llmjepa"] = 1000
ARM_LR_WARMUP["probe_d8_llmjepa"] = 1000
ARM_TAPER["probe_d8_llmjepa"] = ARM_TAPER["living_v5_4x_d4"]
ARM_FILELIST["probe_d8_llmjepa"] = ARM_FILELIST["living_v5_4x_d4"]
ARM_SIGREG["probe_d8_llmjepa"] = ARM_SIGREG["living_v5_4x_d4"]
ARM_COSINE["probe_d8_llmjepa"] = ARM_COSINE["living_v5_4x_d4"]
for _a in ("probe_d8_llmjepa6k",):
    ARM_DEEP_CADENCE[_a] = 100
    ARM_GUARD_MIN_STEP[_a] = 1000
    ARM_LR_WARMUP[_a] = 1000
    ARM_TAPER[_a] = ARM_TAPER["living_v5_4x_d4"]
    ARM_FILELIST[_a] = ARM_FILELIST["living_v5_4x_d4"]
    ARM_SIGREG[_a] = ARM_SIGREG["living_v5_4x_d4"]
    ARM_COSINE[_a] = ARM_COSINE["living_v5_4x_d4"]
ARM_CONFIGS["probe_d8_llmjepa_v2"] = dict(ARM_CONFIGS["probe_d8_llmjepa"])
for _spa_arm, _spa in (("probe_d8_v2_spa5", 1e-5), ("probe_d8_v2_spa4", 1e-4),
                       ("probe_d8_v2_spa3", 1e-3)):
    ARM_CONFIGS[_spa_arm] = dict(ARM_CONFIGS["probe_d8_llmjepa_v2"],
                                 set_point_adapt_rate=_spa)
    ARM_W_NTP[_spa_arm] = 3.0
    ARM_PPL_VETO[_spa_arm] = 8000.0
    ARM_DEEP_CADENCE[_spa_arm] = 100
    ARM_GUARD_MIN_STEP[_spa_arm] = 1000
    ARM_LR_WARMUP[_spa_arm] = 1000
    ARM_TAPER[_spa_arm] = ARM_TAPER["living_v5_4x_d4"]
    ARM_FILELIST[_spa_arm] = ARM_FILELIST["living_v5_4x_d4"]
    ARM_SIGREG[_spa_arm] = ARM_SIGREG["living_v5_4x_d4"]
    ARM_COSINE[_spa_arm] = ARM_COSINE["living_v5_4x_d4"]
# ---------------------------------------------------------------------------
# VISReg replacement arm (2026-08-11), stage 54.
#
# Doses are the PAPER'S, adopted as published (the week's costliest
# errors were re-derived dosings): convex lambda 0.6 (their small-
# dataset recommendation), component weights 1.0 each, K = 1024 (their
# K = C*D guidance with C > 1 at D = 512). All three VISReg terms are
# O(1) by construction -- none can run to the 10^3 our Epps-Pulley
# statistic hits at init -- so no gradient-share calibration pass is
# needed before first flight; the per-term logging (l_vis_*) is the
# check that this holds in practice.
#
# sigreg_projection="none" is REQUIRED by the loss module (trunk-latents
# ruling: the learnable head absorbs scale, measured 0.42-0.55 singular
# values, and would blind VISReg exactly as it blinded SIGReg).
# ARM_SIGREG is set to 0.0 for record honesty: SIGReg does not run in
# this arm, and a persisted 0.2 would imply a route that does not exist.
# NO ARM_W_NTP / ARM_PPL_VETO entries: NTP is out of v1 by ruling, and
# the ppl veto is meaningless without it -- the plain NMSE guard applies.
# ---------------------------------------------------------------------------
ARM_VISREG: dict[str, float] = {"probe_d8_visreg": 0.6}
ARM_VISREG_NUM_PROJ: dict[str, int] = {"probe_d8_visreg": 1024}
# Middle-ground kill-rule overrides (2026-08-11). Empty = every arm runs
# the ruled defaults; entries exist so a registered experiment can restore
# the legacy first-reading kill (persist 0) or disable a rule (0).
ARM_DIV_PERSIST: dict[str, int] = {}
ARM_DIV_CEIL_MULT: dict[str, float] = {}
ARM_DIV_RANK_VETO: dict[str, float] = {}

# ---------------------------------------------------------------------------
# 768x8 ruled-scale family (2026-08-11), stage 55. Spec + amendments:
# docs/research/2026-08-11_768x8-family-spec.md. Brian's rulings: width
# 768, heads 8x96, FULL corpus seen (one epoch of the 1051-file 113.0M-
# token list Opus built, bf1649a). K = 1536 (C=2, Opus's slice-coverage
# argument). n_heads=8 here also moves the JEPA predictor (it inherits
# encoder heads) -- recorded in Amendment 1, deliberately unpinned.
# Launch: --stage 55 --seeds 46,95,97 --epochs 1 (NO max-batches cap;
# cosine spans the epoch by construction). Batch from the two-run smoke;
# if 16 is forced, lambda_shape 2.0 rides as a registered
# batch-compensating dose (Amendment 1, section 3).
# ---------------------------------------------------------------------------
ARM_CONFIGS["probe_768_visreg"] = dict(
    ARM_CONFIGS["probe_v5_d8"], mu_pc_enabled=False, n_heads=8,
)
ARM_VISREG["probe_768_visreg"] = 0.6
ARM_VISREG_NUM_PROJ["probe_768_visreg"] = 1536
ARM_SIGREG_PROJ["probe_768_visreg"] = "none"
ARM_SIGREG["probe_768_visreg"] = 0.0
ARM_DEEP_CADENCE["probe_768_visreg"] = 100
ARM_GUARD_MIN_STEP["probe_768_visreg"] = 1000
ARM_LR_WARMUP["probe_768_visreg"] = 1000
ARM_TAPER["probe_768_visreg"] = ARM_TAPER["living_v5_4x_d4"]
ARM_COSINE["probe_768_visreg"] = ARM_COSINE["living_v5_4x_d4"]
ARM_FILELIST["probe_768_visreg"] = str(
    REPO_ROOT / "corpus_build" / "gutenberg_768_filelist.txt"
)
ARM_CONFIGS["probe_d8_visreg"] = dict(
    ARM_CONFIGS["probe_v5_d8"], mu_pc_enabled=False,
)
ARM_SIGREG_PROJ["probe_d8_visreg"] = "none"
ARM_SIGREG["probe_d8_visreg"] = 0.0
ARM_DEEP_CADENCE["probe_d8_visreg"] = 100
ARM_GUARD_MIN_STEP["probe_d8_visreg"] = 1000
ARM_LR_WARMUP["probe_d8_visreg"] = 1000
ARM_TAPER["probe_d8_visreg"] = ARM_TAPER["living_v5_4x_d4"]
ARM_FILELIST["probe_d8_visreg"] = ARM_FILELIST["living_v5_4x_d4"]
ARM_COSINE["probe_d8_visreg"] = ARM_COSINE["living_v5_4x_d4"]

for _a in ("probe_d8_llmjepa_v2",):
    ARM_DEEP_CADENCE[_a] = 100
    ARM_GUARD_MIN_STEP[_a] = 1000
    ARM_LR_WARMUP[_a] = 1000
    ARM_TAPER[_a] = ARM_TAPER["living_v5_4x_d4"]
    ARM_FILELIST[_a] = ARM_FILELIST["living_v5_4x_d4"]
    ARM_SIGREG[_a] = ARM_SIGREG["living_v5_4x_d4"]
    ARM_COSINE[_a] = ARM_COSINE["living_v5_4x_d4"]

for _a in ("probe_d8_w768", "probe_d4_c100"):
    ARM_DEEP_CADENCE[_a] = 100
    ARM_GUARD_MIN_STEP[_a] = 1000
    ARM_LR_WARMUP[_a] = 1000
    ARM_TAPER[_a] = ARM_TAPER["living_v5_4x_d4"]
    ARM_FILELIST[_a] = ARM_FILELIST["living_v5_4x_d4"]
    ARM_SIGREG[_a] = ARM_SIGREG["living_v5_4x_d4"]
    ARM_COSINE[_a] = ARM_COSINE["living_v5_4x_d4"]
# Surgery twin (2026-08-07): identical model config; the intervention
# lives in the pre-seeded checkpoint, not the config.
# +50% ramp twin (2026-08-06, Brian's call): identical model config,
# ramp 1500 via ARM_LR_WARMUP, guard hold 1500 above.
# `probe_surprise_d8_nomupc` is `probe_surprise_d8` with mu_pc_enabled False and
# NOTHING else changed -- same 8 blocks, same clip of 1000, same "linear"
# projection (the "none" variant was refuted 2026-07-30 and made prediction
# 7.3x worse, so it is not carried forward).
#
# Disabling muPC changes two things at once, deliberately: residual_scale goes
# 1/(8**0.25) = 0.5946 -> 1.0, and the depth-scaled init is skipped. They are
# tested together because a clean result implicates muPC as a whole and a null
# result exonerates it as a whole; separating them costs a second run and only
# matters if the first one is positive. Init has already been shown to wash out
# by step 3000 (block-0 q_proj std 0.0322 at d4 vs 0.0325 at d8), so the
# residual scale is the live half.
# Scheduled-muPC arm config (2026-08-07): must follow the nomupc arm it
# copies (the earlier misplaced assignment KeyError'd at import).


def _param_groups(loss_module, model, arm: str, base_lr: float):
    """Optimizer param groups, with optional backprop-LR compensation.

    Every block computes `x = x + residual_scale * f(x)`, so by the chain rule
    every BACKPROP-trained parameter inside a block receives a gradient scaled
    by `residual_scale`. Parameters outside the blocks -- the embeddings, the
    predictor, the projection heads -- do not: `x0` reaches the output through
    the unattenuated skip path.

    So muPC quietly applies a smaller effective learning rate to the trunk than
    to everything around it, and the deeper the model the wider that gap.

    Why this matters here (measured 2026-07-31): block 0's job is to strip the
    ~0.58 shared component that real text carries into the trunk. With muPC off
    its attention learns an output that OPPOSES that component (delta -0.316).
    Under attenuation it learns one that REINFORCES it (+0.252). Stage 22 showed
    rescaling the input cannot fix this -- it changed the magnitude by the
    predicted 1.68x and never changed the sign. What differs is what attention
    LEARNED, and what it learns is shaped by the gradient it receives.

    `ARM_BACKPROP_LR_COMPENSATE` multiplies the block parameters' learning rate
    by `1 / residual_scale`, restoring parity with the un-attenuated parameters.
    This is the exact counterpart of `mu_pc_rate_power`, which did the same for
    the PC side and produced the best configuration found -- applied to the side
    that actually owns block 0's canceller.

    Returns a single group when compensation is off, so the optimizer state is
    byte-identical to every prior run.
    """
    trainable = [p for p in loss_module.parameters() if p.requires_grad]
    n_comp = int(ARM_BACKPROP_LR_COMPENSATE.get(arm, 0))
    if n_comp == 0:
        return trainable

    rs = float(getattr(model.blocks[0], "residual_scale", 1.0))
    if rs >= 1.0:
        return trainable

    # SCOPE (2026-07-31). Stage 23 compensated ALL 64 block tensors and the
    # trunk diverged (NMSE 556.77, killed at 12 min) -- while confirming the
    # mechanism: block 0's attention delta crossed zero to -0.1156, the first
    # negative value in any muPC-on run.
    #
    # So the compensation is correct in kind and was far too broad in scope. The
    # offset needs stripping ONCE, in block 0; the deeper blocks appear to need
    # the attenuation for stability, since that is what came apart. n_comp is
    # the number of LEADING blocks to compensate; -1 means all (stage 23's
    # behaviour, retained so the refuted setting stays reproducible).
    targets = (list(model.blocks) if n_comp < 0
               else list(model.blocks)[:n_comp])
    block_ids = {id(p) for b in targets for p in b.parameters()
                 if p.requires_grad}
    in_blocks = [p for p in trainable if id(p) in block_ids]
    rest = [p for p in trainable if id(p) not in block_ids]
    boosted = base_lr / rs
    print(f"  [backprop-lr] compensating {len(targets)} of {len(model.blocks)} "
          f"blocks: {len(in_blocks)} tensors at lr={boosted:.3e} "
          f"(1/{rs:.4f} = {1/rs:.3f}x), {len(rest)} others at lr={base_lr:.3e}")
    return [
        {"params": in_blocks, "lr": boosted},
        {"params": rest, "lr": base_lr},
    ]


def _device() -> torch.device:
    """The accelerator, or a LOUD failure.

    Hardened 2026-08-13 (Opus 5 audit). The previous version returned CPU
    silently whenever ``import torch_directml`` raised, which is the
    forbidden shape from CLAUDE.md: "Missing resources fail loudly, never
    silently fall back." Two ways it bites, both silent:

    - The repo's own ``.venv`` is a uv-managed CPython 3.14 carrying
      torch 2.10.0+cpu and NO torch_directml, while the training
      environment is ``C:\\Program Files\\Python310\\python.exe`` with
      torch 2.4.1 + DirectML. Anything launched from the obvious-looking
      in-repo venv took the fallback and reported nothing.
    - `pilot_verdict.py` carries determinism guards written specifically
      for DirectML warm-up numerics. Recomputing a verdict on CPU
      produces different numbers with no marker saying so.

    Set ``LUTHI_ALLOW_CPU=1`` to opt in deliberately; the fallback then
    still announces itself on stderr, because a run whose device is a
    surprise is a run whose numbers are a surprise.
    """
    try:
        import torch_directml
        return torch_directml.device()
    except ImportError:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    msg = (
        "No accelerator: torch_directml is not importable and CUDA is "
        f"unavailable (python={sys.executable}, torch={torch.__version__}). "
        "The training environment for this repo is Python 3.10 with "
        "torch 2.4.1 + torch-directml; the in-repo .venv is NOT it. "
        "Set LUTHI_ALLOW_CPU=1 to run on CPU deliberately."
    )
    if os.environ.get("LUTHI_ALLOW_CPU") != "1":
        raise RuntimeError(msg)
    print(f"  [device] WARNING -- running on CPU by LUTHI_ALLOW_CPU=1. {msg}",
          file=sys.stderr, flush=True)
    return torch.device("cpu")


class _DeviceLoader:
    """Wraps a MultimodalDataLoader; delivers device-local batches."""

    def __init__(self, inner, device: torch.device):
        self._inner = inner
        self._device = device

    def _move(self, batch: dict) -> dict:
        return {k: v.to(self._device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

    def next_batch(self, modality: str) -> dict:
        return self._move(self._inner.next_batch(modality))

    def batch_token_count(self, modality: str, batch: dict) -> int:
        return self._inner.batch_token_count(modality, batch)

    def state_dict(self) -> dict:
        return self._inner.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self._inner.load_state_dict(state)

    def holdout_batch_count(self, modality: str, batch_size: int) -> int:
        return self._inner.holdout_batch_count(modality, batch_size)

    def holdout_batches(self, modality: str, batch_size: int):
        for batch in self._inner.holdout_batches(modality, batch_size):
            yield self._move(batch)


def _run_name(arm: str, d_model: int, seed: int) -> str:
    return f"{arm}_{d_model}d_seed{seed}"


def _result_path(arm: str, d_model: int, seed: int) -> Path:
    return OUTPUT_ROOT / _run_name(arm, d_model, seed) / "pilot_result.json"


def _run_one(arm: str, d_model: int, seed: int, args) -> dict:
    from luthi.tokenizer import BPETokenizer
    from luthi.v2.eval_heldout import (
        fit_next_token_probe,
        heldout_latent_prediction,
        probe_accuracy,
    )
    from luthi.v2.jepa_loss import JEPALoss, SIGREG_LAMBD
    from luthi.v2.jepa_runner import (
        CheckpointConfig,
        EpochConfig,
        JEPATrainer,
        KillCriteriaConfig,
        LoggingConfig,
        LRScheduleConfig,
        ModalitySampler,
        RunnerConfig,
        SamplerConfig,
        TaperConfig,
    )
    from luthi.v2.multimodal_data import (
        MultimodalDataLoaderImpl,
        TextDataset,
        TextDatasetConfig,
    )
    from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM

    device = _device()
    run_dir = OUTPUT_ROOT / _run_name(arm, d_model, seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)

    # Data source: per-arm filelist (the 4x superset) when registered,
    # else the default 1x directory. The 4x corpus's holdout is its own
    # tail (a DIFFERENT test set than the 1x tail) -- cross-corpus
    # comparisons are directional only; the registry carries the caveat.
    if arm in ARM_FILELIST:
        filelist = Path(ARM_FILELIST[arm])
        source_paths = [
            line.strip() for line in
            filelist.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        source_paths = [args.data_dir]
    text_ds = TextDataset(TextDatasetConfig(
        source_paths=source_paths,
        tokenizer_path=Path(args.tokenizer),
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        stride=args.stride,
        base_seed=seed,
        holdout_fraction=args.holdout_fraction,
    ))
    loader = _DeviceLoader(MultimodalDataLoaderImpl(text=text_ds), device)

    # Model kwargs: driver defaults, overridden by the arm's declared
    # config (ARM_CONFIGS is the single source of truth, shared with the
    # verdict rebuild). The depth arm carries its own n_blocks/mu_pc_*
    # through this merge; arms without a key keep the smoke default.
    model_kwargs = dict(
        vocab_size=text_ds.vocab_size(),
        d_model=d_model,
        n_blocks=args.n_blocks,
        n_heads=4,
        ffn_expansion=1,
        max_seq_len=args.seq_len,
        backward_pass_enabled=False,
    )
    model_kwargs.update(ARM_CONFIGS[arm])
    # Silent-fallback guard (2026-07-21): the C++ extension rebuilds
    # between families (the running trainer locks the .pyd, so source
    # changes compile at the NEXT process start). If that rebuild fails,
    # pc_ops silently falls back to pure Python -- ~50x slower, which
    # would turn a 9h seed into weeks, unattended. Refuse loudly instead.
    if model_kwargs.get("relative_trust"):
        from luthi.v2.pc_ops import is_cpp_available
        if not is_cpp_available():
            raise SystemExit(
                f"[jepa-pilot] {arm}: relative_trust requires the C++ "
                "pc_ops extension, which failed to load in this process. "
                "Fix the build (python -c \"from luthi.v2 import pc_ops; "
                "print(pc_ops.is_cpp_available())\") before running this arm."
            )
    model = MultimodalPredictiveCodingLM(**model_kwargs).to(device)
    loss_module = JEPALoss(
        online_encoder=model,
        sigreg_lambd=ARM_SIGREG.get(arm, SIGREG_LAMBD),
        sigreg_projection=ARM_SIGREG_PROJ.get(arm, "linear"),
        interior_sigreg_alpha=ARM_WSIG_ALPHA.get(arm, 0.0),
        w_ntp=ARM_W_NTP.get(arm, 0.0),
        visreg_lambda=ARM_VISREG.get(arm, 0.0),
        visreg_num_proj=ARM_VISREG_NUM_PROJ.get(arm, 1024),
    ).to(device)
    # Provenance-consistency contract (2026-08-08): the stage-50 family
    # ran with w_ntp=400 in pilot_result.json while the loss module held
    # the default 0.0 -- the ctor call was missing the argument. The
    # persisted record and the live module MUST agree, or the record is
    # a lie. Assert every dual-sourced dose.
    _prov = {
        "w_ntp": ARM_W_NTP.get(arm, 0.0),
        "interior_sigreg_alpha": ARM_WSIG_ALPHA.get(arm, 0.0),
        "visreg_lambda": ARM_VISREG.get(arm, 0.0),
    }
    for _k, _v in _prov.items():
        _live = float(getattr(loss_module, _k, float("nan")))
        if abs(_live - float(_v)) > 1e-9:
            raise RuntimeError(
                f"provenance mismatch: {_k} persisted as {_v} but the live "
                f"loss module holds {_live} -- the record would lie."
            )

    # Cosine LR needs the planned run length. tokens_per_pass is
    # n_sequences * seq_len (exact), so steps/epoch falls out directly;
    # epoch-boundary partial batches can shift the true total by a step
    # or two, which cosine absorbs (progress past 1.0 holds the floor).
    steps_per_epoch = math.ceil(
        text_ds.tokens_per_pass() / args.seq_len / args.batch_size
    )
    lr_schedule = LRScheduleConfig(
        enabled=ARM_COSINE.get(arm, False),
        min_lr_ratio=0.1,
        total_steps=args.epochs * steps_per_epoch,
        warmup_steps=ARM_LR_WARMUP.get(arm, 0),
    )

    sampler_cfg = SamplerConfig(
        corpus_sizes_tokens={"text": text_ds.tokens_per_pass()}, alpha=0.7,
    )
    gen = torch.Generator(device="cpu").manual_seed(seed)
    trainer = JEPATrainer(
        loss_module=loss_module,
        optimizer=optim.AdamW(
            _param_groups(loss_module, model, arm, args.lr),
            lr=args.lr,
        ),
        sampler=ModalitySampler(sampler_cfg, generator=gen),
        data_loader=loader,
        config=RunnerConfig(
            sampler=sampler_cfg,
            checkpoint=CheckpointConfig(
                interval_seconds=args.checkpoint_interval,
                rolling_slots=args.checkpoint_slots,
            ),
            logging=LoggingConfig(
                heldout_eval_batches=args.heldout_batches,
                deep_interval_batches=ARM_DEEP_CADENCE.get(arm, 1000),
            ),
            kill_criteria=KillCriteriaConfig(
                warmup_batches=args.kill_warmup,
                # Pilot-derived thresholds (calibration pass 1, 2026-07-16
                # -- docs/research/2026-07-16_jepa-pilot-calibration-pass.md).
                # The static defaults killed 10/10 healthy runs: kill-1's
                # init-window baseline fired while effective rank was
                # RISING; kill-6's 25% band around a transiently-latched
                # running min fired at the substrate's healthiest moment.
                stationary_deviation_pct=0.85,
                substrate_health_degradation_pct=1.0,
                trending_smoothing_window=9,
                substrate_health_window=10,
            ),
            epoch=EpochConfig(
                max_epochs=args.epochs,
                abort_continue_at_epoch_1=False,  # sweep runs are unattended
                max_batches_per_epoch=args.max_batches_per_epoch,
            ),
            taper=TaperConfig(enabled=ARM_TAPER.get(arm, False)),
            lr_schedule=lr_schedule,
            grad_clip_norm=ARM_GRAD_CLIP.get(arm, 0.0),
            guard_min_step=ARM_GUARD_MIN_STEP.get(arm, 0),
            divergence_ppl_veto=ARM_PPL_VETO.get(arm, 0.0),
            # Middle-ground kill rules (2026-08-11, Brian's ruling). The
            # per-arm dicts are the toggle; absent entries take the ruled
            # defaults (persist 500 / ceiling 10x / rank veto at eff 100).
            divergence_persist_steps=ARM_DIV_PERSIST.get(arm, 500),
            divergence_hard_ceiling_mult=ARM_DIV_CEIL_MULT.get(arm, 10.0),
            # Width-relative default (Opus's §2 measurement, 2026-08-11):
            # healthy eff scales ~ D^0.75, not linearly. 100 at 512 -> 135
            # at 768. Linear would be ~10% stiffer than what the 512
            # family passed.
            divergence_rank_veto_min_eff=ARM_DIV_RANK_VETO.get(
                arm, float(round(100.0 * (d_model / 512.0) ** 0.75))
            ),
        ),
        run_dir=run_dir,
    )

    started = time.time()
    # Mid-seed resilience (2026-07-20, Brian's terminal-close question):
    # an interrupted seed continues from its latest rolling checkpoint
    # (<=15 min lost) instead of restarting from zero. resume_from_latest
    # falls back through older slots if the newest is a partial write.
    ckpt_dir = run_dir / "checkpoints"
    if any(ckpt_dir.glob("ckpt_*.pt")):
        loaded = trainer.resume_from_latest()
        print(f"[jepa-pilot] {_run_name(arm, d_model, seed)}: resumed "
              f"mid-seed from {loaded.name} (step {trainer.global_step})")
    outcome = trainer.run()

    heldout = heldout_latent_prediction(
        loss_module,
        loader.holdout_batches("text", 8),
        "text",
        max_batches=args.heldout_batches,
    )
    train_batches = [loader.next_batch("text") for _ in range(args.probe_batches)]
    probe = fit_next_token_probe(
        loss_module, train_batches,
        vocab_size=text_ds.vocab_size(),
        max_batches=args.probe_batches,
    )
    heldout_list = list(loader.holdout_batches("text", 8))
    probe_real = probe_accuracy(loss_module, probe, heldout_list)
    probe_floor = probe_accuracy(
        loss_module, probe, heldout_list, shuffled_label_floor=True,
    )

    result = {
        "arm": arm,
        "d_model": d_model,
        "seed": seed,
        "outcome": outcome,
        # Collapse-admissibility (protocol section 1): a killed run's
        # numbers are reported but flagged inadmissible for comparison.
        "admissible": outcome == "completed",
        "heldout": heldout,
        "probe": probe_real,
        "probe_shuffled_floor": probe_floor,
        "wall_clock_seconds": time.time() - started,
        "config": {
            # Effective model shape (the depth arm overrides n_blocks
            # via ARM_CONFIGS -- record what actually trained).
            "n_blocks": model_kwargs["n_blocks"], "epochs": args.epochs,
            "batch_size": args.batch_size, "seq_len": args.seq_len,
            "stride": args.stride, "lr": args.lr,
            "holdout_fraction": args.holdout_fraction,
            "data_dir": str(args.data_dir),
            # Honest data provenance: the verdict rebuild must reload
            # the SAME corpus (the holdout is corpus-derived).
            "file_list": ARM_FILELIST.get(arm),
            # v4 bundle provenance (2026-07-20): loss/trainer-side
            # settings that don't ride in ARM_CONFIGS model kwargs.
            "sigreg_lambd": ARM_SIGREG.get(arm, SIGREG_LAMBD),
            "cosine_lr": ARM_COSINE.get(arm, False),
            "lr_total_steps": lr_schedule.total_steps,
            # Full mechanism provenance (2026-08-05, Opus's brief §0.5):
            # which mechanisms were active in a run used to live only in
            # the arm NAME and this file's edit history -- an attribution
            # gap an ablation ladder cannot afford. Record the complete
            # merged model kwargs and the remaining trainer-side settings
            # so each pilot_result.json is self-describing.
            "model_kwargs": {k: v for k, v in model_kwargs.items()
                             if isinstance(v, (bool, int, float, str))},
            # Execution provenance (2026-08-13, Opus 5 audit). The record
            # described the model and every mechanism but never the
            # machine, so a run that took the old silent CPU fallback --
            # different backend, different numerics, ~100x slower -- wrote
            # a result file indistinguishable from a DirectML run. Same
            # class as the w_ntp note above, one layer further out again.
            "device": str(device),
            "torch_version": torch.__version__,
            "python": sys.version.split()[0],
            "grad_clip_norm": ARM_GRAD_CLIP.get(arm, 0.0),
            "taper": ARM_TAPER.get(arm, False),
            "deep_interval_batches": ARM_DEEP_CADENCE.get(arm, 1000),
            "guard_min_step": ARM_GUARD_MIN_STEP.get(arm, 0),
            "lr_warmup_steps": ARM_LR_WARMUP.get(arm, 0),
            "divergence_ppl_veto": ARM_PPL_VETO.get(arm, 0.0),
            "interior_sigreg_alpha": ARM_WSIG_ALPHA.get(arm, 0.0),
            # Restored 2026-08-11: the 08-10 prune dropped this entry from
            # the result config while w_ntp remained a live mechanism (the
            # v2 family runs at 3.0). The provenance assert above compares
            # the ARM dict to the live module, so it could not catch the
            # persisted FILE going silent -- exactly the class the assert
            # exists for, one layer further out.
            "w_ntp": ARM_W_NTP.get(arm, 0.0),
            # VISReg replacement provenance (2026-08-11). When lambda > 0
            # the sigreg_lambd above is 0.0 by arm construction and SIGReg
            # did not run; the convex mix replaces the additive form.
            "visreg_lambda": ARM_VISREG.get(arm, 0.0),
            "visreg_num_proj": ARM_VISREG_NUM_PROJ.get(arm, 1024),
            "sigreg_projection": ARM_SIGREG_PROJ.get(arm, "linear"),
            # Middle-ground kill rules (2026-08-11) -- provenance for
            # which guard contract judged this run.
            "divergence_persist_steps": ARM_DIV_PERSIST.get(arm, 500),
            "divergence_hard_ceiling_mult": ARM_DIV_CEIL_MULT.get(arm, 10.0),
            "divergence_rank_veto_min_eff": ARM_DIV_RANK_VETO.get(
                arm, float(round(100.0 * (d_model / 512.0) ** 0.75))
            ),
        },
    }
    _result_path(arm, d_model, seed).write_text(json.dumps(result, indent=2))
    return result


def run(stages: list[int], args) -> int:
    # Explicit --seeds overrides the prefix-of-SEEDS selection (added for
    # the stage-11 seed44 rerun, which must run seed 44 alone).
    run_seeds = (
        [int(x) for x in args.seeds.split(",")]
        if args.seeds else list(SEEDS[: args.n_seeds])
    )
    plan = [
        (arm, d, s)
        for stage in stages
        for arm, d in STAGES[stage]
        for s in run_seeds
    ]
    done = [c for c in plan if _result_path(*c).exists()]
    todo = [c for c in plan if not _result_path(*c).exists()]
    print(f"[jepa-pilot] plan: {len(plan)} runs "
          f"({len(done)} complete, {len(todo)} to run)")
    for arm, d_model, seed in todo:
        name = _run_name(arm, d_model, seed)
        if args.dry_run:
            print(f"  DRY-RUN: {name} (n_blocks={args.n_blocks}, "
                  f"epochs={args.epochs}, data={args.data_dir})")
            continue
        print(f"[jepa-pilot] starting {name}")
        try:
            result = _run_one(arm, d_model, seed, args)
        except Exception as e:  # noqa: BLE001 -- stop the queue loudly
            print(f"[jepa-pilot] FAILED: {name}: {type(e).__name__}: {e}")
            raise
        print(f"[jepa-pilot] {name}: outcome={result['outcome']} "
              f"heldout_l_pred={result['heldout']['l_pred_mean']:.6f} "
              f"probe_top1={result['probe']['top1']:.4f} "
              f"({result['wall_clock_seconds']/3600:.2f}h)")
    return 0


def aggregate() -> int:
    """Per-condition summary. Prints the ingredients; the curve-level
    verdict (effective-capacity placement) is the analysis doc's job."""
    conditions: dict[str, list[dict]] = {}
    for path in sorted(OUTPUT_ROOT.glob("*/pilot_result.json")):
        r = json.loads(path.read_text())
        conditions.setdefault(f"{r['arm']}_{r['d_model']}d", []).append(r)
    if not conditions:
        print("[aggregate] no completed runs")
        return 1
    summary = {}
    hdr = f"{'condition':<14} {'n':>2} {'adm':>3} {'l_pred mean':>12} {'std':>9} {'probe_top1':>10}"
    print(hdr)
    for cond, runs in sorted(conditions.items()):
        adm = [r for r in runs if r["admissible"]]
        vals = [r["heldout"]["l_pred_mean"] for r in adm]
        probes = [r["probe"]["top1"] for r in adm]
        mean = statistics.mean(vals) if vals else float("nan")
        std = statistics.stdev(vals) if len(vals) > 1 else float("nan")
        p1 = statistics.mean(probes) if probes else float("nan")
        summary[cond] = {
            "n_total": len(runs), "n_admissible": len(adm),
            "l_pred_mean": mean, "l_pred_std": std,
            "probe_top1_mean": p1,
            "inadmissible": [
                {"seed": r["seed"], "outcome": r["outcome"]}
                for r in runs if not r["admissible"]
            ],
        }
        print(f"{cond:<14} {len(runs):>2} {len(adm):>3} {mean:>12.6f} "
              f"{std:>9.6f} {p1:>10.4f}")
        if len(adm) < len(runs):
            print(f"  NOTE: {len(runs)-len(adm)} inadmissible run(s) "
                  f"(killed) -- reported, excluded from means")
    (OUTPUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[aggregate] written to {OUTPUT_ROOT / 'summary.json'}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stage", type=str, default=None, help="1, 2, 3, or 'all'")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--aggregate", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny CPU shakeout: 64d/1-block, capped steps.")
    p.add_argument("--data_dir", type=str,
                   default=str(REPO_ROOT / "corpus_build" / "gutenberg_100"))
    p.add_argument("--tokenizer", type=str,
                   default=str(REPO_ROOT / "corpus_build" / "tokenizer_32k.json"))
    p.add_argument("--n-blocks", dest="n_blocks", type=int, default=2)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--stride", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--holdout-fraction", dest="holdout_fraction",
                   type=float, default=0.02)
    p.add_argument("--heldout-batches", dest="heldout_batches",
                   type=int, default=50)
    p.add_argument("--probe-batches", dest="probe_batches",
                   type=int, default=32)
    p.add_argument("--kill-warmup", dest="kill_warmup",
                   type=int, default=5000)
    # Checkpoint retention. Defaults preserve the previous behaviour exactly
    # (900s / 3 slots) so nothing already registered changes.
    #
    # For a long run those defaults are a trap: at 1.43 GB per checkpoint and
    # 900s spacing, an 18-hour run writes 72 checkpoints and keeps the last 3
    # -- the final 45 minutes. Every diagnostic that found the 2026-07-29/30
    # depth-8 defect came off checkpoints (per-block offset dominance, the
    # update_ema trajectory, the input-sensitivity test), and all of them
    # compared EARLY against LATE. Spread matters more than density; hourly
    # coverage of the whole run beats 15-minute coverage of the last 5% of it.
    #
    # Suggested for the 18h depth-8 run: --checkpoint-interval 3600
    # --checkpoint-slots 20 (~29 GB, full-run hourly coverage).
    p.add_argument("--checkpoint-interval", dest="checkpoint_interval",
                   type=int, default=900)
    p.add_argument("--checkpoint-slots", dest="checkpoint_slots",
                   type=int, default=3)
    p.add_argument("--max-batches-per-epoch", dest="max_batches_per_epoch",
                   type=int, default=-1)
    p.add_argument("--n-seeds", dest="n_seeds", type=int, default=len(SEEDS))
    p.add_argument("--seeds", type=str, default=None,
                   help="Explicit comma-separated seed list (overrides "
                        "--n-seeds), e.g. '44'.")
    args = p.parse_args()

    if args.smoke:
        args.epochs = 1
        args.max_batches_per_epoch = 20
        args.heldout_batches = 3
        args.probe_batches = 4
        args.n_seeds = 1
        args.kill_warmup = 10**9
        args.batch_size = 4
        args.seq_len = 64
        # Smoke shrinks the conditions too: one living + one dead at 64d.
        STAGES[1] = [("living", 64), ("dead", 64)]
        print("[jepa-pilot] SMOKE MODE: 64d, 20 steps, 1 seed, CPU-ok")

    if args.aggregate:
        return aggregate()
    if args.stage is None:
        p.error("--stage required unless --aggregate")
    stages = [1, 2, 3] if args.stage == "all" else [int(args.stage)]
    return run(stages, args)


if __name__ == "__main__":
    raise SystemExit(main())
