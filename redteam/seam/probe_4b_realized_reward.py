"""Adversarial probes for Phase 4b realized-reward grounding (#3 review).

Run:  python -m redteam.seam.probe_4b_realized_reward

Four attacks on M9Trainer.observe_transition's realized-reward path + the
K-M9-3 safeguard, per 4.7's hand-back (2026-06-23):

  A  no-grad robustness — does the stop-grad invariant survive a careless
     caller (no torch.no_grad() wrapper, requires_grad inputs)?
  B1 reward scale sanity — on NORMAL bounded transitions, is r_realized O(1)
     and V bounded well under the 1e3 ceiling (no false-positive kill)?
  B2 ceiling fires — when V is driven sustained over 1e3, does K-M9-3 fire
     within value_ceiling_sustained cycles (the safety actually works)?
  C  deadly-triad oscillation — under rapidly-alternating transitions, does V
     oscillate/diverge, or does the slow Polyak target keep it stable?

confirmed=True means the concern is REAL (attack landed).
"""

from __future__ import annotations

import tempfile
import traceback
from pathlib import Path

import torch

from redteam.m9_step1._common import Verdict, report
from redteam.seam.probe_seam_review import _build_trainer

D = 32  # matches probe_seam_review._build_trainer's d_model


def _efe_module_params(trainer):
    mods = [trainer.efe.decoders, trainer.efe.preferences,
            trainer.efe.delta_s_module]
    for m in mods:
        for p in m.parameters():
            yield p


def _value_kill_fired(trainer) -> tuple[bool, object]:
    fired = trainer.m9_kills.fired()
    hit = any(("value" in str(k).lower()) or ("m9_3" in str(k).lower())
              or str(k).endswith("3") for k in fired)
    return hit, fired


# ---------------------------------------------------------------------------
# A — no-grad robustness
# ---------------------------------------------------------------------------
def probe_a() -> list[Verdict]:
    v = []
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        trainer = _build_trainer(Path(tmp))
        try:
            for p in _efe_module_params(trainer):
                p.grad = None
            s_t = torch.randn(1, D, requires_grad=True)
            a_t = torch.randn(1, D, requires_grad=True)
            s_next = torch.randn(1, D, requires_grad=True)
            # Careless caller: NO torch.no_grad() wrapper, grad-tracked inputs.
            out = trainer.efe.compute_g_realized(s_t, a_t, s_next)
            # Replicate observe_transition's downstream: r = -G feeds a TD loss.
            r = -out["G"]
            v_pred = trainer.v_head(s_t.detach())
            loss = (v_pred - r).pow(2).mean()
            loss.backward()
            leaked = [p for p in _efe_module_params(trainer) if p.grad is not None]
            g_requires = bool(out["G"].requires_grad)
            confirmed = bool(leaked) or g_requires
            v.append(Verdict(
                "A: compute_g_realized leaks gradient to EFE modules when the "
                "caller omits torch.no_grad()",
                confirmed,
                f"returned G.requires_grad={g_requires}; EFE-module params with "
                f".grad after a TD backward = {len(leaked)}. "
                + ("LEAK — the invariant depends on the caller." if confirmed else
                   "No leak: the .detach() in the return holds even without the "
                   "caller's no_grad(). Robust (double-protected). RECO: still "
                   "decorate compute_g_realized @torch.no_grad() so it's "
                   "self-protecting AND skips building a throwaway graph — "
                   "cheap defense-in-depth, since today only the return-detach "
                   "stands between a future de-detached refactor and a leak.")))
        except Exception:
            v.append(Verdict("A: probe raised", False, traceback.format_exc()))
    return v


# ---------------------------------------------------------------------------
# B1 / B2 — reward scale + ceiling
# ---------------------------------------------------------------------------
def probe_b() -> list[Verdict]:
    v = []
    # B1: normal bounded transitions
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        trainer = _build_trainer(Path(tmp))
        try:
            torch.manual_seed(5)
            rs, vs = [], []
            for _ in range(40):
                s_t = torch.randn(1, D)
                a_t = torch.randn(1, D)
                s_next = torch.randn(1, D)
                m = trainer.observe_transition(s_t, a_t, s_next, ctx={})
                rs.append(m["r_realized"])
                vs.append(float(trainer.v_head(s_t).mean().item()))
            fired, fset = _value_kill_fired(trainer)
            r_absmax = max(abs(x) for x in rs)
            v_absmax = max(abs(x) for x in vs)
            # concern if normal ops produce a false-positive kill OR V approaches
            # the ceiling (>10% of 1e3) on bounded inputs.
            confirmed = bool(fired or v_absmax > 100.0)
            v.append(Verdict(
                "B1: normal bounded transitions push V toward the 1e3 ceiling "
                "or false-trip K-M9-3",
                confirmed,
                f"over 40 normal cycles: |r_realized| max = {r_absmax:.3f} "
                f"(4.7 claimed O(1)), |V| max = {v_absmax:.3f}, ceiling = 1e3; "
                f"value-kill fired = {fired} ({fset}). "
                + ("FALSE-POSITIVE risk / V too hot on normal input." if confirmed
                   else "r_realized is O(1) as claimed, V bounded well under "
                        "ceiling, no false kill. 4.7's scale analysis holds at "
                        "this size.")))
        except Exception:
            v.append(Verdict("B1: probe raised", False, traceback.format_exc()))

    # B2: drive V sustained over the ceiling; the kill must fire.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        trainer = _build_trainer(Path(tmp))
        try:
            big = 1.0e4  # scale s_t so v_head(s_t) comfortably exceeds 1e3 every cycle
            vseen, first_fire = [], None
            for i in range(20):
                s_t = torch.randn(1, D) * big
                a_t = torch.randn(1, D)
                s_next = torch.randn(1, D)
                trainer.observe_transition(s_t, a_t, s_next, ctx={})
                vseen.append(float(trainer.v_head(s_t).mean().item()))
                fired, _ = _value_kill_fired(trainer)
                if fired and first_fire is None:
                    first_fire = i + 1
            v_absmax = max(abs(x) for x in vseen)
            fired, fset = _value_kill_fired(trainer)
            # concern (kill is BROKEN) if V was sustained over ceiling but kill
            # never fired.
            over_ceiling = sum(1 for x in vseen if abs(x) >= 1e3)
            confirmed = bool(over_ceiling >= 8 and not fired)
            v.append(Verdict(
                "B2: K-M9-3 fails to fire when V is sustained over the 1e3 "
                "ceiling (safety hole)",
                confirmed,
                f"|V| max = {v_absmax:.1f}, cycles with |V|>=1e3 = {over_ceiling}, "
                f"value-kill fired = {fired} (first at cycle {first_fire}); "
                f"sustained threshold = {trainer.m9_kills.value_ceiling_sustained}. "
                + ("SAFETY HOLE — V diverged past the ceiling and the kill stayed "
                   "silent." if confirmed else
                   ("kill fired correctly once V was sustained over the ceiling — "
                    "the seam path's V is visible to K-M9-3 and the ceiling works."
                    if fired else
                    "V did not stay over the ceiling long enough to test firing "
                    "(inconclusive — increase `big`)."))))
        except Exception:
            v.append(Verdict("B2: probe raised", False, traceback.format_exc()))

    # B2-direct: the ceiling logic in isolation (deterministic; removes the
    # v_head random-projection variance that made the integration attempt
    # non-consecutive). Wiring (seam -> observe_value) is covered by 4.7's
    # test_k_m9_3_observes_seam_path_v; this confirms the firing itself.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        trainer = _build_trainer(Path(tmp))
        try:
            kr = trainer.m9_kills
            sustained = kr.value_ceiling_sustained
            for _ in range(sustained - 1):
                kr.observe_value(2.0e3)
            fired_before, _ = _value_kill_fired(trainer)
            kr.observe_value(2.0e3)  # the sustained-th consecutive over-ceiling
            fired_after, fset = _value_kill_fired(trainer)
            # also confirm an under-ceiling obs would have reset (consecutive,
            # not cumulative): fresh registry, 4 over, 1 under, 4 over -> no fire
            kr2 = _build_trainer(Path(tmp), seed=21).m9_kills
            for _ in range(sustained - 1):
                kr2.observe_value(2.0e3)
            kr2.observe_value(0.0)  # reset
            for _ in range(sustained - 1):
                kr2.observe_value(2.0e3)
            reset_fired = any(("value" in str(k).lower()) or str(k).endswith("3")
                              for k in kr2.fired())
            confirmed = bool(fired_before or (not fired_after) or reset_fired)
            v.append(Verdict(
                "B2-direct: K-M9-3 ceiling backstop fails (wrong consecutive "
                "semantics or no fire)",
                confirmed,
                f"fired after {sustained-1} consecutive over-ceiling = "
                f"{fired_before} (want False); after the {sustained}th = "
                f"{fired_after} (want True), set={fset}; "
                f"reset-test (over x{sustained-1}, under, over x{sustained-1}) "
                f"fired = {reset_fired} (want False). "
                + ("BROKEN." if confirmed else
                   "fires exactly at the sustained threshold, not before, and an "
                   "under-ceiling observation resets the counter (consecutive, "
                   "not cumulative). Combined with the seam-feeds-observe_value "
                   "wiring test, K-M9-3 is confirmed end-to-end.")))
        except Exception:
            v.append(Verdict("B2-direct: probe raised", False, traceback.format_exc()))
    return v


# ---------------------------------------------------------------------------
# C — deadly-triad oscillation
# ---------------------------------------------------------------------------
def probe_c() -> list[Verdict]:
    v = []
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        trainer = _build_trainer(Path(tmp))
        try:
            torch.manual_seed(9)
            # Two very different transition regimes, alternated each cycle —
            # the off-policy + bootstrapping + function-approx stress case.
            pat_a = torch.randn(1, D) * 2.0
            pat_b = torch.randn(1, D) * 2.0
            vs = []
            for i in range(60):
                s_t = pat_a if i % 2 == 0 else pat_b
                s_next = pat_b if i % 2 == 0 else pat_a
                a_t = torch.randn(1, D)
                trainer.observe_transition(s_t, a_t, s_next, ctx={})
                vs.append(float(trainer.v_head(s_t).mean().item()))
            # divergence signature: growing oscillation amplitude. Compare the
            # spread (max-min) of the last third vs the first third.
            third = len(vs) // 3
            early = vs[:third]
            late = vs[-third:]
            early_spread = max(early) - min(early)
            late_spread = max(late) - min(late)
            growth = late_spread / (early_spread + 1e-6)
            fired, _ = _value_kill_fired(trainer)
            confirmed = bool(growth > 3.0 or fired or abs(vs[-1]) > 1e3)
            v.append(Verdict(
                "C: alternating transitions drive growing V oscillation / "
                "divergence (deadly triad)",
                confirmed,
                f"V spread early third = {early_spread:.3f}, late third = "
                f"{late_spread:.3f} (growth {growth:.2f}x), final V = {vs[-1]:.3f}, "
                f"value-kill fired = {fired}. "
                + ("OSCILLATION GROWING — Polyak target not damping the triad."
                   if confirmed else
                   "oscillation amplitude is stable/shrinking across the run — the "
                   "slow Polyak target (tau=0.005) damps the triad at this size.")))
        except Exception:
            v.append(Verdict("C: probe raised", False, traceback.format_exc()))
    return v


def main() -> int:
    allv = []
    allv += report("A: no-grad robustness", probe_a())
    allv += report("B: reward scale (B1) + ceiling fires (B2)", probe_b())
    allv += report("C: deadly-triad oscillation", probe_c())
    n = sum(1 for x in allv if x.confirmed)
    print("\n" + "=" * 62)
    print(f"PHASE 4b ADVERSARIAL: {n}/{len(allv)} concerns CONFIRMED")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
