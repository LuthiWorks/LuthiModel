"""PROBE F (round 2) -- F1's legacy fallback reopens the original
seam SILENTLY under missing or partial §A wiring.

4.7's audit brief flagged exactly this: "can a downstream caller
accidentally use the legacy path and recreate the original seam
silently? ... worth probing whether a partial wiring -- e.g.,
decoders provided but bands not -- fails loudly or silently."

`compute_g_candidates` originally dispatched on
`has_per_candidate_path()` (all four §A modules present), and
when any were missing it silently fell back to the legacy
candidate-invariant `compute_g` path -- the exact bug probe_a
documented -- with NO error, NO warning.

Updated 2026-06-12 (F-C fix): construction now FAIL-LOUDS on
partial or implicit-legacy wiring. The probe's same attack
assertions flip to REFUTED because the construction raises a
ValueError where the legacy fallback used to happen silently.

  F1a (REFUTED after F-C). Constructing an EFEEvaluator with NO
       §A modules and NO `allow_legacy=True` now raises a
       ValueError telling the caller exactly what's wrong.
  F1b (REFUTED after F-C). PARTIAL wiring (decoders + bands but
       delta_s_* missing) raises a ValueError naming the missing
       modules. Partial wiring is now indistinguishable from a
       loud failure -- the silent fallback is gone.
"""

from __future__ import annotations

import torch

from luthi.v2.m9.efe import EFEEvaluator

from ._common import (
    Verdict,
    build_activity_bands,
    build_decoders,
    build_predictor,
    build_preferences,
    context_and_positions,
    report,
)

K = 8


def run() -> list[Verdict]:
    torch.manual_seed(0)
    predictor = build_predictor()
    predictor.eval()
    prefs = build_preferences()

    # --- F1a: no §A modules at all -- legacy is no longer the silent default. ---
    raised_bare = False
    raised_bare_msg = ""
    try:
        EFEEvaluator(predictor, prefs)  # bare construction (no allow_legacy)
    except ValueError as e:
        raised_bare = True
        raised_bare_msg = str(e)

    v1 = Verdict(
        "F1a: an EFEEvaluator with NO §A modules silently runs the "
        "legacy bug-mode path -- candidate-invariant c_con (the original "
        "A1 seam) -- with no error and no warning",
        not raised_bare,
        f"construction raised={raised_bare}; the F-C fix requires "
        f"`allow_legacy=True` for the bare path so silent reverts are "
        f"impossible. Raise message: {raised_bare_msg[:120]!r}",
    )

    # --- F1b: partial wiring -- decoders + bands but delta_s_* missing. ---
    raised_partial = False
    raised_partial_msg = ""
    try:
        EFEEvaluator(
            predictor, prefs,
            decoders=build_decoders(),
            activity_bands=build_activity_bands(),
            # delta_s_module / delta_s_band deliberately omitted
        )
    except ValueError as e:
        raised_partial = True
        raised_partial_msg = str(e)

    v2 = Verdict(
        "F1b: PARTIAL wiring (decoders + activity_bands present, "
        "delta_s_* missing) also silently degrades to legacy bug-mode -- "
        "a caller who wired 'most of it' gets the seam back with no "
        "loud failure",
        not raised_partial,
        f"construction raised={raised_partial}; the F-C fix names the "
        f"missing modules in the message so the caller cannot leave a "
        f"half-wired EFE in production. Raise message: "
        f"{raised_partial_msg[:120]!r}",
    )

    # --- F1c (new round-2 follow-up): explicit legacy still works. ---
    # The opt-in path must remain available for backward-compat tests of
    # the pre-F1 behavior, otherwise the F-C fix breaks legitimate use.
    bare_legacy_ok = False
    try:
        EFEEvaluator(predictor, prefs, allow_legacy=True)
        bare_legacy_ok = True
    except Exception:
        pass

    v3 = Verdict(
        "F1c: explicit legacy (allow_legacy=True) no longer permitted -- "
        "the F-C fix accidentally broke backward-compat tests",
        not bare_legacy_ok,
        f"allow_legacy=True construction succeeded: {bare_legacy_ok} "
        "(legitimate opt-in must remain available; the fix targets silent "
        "fallback, not all legacy use)",
    )

    return report(
        "PROBE F: F1 legacy fallback reopens the seam silently",
        [v1, v2, v3],
    )


if __name__ == "__main__":
    vs = run()
    confirmed = sum(1 for v in vs if v.confirmed)
    if confirmed == 0:
        print(f"\nAll {len(vs)} attacks REFUTED -- partial/missing wiring fails loudly.")
    else:
        print(f"\n{confirmed}/{len(vs)} attacks confirmed -- silent fallback.")
