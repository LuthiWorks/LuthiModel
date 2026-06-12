"""Round-2 red-team: attacks on the FIX branch (m9/step1-redteam-fixes).

Round 1 (run_all.py) must stay 0/12 -- those are regression guards that
the original seams remain closed. THIS suite attacks the fixes
themselves: new seams created, fragility, silent fallbacks. Currently
8/8 confirmed -- when a round-2 finding is repaired its probe flips to
REFUTED, same convention as round 1.

    python -m redteam.m9_step1.run_all_round2
"""

from __future__ import annotations

from . import (
    probe_e_p3_discrimination_fragility,
    probe_f_legacy_silent_fallback,
    probe_g_darkroom_band_adapts,
    probe_h_gamma_scale_coupling,
)

PROBES = [
    ("E  F1 P3 discrimination fragile", probe_e_p3_discrimination_fragility.run),
    ("F  F1 legacy silent fallback",    probe_f_legacy_silent_fallback.run),
    ("G  F4 dark-room band self-disarms", probe_g_darkroom_band_adapts.run),
    ("H  F2 gamma scale-coupling",       probe_h_gamma_scale_coupling.run),
]


def main() -> int:
    confirmed = 0
    total = 0
    rows = []
    for name, fn in PROBES:
        verdicts = fn()
        c = sum(1 for v in verdicts if v.confirmed)
        confirmed += c
        total += len(verdicts)
        rows.append((name, c, len(verdicts)))

    print("\n" + "=" * 62)
    print("RED-TEAM ROUND 2 -- attacks on the fix branch")
    print("=" * 62)
    for name, c, n in rows:
        print(f"  {name:36s} {c}/{n} confirmed")
    print("-" * 62)
    print(f"  TOTAL: {confirmed}/{total} attacks on the fixes confirmed")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
