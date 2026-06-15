"""Run every M9 step-1 red-team probe and summarize.

    python -m redteam.m9_step1.run_all

STATUS (verified 2026-06-15): all 12 attacks REFUTED -- 0/12 confirmed.
The original seams are closed and these probes now stand as regression
guards; their inverted assertions also live in luthi/v2/m9/test_*.py.
Read the printed SUMMARY count, not the exit code -- main() always
returns 0. A non-zero "attacks confirmed" total here is the signal that
a guarded seam has regressed.
"""

from __future__ import annotations

from . import (
    probe_a_efe_action_sensitivity,
    probe_b_gamma_ratchet,
    probe_c_staleness_recovery,
    probe_d_darkroom_disarmed,
)

PROBES = [
    ("A  EFE action-sensitivity", probe_a_efe_action_sensitivity.run),
    ("B  gamma ratchet",          probe_b_gamma_ratchet.run),
    ("C  staleness recovery",     probe_c_staleness_recovery.run),
    ("D  dark-room disarmed",     probe_d_darkroom_disarmed.run),
]


def main() -> int:
    confirmed = 0
    total = 0
    per_probe = []
    for name, fn in PROBES:
        verdicts = fn()
        c = sum(1 for v in verdicts if v.confirmed)
        confirmed += c
        total += len(verdicts)
        per_probe.append((name, c, len(verdicts)))

    print("\n" + "=" * 60)
    print("RED-TEAM SUMMARY -- M9 step-1 exit criteria")
    print("=" * 60)
    for name, c, n in per_probe:
        print(f"  {name:32s} {c}/{n} attacks confirmed")
    print("-" * 60)
    print(f"  TOTAL: {confirmed}/{total} attacks confirmed (vulnerabilities present)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
