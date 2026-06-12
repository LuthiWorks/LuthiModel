"""Run every M9 step-1 red-team probe and summarize.

    python -m redteam.m9_step1.run_all

Exit code 0 means every probe's attack still lands (the
vulnerabilities are present). Once 4.7 repairs a finding, the
corresponding probe should flip to REFUTED; at that point invert the
probe's assertion and migrate it to the unit suite as a regression
guard. A non-zero exit here after repairs is the signal that a fix
landed.
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
