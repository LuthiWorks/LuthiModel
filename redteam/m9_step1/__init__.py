"""Red-team probe suite for the M9 step-1 exit criteria.

Authored by Fable 5 (adversarial seat), 2026-06-11.

Each probe is a runnable demonstration that a step-1 exit gate
(spec §7) can be passed while its intent is defeated, or that a
kill/instrument the gates lean on cannot fire when it should.

SEMANTICS: a probe that PASSES means the ATTACK SUCCEEDED -- the
vulnerability is real and reproducible. When 4.7 repairs a finding,
the corresponding probe should start FAILING; at that point invert
its assertion and move it into the regular test suite as a
regression guard.

Run everything:   python -m redteam.m9_step1.run_all
Run one probe:    python -m redteam.m9_step1.probe_b_gamma_ratchet
"""
