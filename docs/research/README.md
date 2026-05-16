# Research Log

Dated research notes documenting the iterative process of building, testing,
and revising LuthiModel. Each entry captures not just the result but the full
chain of reasoning — including wrong turns, unexpected findings, and the
revisions they prompted.

See `CLAUDE.md` section "Research Log" for the spec that governs these entries.

## Entries

*(newest first)*

- [2026-05-16 — Catastrophic-forgetting harness for v2 consolidation](2026-05-16_catastrophic-forgetting-harness.md)
  — Building the behavioral falsifier for Salvatori attractor consolidation.
  Four iterations: discovered v2's substrate intrinsically resists forgetting,
  metric had to switch from pred_err to weight_drift, and the attractor
  pathway preserves *dynamics* (not weight) by design. Two xfail-strict
  markers pin the findings to code. The right behavioral test for attractor
  (recovery-probe pattern) is the next step.
