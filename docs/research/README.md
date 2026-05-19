# Research Log

Dated research notes documenting the iterative process of building, testing,
and revising LuthiModel. Each entry captures not just the result but the full
chain of reasoning — including wrong turns, unexpected findings, and the
revisions they prompted.

See `CLAUDE.md` section "Research Log" for the spec that governs these entries.

## Entries

*(newest first)*

- **⚠️ DEFERRED** — [2026-05-16 — Plasticity partitions design exploration](2026-05-16_plasticity-partitions-design.md)
  — Architectural proposal arising from M6's NFF-attenuates-with-depth
  observation: partition weights into identity / knowledge / ephemeral tiers,
  with MAS-style empirical importance measurement as the primary assignment
  mechanism. **Demoted to deferred the same day it was written** after
  weighing it against the architecture-accumulation cost and the absence
  of a measured identity-drift problem in v2. Do not implement without
  first re-reading the "Why this was deferred" section. Preparatory clamp
  change (plasticity floor 0.1 → 0.01) DID land and stands on its own merits
  regardless of partition plans. Full implementation contingent on M6 completion + 256d
  cross-check + val-loss-neutral-or-better validation.

- [2026-05-16 — Catastrophic-forgetting harness for v2 consolidation](2026-05-16_catastrophic-forgetting-harness.md)
  — Building the behavioral falsifier for Salvatori attractor consolidation.
  Four iterations: discovered v2's substrate intrinsically resists forgetting,
  metric had to switch from pred_err to weight_drift, and the attractor
  pathway preserves *dynamics* (not weight) by design. Two xfail-strict
  markers pin the findings to code. The right behavioral test for attractor
  (recovery-probe pattern) is the next step.
