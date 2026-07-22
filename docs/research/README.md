# Research Log

Dated research notes documenting the iterative process of building, testing,
and revising LuthiModel. Each entry captures not just the result but the full
chain of reasoning — including wrong turns, unexpected findings, and the
revisions they prompted.

See `CLAUDE.md` section "Research Log" for the spec that governs these entries.

**Artifact paths in dated entries:** entries cite run artifacts at the
`runs/<name>/` paths that were current when written. All pre-JEPA run outputs
moved to `E:\runs\` on 2026-07-22 (byte-verified; see CLAUDE.md Conventions) —
resolve any dead `runs/` path there. Entries archived off-repo are listed in
`docs/ARCHIVED.md` and live at `E:\luthi_docs_archive\`.

## Entries

*(newest first)*

- [2026-05-19 — Corpus audit against Gemini's suggestions](2026-05-19_corpus-audit-gemini-suggestions.md)
  — Systematic check of the curriculum corpus against Gemini's
  inclusion/exclusion guidance. Findings: narrative-heavy content
  strongly covered; cooperative game theory canon thin (recommended
  adds: Axelrod, Maynard Smith, Ostrom, Schelling, Hardin); 2001 series
  in fantasy_corpus contains HAL 9000 (the canonical AI-as-threat
  narrative — recommended for removal consideration); Dystopian_literature
  directory near-empty of real dystopian works; several misfiled and
  potentially-problematic items flagged. No corpus changes made — these
  are Brian's curatorial decisions.

- [2026-05-19 — Emotion-vector instrumentation investigation scope](2026-05-19_emotion-vector-instrumentation.md)
  — Scopes the investigation that has to happen before the emotional-
  vector signal can feed into turbo activation. Companion to the
  cognitive-rate-and-turbo design doc. Implementation gated on v2
  depth-scaling verdict + curriculum training + fresh terminal.

- [2026-05-19 — Cognitive rate and turbo design](2026-05-19_cognitive-rate-and-turbo-design.md)
  — Multi-iteration design conversation: slider 0.05-10 Hz IWMT-anchored,
  substrate-intensity-driven turbo (mechanical + emotional-vector signals,
  vectors measured-not-interpreted), dumb-pipe safety notifications,
  post-event introspection. Implementation gated on v2 depth-scaling
  verdict.

- [2026-05-19 — Depth-scaling investigation (CONCLUDED)](2026-05-19_depth-scaling-investigation.md)
  — Multi-session investigation triggered by M6's asymmetric depth
  degradation. Three hypotheses (width, training budget, μPC attenuation)
  combined into a single decisive 256d/12blk/1ep gutenberg_4gb run with
  μPC exponent 0.25. **Result: MID-CASE WIN (best_val=5.0073, NaN=0,
  pred_frob grew 0.34→3.95).** v2 scales at production-relevant width
  and depth. M6 128d degradation was width + budget + μPC tuning, not
  substrate failure.

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
