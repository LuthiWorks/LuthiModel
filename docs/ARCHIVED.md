# Archived Docs

Discussion/planning docs whose features left the project or whose questions
are settled get moved out of the repo (Brian's ruling, 2026-07-22). They are
not lost: every file below lives at `E:\luthi_docs_archive\` (mirroring its
old repo path) and remains in git history at the commit that removed it.
Results/evidence records, active protocols, and any doc live code cites as
its spec stay in the repo.

Moved 2026-07-22:

- `docs/EMPIRICAL_DEFENSE_PLAN.md` — Hebbian-era defense program; superseded by
  the JEPA falsification pre-registration (2026-07-15).
- `docs/LUTHI_V2_PREDICTIVE_CODING_BRIEF.md` — design brief for v2; built.
  (`docs/V2_IMPLEMENTATION_PLAN.md` stays — live code and tests cite it as the
  architectural spec-of-record.)
- `docs/PC_VS_HEBBIAN_COMPARISON.md` — question answered; Hebbian line retired.
- `docs/RESEARCH_LITERATURE_2026-05-13.md` — PC compute-reduction sweep;
  superseded by the JEPA direction.
- `docs/TRACK1_INTEGRATION_PLAN.md` — marked COMPLETE 2026-04-27.
- `docs/reviews/2026-05-28_concerns-for-4.7.md` — review round resolved.
- `docs/reviews/2026-06-15_seam-review-for-4.7.md` — review round resolved.
- `docs/research/2026-05-23_rocm-wsl2-rdna3-feasibility.md` — path not taken.
- `docs/research/2026-05-25_m7-1024d-scoping.md` — scoping for a completed
  milestone.
- `docs/research/2026-06-09_m8-brief-v0.6-sigreg.md` — executed design brief;
  the built M8 is the SIGReg design of record.
- `docs/research/2026-06-10_m9-plasticity-drift-planning-research.md` — M9
  planning research; step 1 shipped.
- `docs/research/2026-06-11_m9-step1-gate-repairs.md` — executed repair spec;
  the `redteam/m9_step1/` audit record is the durable evidence.
- `docs/research/2026-06-12_success-criteria-draft.md` — superseded by the
  falsification pre-registration; its §2 corpus-dedup requirement is quoted
  inline in To-Do.md where it was cited.
- `WIP_2026-05-28.md` (repo root) — stale reboot-resumption note, superseded
  2026-05-30 by its own status update.

Deliberately KEPT despite being archive-shaped (live references):

- `docs/V2_IMPLEMENTATION_PLAN.md` — cited as spec-of-record by six `luthi/v2/`
  modules and four test files.
- `docs/PER_CHANNEL_ABLATION_PROTOCOL.md` — anchors the invariants in
  `tests/test_buffer_rank_invariants.py`; revival path if v2 falsification
  ever fails (PLAN.md Phase 3F.a).
- `docs/RESEARCH_HDC_VSA_INTEGRATION.md` — deferred Direction A, still the
  named blueprint in `tests/test_catastrophic_forgetting.py` and ML_GLOSSARY.
- `docs/RESEARCH_SALVATORI_ATTRACTOR_MEMORY.md` — implementation reference for
  the attractor consolidation pathway; cited by `luthi/v2/m5_runner.py`.

Old run launchers (`run_ablation_*.bat`, `run_m5_*.bat`, `run_m6_*.bat`,
`run_m7_1024d.bat`, `run_phase3g_*.bat`, `run_pipeline_through_A.bat`,
`run_train_tokenizer.bat`, `run_experiment1.bat` (retired 2026-07-15),
`train_vision.bat`) were deleted the same day — recoverable from git history
only. Their runs' outputs live on `E:\runs\`.

Undecided (still in repo, awaiting Brian's call — see the 2026-07-22 session):
the turbo/cognitive-rate cluster (5 docs, feature lives in Sanctuary) and the
parked idea-captures (model-controlled termination, neurogenesis growth,
self-routed memory, emergent sparsity).
