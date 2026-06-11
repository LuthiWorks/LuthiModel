"""M9: unified active-inference planning over a full-latent action space.

See `docs/research/2026-06-10_m9-step1-spec.md` for the build-ready
step-1 spec, `docs/research/2026-06-10_m9-build-plan.md` for the
full plan, and `docs/research/2026-06-09_m9-unified-planning-aif-research.md`
for the research grounding.

Step 1 scope: pragmatic-only planning loop. `beta_epi = 0`;
epistemic (MC-dropout parameter-novelty) is step 2. Width 256d;
gamma inferred from launch with the K-M9-4 divergence kill live.
"""
