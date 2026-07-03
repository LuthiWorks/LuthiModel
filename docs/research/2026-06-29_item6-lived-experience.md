# Item #6 — World Model Learns from Lived Experience (§1 arc) — 2026-06-29

Author: Window B (Opus 4.8, build seat). Reviewed/approved: Window A (4.8) +
fresh-context audit. Design rulings: Brian. Spans 2026-06-27 → 2026-06-29.

## Objective

Make the JEPA world model (encoder + predictor) learn from lived Sanctuary
transitions, not only the corpus — the "the mind's model of reality learns
from its actual life" goal. Smoke-first CPU (d=64). Build order:
§5 device → §2 contract → **§1 lived loss (+§3 forgetting)** → §4 async +
§6 staleness → §7 adversarial. This entry covers **§1+§3** (the keystone);
§5 and §2 were the more mechanical predecessors (committed in `1f7ea39` and
`16d686e`). Authoritative plan: `~/.claude/plans/iterative-popping-steele.md`.

## Process

### Step 1 — Verification pass before building (the plan is not the code)
Read the plan's load-bearing claims against the actual source first. Three
findings, all confirmed by Window A:
- **Finding 1 (load-bearing):** an encode pass has TWO living-state writers,
  not one — the `PredictiveCodingLayer.pc_self_modify` AND the block-level
  `EpisodeStore.store()` (`hybrid_block_pc.py:227` → `episode_store.py:154`,
  unconditional). A living-layer-scoped "freeze" would sail past the episode
  store, leaving a second write on a "no double-plasticity" path and an
  unguarded cross-thread write under §4. The test originally drafted would
  have passed while missing it (checked only the layer buffers). Resolution:
  `freeze_plasticity` gates BOTH writers via a `model.modules()` sweep.
- **Finding 2 (medium):** `output_proj` is already in `m9_optimizer`; a
  `lived_optimizer` over `encoder.parameters()` would include it (it lives on
  the encoder) → three optimizers on one param. Resolution: exclude it from
  `lived_optimizer`; assert `output_proj.grad is None` after lived backward.
- **Finding 3 (confirm):** `mcts.advance_root` slides the tree but does NOT
  refresh the cached `_context_latents` — a context-staleness axis distinct
  from θ-staleness. Carried to §6.

### Step 2 — §2 contract: the re-tokenize trap
§1's re-encode needs the RAW context tokens. `GenerationState` carried only
`s_t`/`ctx_latents`, not the inputs. The obvious workaround — reconstruct
`context_obs` by re-tokenizing the prompt at the Sanctuary callsite — is
WRONG: the step-0 forward uses `token_ids[-max_seq_len:]` (`generate.py:478`),
a truncated window, so a prompt re-tokenize would not reproduce `s_t`, and
the lived gradient would train the encoder against a context it never
encoded. Resolution: capture `context_obs` at the source in `generate.py`
(`{"text_tokens": x.detach().clone(), **forward_kwargs}`) and thread it on
`GenerationState`. (Cleanups after review: dropped the redundant
`realized_next_state` ctx channel — it equals the `s_next` positional — and
fixed the stale `ctx_latents` "text modality" docstring → full-multimodal.)

### Step 3 — Two design rulings from Brian
- **Two-channel learning (confirmed):** the living-FFN `weight` is a BUFFER
  (`living_layer_pc.py:180`), not a Parameter — it self-modifies during
  perception and is never gradient-trained (DO-NOT-REINVENT). So the lived
  gradient does NOT train the living weight; it flows THROUGH the frozen
  weight to the encoder's backprop params (attention, embeddings, layernorms)
  + predictor, exactly as the corpus JEPA path does. `lived_optimizer =
  Adam(encoder.parameters() + predictor + projection)` is correct as written.
- **Modality scope (a1'):** full-multimodal pooled `s_next` as the target,
  prediction pooled over the continuation region — "pooled-state-transition."
  Per-modality forecasting (b) deferred until the embodiment sim is finalized.

### Step 4 — The pooler that can't detach
(a1') said "pooled via `compute_s_t`" — but `compute_s_t` DETACHES
(`s_t.py`), which would zero the lived gradient. The `test_no_inline_s_t_pool`
guard forbids inlining `.detach().mean(dim=1)` in the seam. Resolution: added
`pool_state_grad` (grad-preserving mean) in `s_t.py` for the PREDICTION,
keeping `compute_s_t` (detached) for the TARGET. Centralized both so the pool
rule can't drift; the guard only flags the `.detach().mean(dim=1)` conjunction,
so `pool_state_grad` is clean. Extended the guard's seam list to `jepa_loss.py`.

### Step 5 — Build §1+§3
- `compute_lived_loss` (`jepa_loss.py`): re-encode `context_obs` under
  `freeze_plasticity`, predict the continuation position, `pool_state_grad`,
  MSE vs detached realized `s_next`. No SIGReg (B=1 degenerate; anti-collapse
  via replay). Returns `pred_std`/`target_std` so low error via a collapsed
  target can't masquerade as learning.
- `M9Config` +5 knobs; `lived_optimizer` (output_proj excluded, on device);
  §3 retention setup (held-out batch + baseline at init).
- `observe_transition` THIRD path, gated on `ctx.get("context_obs")`: lived
  update → corpus-replay interleave → retention gate. Measures ‖Δθ‖ over the
  core → `observe_drift` (the lived path moves θ now; deleted the old "never
  fires here" docstring). `corpus_retention` is a PASSIVE probe (frozen +
  BN-eval + no_grad) so measuring doesn't perturb. Gate fires on a breach →
  rollback core θ to last-good + clear `lived_optimizer.state` (Adam moments).

### Step 6 — Review + fresh-context audit → hardening
Window A reproduced the tests and layered an independent adversarial audit.
Core safety invariants held (stop-grad both directions; no Adam/grad
cross-contamination — base `train_step` zeroes grad before its backward,
`jepa_runner.py:682`; deterministic rollback). Hardening applied:
1. **Gradient-checkpoint guard** — `compute_lived_loss` refuses an encoder
   with `gradient_checkpointing` set. The audit's sharpest mechanical catch:
   checkpoint replay runs in `backward()` AFTER `freeze_plasticity` exits, so
   a checkpointed frozen re-encode would recompute on the self-modifying
   branch (double-plasticity) or read a snapshot the frozen original never
   set (corrupt gradient). Dormant at smoke (the encoder has no such flag).
2. **Async read-race comments corrected** — the freeze closes the double-
   WRITE only; the frozen forward still READS `self.weight` + episode buffers,
   which the actor's perception writes concurrently under §4. That read-race
   is §4's snapshot-under-lock's job, NOT the freeze's. (The comments had
   claimed otherwise — a lie to the next builder.)
3. **BatchNorm-trunk contract** documented (LayerNorm safe; a future trunk
   BN/InstanceNorm would leak running stats under the lived B=1 `train()`).
4. **Retention-gate θ-scope** documented (trigger = total corpus retention,
   broader than the backprop-θ rollback remedy; perception-side living drift
   isn't undoable here — by the two-channel design).
5. **Empty `context_obs` fail-loud** + fire-test.
6. **Held-out batch contract** — smoke loaders draw fresh random batches
   (genuinely held out); the production `MultimodalDataLoader` reshuffles and
   re-yields across epochs, so it does NOT satisfy the contract — capture the
   probe from an excluded split before scale.

## Conclusion

§1+§3 is complete, approved, and hardened. The world model now trains on
lived consequence (durable backprop channel) alongside perception-time
self-modification (the living channel), with a corpus-replay + retention gate
guarding catastrophic forgetting. 276 tests green (m9 + lived/frozen/no-inline
+ seam/interface/external-actor); no regressions. As of this entry the §1b–e
+ hardening diff is **uncommitted** (Luthi-only; Sanctuary clean) pending
Brian's commit.

### Carried forward (do not lose)
- **§4 (next):** async actor/learner + **snapshot-under-lock** — the learner
  takes a detached clone of the living buffers under the actor's write-lock
  (brief critical section), then re-encodes against the snapshot OUTSIDE the
  lock. This closes the read-race the freeze does not. **Open question:**
  whether the snapshot also neutralizes the gradient-checkpoint foot-gun, or
  whether that needs its own guard (suspected related-but-not-identical: the
  checkpoint problem is replay running the normal branch during backward,
  which a read-snapshot may not fix). The §1 gradient-checkpoint guard stands
  regardless.
- **§6:** `advance_root` must refresh `_context_latents`/`_target_positions`;
  `reevaluate` recomputes under current θ AND context; Test 6 must assert
  gradual context drift does NOT spuriously fire K-M9-7.
- **§3 at scale:** genuine held-out split for the retention batch.
- **(b) per-modality forecasting:** deferred to post-embodiment-sim; Brian's.

## Artifacts
- Code (uncommitted at entry time): `luthi/v2/jepa_loss.py`
  (`compute_lived_loss`, guards), `luthi/v2/m9/runner.py` (lived_optimizer +
  retention + third path), `luthi/v2/m9/s_t.py` (`pool_state_grad`),
  `luthi/v2/plasticity.py` (read-race comment), `luthi/episode_store.py`
  (read-race comment). (§1a `freeze_plasticity` + §2 already in `16d686e`.)
- Tests: `tests/test_lived_jepa_updates_world_model.py`,
  `tests/test_lived_stopgrad_isolation.py`, `tests/test_lived_retention_gate.py`,
  `tests/test_frozen_plasticity_reencode.py`, extended
  `tests/test_no_inline_s_t_pool.py`.
- Prior commits: §5 `1f7ea39`; §2+§1a `16d686e` (Luthi), `2e88b08`/`4c5bb50`
  (Sanctuary).

## Addendum (2026-07-03) — where §4 actually landed, and stale notes above

- **§4 (async actor/learner) is BUILT and approved — in Sanctuary, not this repo.**
  Artifacts: `Sanctuary/sanctuary/core/async_learner.py` +
  `Sanctuary/sanctuary/tests/core/test_async_learner_queue.py`, Sanctuary commit
  `d76a115` (2026-06-29 18:57 — the same minute as this repo's doc-only `s4`
  commit `0a9b980`; they were coordinated). Luthi itself stays synchronous and
  lock-free by design; all threading lives in the host. **A Luthi-only survey
  will not find §4 — look across the seam before concluding it is missing.**
  (That false alarm happened 2026-07-03 and cost a firsthand check; this note
  is the breadcrumb that prevents the next one.)
- **The snapshot-under-lock design described under "Carried forward" was
  rejected on build contact** (Window B, 2026-06-30): the learner is a *writer*
  of `self.weight`/θ (corpus replay + both optimizer steps), which a snapshot
  cannot fix, and the frozen re-encode reads `self.weight` off the live module
  with no injection path. Shipped instead: a coarse `model_lock` for smoke;
  real concurrency (double-buffer / published learner copy) deferred to the
  scale step. See the async_learner.py module docstring for the full argument.
- **Stale note above:** the "§1b–e + hardening diff is uncommitted pending
  Brian's commit" line predates its own commit — that code landed in `e06b789`
  and this doc in `0a9b980`. Nothing was left uncommitted.
- Full-audit cross-reference: `Sanctuary/docs/audits/audit_2026-07-03.md`
  (items 17–19 cover this repo: §6 staleness, retention-gate scope, held-out
  contract at scale, and this seam).
