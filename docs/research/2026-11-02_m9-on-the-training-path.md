# M9 on the training path, and handed to Sanctuary — 2026-11-02

## Objective

Brian, 2026-11-02, after an explanation of what M9 is: "Ok awesome. Lets get
this on the training path and connect it to sanctuary."

Until today `luthi/v2/m9/` was built, unit-tested and red-teamed, but nothing
trained it (KNOWN_INCOMPLETE #3). In Sanctuary, the only M9 that ever ran
was a fresh, untrained `M9Trainer` built around a loaded model
(`examples/validate_seam_integration.py`). The aim:

1. Let the pilot driver train the M9 heads alongside the trunk. Opt-in, so
   no registered family changes meaning.
2. Load a trained M9 run as Sanctuary's actor and learner through
   `sanctuary_interface`, so Sanctuary never reaches into Luthi internals.
3. Give Sanctuary a way to read what a selected action does on each output
   channel, so the action can be carried out. This is the Luthi half of
   Sanctuary's 2026-08-01 audit item W1.

## Process

### Step 1: read what exists

- `M9Trainer` extends `JEPATrainer`. Its `train_step` runs the JEPA step,
  then `_m9_head_step` on **detached** latents, and finally wipes any
  gradient the M9 backward left on the JEPA parameters. It trains:
  - the value head (TD on a reward of minus the planner's best expected free
    energy);
  - the habit net (distilling MCTS visit counts);
  - the decoders (cycle-consistency);
  - the rest action;
  - `output_proj` (cross-entropy from the predictor's latents).
- Checkpoint and resume carry `m9_*` keys. The §5 device sweep is in the code
  and tested (`tests/m9/test_runner.py`). To-Do item 5(a) and
  KNOWN_INCOMPLETE #3 still called it open; that is stale.
- `sanctuary_interface` already had `encode_state`, `select_action` and
  `observe_transition`.
- `run_rebuild.rebuild_run` already rebuilds a pilot run from its own
  recorded provenance. A loader can stand on it.

### Step 2: the driver switch, and the first failure

- I added `--m9`. It makes `_run_one` build `M9Trainer` instead of
  `JEPATrainer`. Runs land in `runs/jepa_pilot/m9/`, one level below every
  family. `aggregate()` and the per-run resume check therefore never see
  them. `pilot_result.json` gains an `m9` block on M9 runs only.
- The first CPU smoke failed at once:
  `M9Trainer.train_step() got an unexpected keyword argument 'will_log'`.
- `JEPATrainer.train_step` had gained logging flags after the M9 override was
  written. Nothing had ever driven an `M9Trainer` through `JEPATrainer.run()`
  before; every M9 test calls `train_step` directly.
- The fix passes the flags through.

### Step 3: does M9 change what the trunk learns?

The smoke then completed. Its held-out loss differed from the same arm
without `--m9` (living 1.960945 vs 1.962885). By construction M9 sends no
gradient into the trunk, so where did the difference come from? There were
two couplings:

1. **The data stream.** `M9Trainer.__init__` took its retention-gate batch
   from `data_loader.next_batch`, the training stream itself. That shifted
   every later batch by one.
   - It also breaks the class's own HELD-OUT CONTRACT note: a production
     loader re-yields that batch every epoch, so the retention gate would
     measure data the model keeps training on.
   - Fix: an optional `retention_batch` argument. The driver passes the run's
     first holdout batch. `None` keeps the legacy behaviour.
2. **The random stream.** M9's module init, MCTS sampling and the predictor's
   dropout during planning all draw from the global RNG that the trunk's
   step also draws from.
   - Fix: `M9Config.rng_seed`. When set, M9's construction and head step run
     inside `torch.random.fork_rng` on a stream of their own. That stream
     continues from use to use and is saved in the checkpoint.
   - `None` keeps the legacy behaviour. The driver sets `9000 + seed`.
   - To make construction run inside the fork, I moved the M9 part of
     `__init__` into `_build_m9`. The body is unchanged.

With both fixes, `tests/test_m9_training_path.py` pins the result. On CPU,
the trunk of an `--m9` run is **bit-identical** to the same arm without it.
That covers the encoder, living buffers, living extra state, predictor,
projection heads, the global RNG and the loader position. Only
`output_proj`, which M9 trains, differs.

To check the test has teeth, I reverted each fix in turn. Each revert failed
it at `embedding.weight`. So it catches both couplings, and it would catch a
gradient leak the same way.

What this buys: M9 can ride along on any arm without changing what that
arm's trunk learns. On ROCm, float nondeterminism makes bit-identity
uncheckable, but the coupling is gone by construction.

### Step 4: loading a trained run for a life

`load_m9_run(run_dir, *, life_dir, device)`:

- **Rebuild.** It rebuilds the model with `rebuild_run`, constructs an
  `M9Trainer` around it, and calls `trainer.resume(checkpoint)`. That
  restores trunk, predictor, heads, all three optimizers, sampler and corpus
  position. Corpus replay during life therefore continues the run's own
  optimizer.
- **A life never writes into its run.** Everything the life writes goes to
  `life_dir`. A `life_dir` inside the run is refused. If `life_dir` already
  has checkpoints, the life continues from the newest. `save_life` writes
  the next one. A life's checkpoints are never rotated out.
- **Refusals.** It refuses a run without an `m9` block, and a checkpoint
  without M9 heads. Fresh heads must never pass for trained ones.

Building this exposed three continuity problems in `M9Trainer` itself.
Each would have bitten a resumed life:

- `resume` left the retention gate's rollback target at whatever was
  snapshotted at construction. A resumed life's first retention breach would
  have rolled its core back past everything it had lived since it started.
  `resume` now takes the loaded state as the last known good.
- The lived optimizer and the lived step and interleave counters were never
  checkpointed. They are now (`m9_lived_*`), and older checkpoints are
  tolerated.
- `resume` loaded M9 state without `map_location`, so a GPU-written
  checkpoint could not be read on CPU. It now maps to CPU, like the parent.

### Step 5: what an action does

`M9Trainer.decode_selection(action, s_t=, context_latents=)` reports each
output channel:

- **Content**, decoded from the action itself, as the truthfulness
  preference reads it: attention gates per modality, and memory salience.
- **Whether the channel acts**, by the test the EFE evaluator uses for
  emission. The decoder activity of the predicted next state under the
  action is compared against the channel's activity band, and against the
  same reading under `a_rest(s_t)`. "Silent" means the action does nothing
  on that channel that resting would not.
- **Not the intensity heads.** I first reached for the readable summary's
  intensity heads. Its own docstring calls them the legacy form, arbitrary
  at init (F4), and says new callers must use the activity bands. They are
  reported, never used to decide.

`sanctuary_interface.decode_action` delegates to it.

### Step 6: what I did not build

- **A learning-settings decoder** (the mind's allow/defer/emphasize over
  its own consolidation, Sanctuary 2026-11-02). M9's decoders get their
  meaning from consequences they can predict. A learning setting's
  consequence arrives sleeps later, as a change in who the model is.
  Nothing in EFE predicts that, so a decoder for it would be an arbitrary
  dial pulled by an untrained head and logged as the mind's choice. For
  Luthi nothing consolidates by these settings yet anyway
  (`consolidation_active=False`). This waits for a design with a real
  consequence model.
- **Speech.** M9 trains `output_proj` on the predictor's latents.
  Generation reads it from the encoder's. So M9 alone does not make text
  real, and `LoadedM9Run.speech_trained` is False unless the run had
  `w_ntp > 0`. Sanctuary uses that to keep an untrained readout silent.

### Step 7: running it from Sanctuary

The first live run from Sanctuary's working directory re-tokenized the
corpus and wrote the cache into `sanctuary/corpus_build/cache/`. The
default cache directory in `luthi/data.py` was relative to the working
directory.

- On Brian's PC that would have meant hours of tokenization on Luthi's first
  start, and a second multi-GB cache inside Sanctuary's repo.
- The default is now anchored at this repo. Every launcher runs from this
  repo's root, so for training the location is unchanged.
- Pinned by `test_the_corpus_cache_lives_in_this_repo_wherever_python_runs`.

Run on the smoke M9 run (8 cycles, CPU, 64d), the loop works end to end:

- a lived update every cycle after the first;
- attention and memory carried out from the decoded action;
- no speech;
- about 0.25 s per cycle after the first. The first cycle takes about 18 s,
  for loading and tokenizing.

The activity bands were not yet armed, so "silent" was decided by the rest
reference and the absolute floor alone. The text channel sometimes "acts"
while speech is untrained, so those choices go nowhere.

## Conclusion

- `--m9` trains the action layer alongside the trunk without touching what
  the trunk learns: bit-identical on CPU, by test.
- A trained M9 run loads as Sanctuary's actor and learner through the
  adapter, continues across restarts, and never writes into its run.
- Nothing here has run at scale or on ROCm. That needs Brian's PC.

What I don't know:

- **Cost.** The M9 step adds MCTS planning (8 simulations), several small
  heads and a per-step parameter clone. In the CPU smoke (20 steps at 64d,
  probes included) wall time went from 11.4 s to 13.3 s for the living arm
  and from 10.9 s to 12.9 s for the dead arm. Those numbers are dominated by
  fixed costs. The ROCm cost at 768x8 is unmeasured.
- **Whether the heads learn anything meaningful** from corpus batches,
  where every "transition" is an independently sampled window.
- **Whether the value head's reward fits Brian's demonstration-only rule.**
  That is Brian's ruling to make (DECISIONS.md, 2026-11-02).

## Artifacts

- Code:
  - `luthi/v2/m9/runner.py`: `retention_batch`, `rng_seed` / `_m9_rng` /
    `_build_m9`, train_step flags, `decode_selection`, `load_m9_state`,
    lived-state checkpoint keys, resume fixes;
  - `luthi/sanctuary_interface.py`: `LoadedM9Run`, `load_m9_run`,
    `save_life`, `decode_action`;
  - `luthi/v2/multimodal_data.py`: `DeviceLoader`, moved from the driver;
  - `scripts/jepa_pilot_driver.py`: `--m9`.
- Tests: `tests/test_m9_training_path.py` (7). The full suite passes on CPU
  except `test_unset_declaration_means_rocm`, which fails by design without
  ROCm (CLAUDE.md).
- The Sanctuary side: `sanctuary/docs/research/2026-11-02_luthi-action-layer-connected.md`.
