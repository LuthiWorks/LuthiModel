# What fits on Brian's card, and a smaller episode store — 2026-11-11

Fable 5.2, in session with Brian.

## Objective

Brian, 16:16: the other reason long training was paused is "that I can't train a full
sized LuthiModel with this hardware." He asked what size the card (RX 7800 XT, 16 GB)
can realistically train, accepting that it would be mostly conjecture. Then, 16:24, on
a memory cost I had found and left alone: "Efficiency improvements are ALWAYS welcome,
as are suggestions of any kind."

## Process

### Step 1: sizes, counted

Built on CPU with the driver's defaults and counted (`vocab 32000`, `max_seq_len 128`):

| Width x blocks | Trained parameters | Stored state | Of which the episode store |
|---|---:|---:|---:|
| 512 x 8 | 43.5M (0.17 GB) | 0.59 GB | 0.54 GB |
| 768 x 8 | 71.9M (0.29 GB) | 1.31 GB | 1.21 GB |
| 768 x 4 | 62.5M | 0.66 GB | |

- Trained parameters fit `4 d^2 L + 69,100 d` to within 1%.
- The episode store is `64 x d^2` numbers a living layer, in FP32. It is 91.7% of all
  stored state at 768 x 8.

### Step 2: speed, from the record

`docs/research/2026-08-11_768x8-family-spec.md`, on this card under DirectML, 4,096
tokens a step: 512 x 8 ran 0.62 s a step and 768 x 8 ran 1.40 s. The ratio (2.26) is
the ratio of `d^2 L` (2.25), so I scaled larger sizes by `d^2 L`.

### Step 3: the estimate

Memory is trained parameters at 16 bytes each (weight, gradient, two optimizer
moments) plus stored state, before working memory.

| Width x blocks | Trained | Memory before working memory | Tokens a day | Days per billion tokens |
|---|---:|---:|---:|---:|
| 512 x 8 | 44M | 1.3 GB | 570M | 2 |
| 768 x 8 | 72M | 2.5 GB | 250M | 4 |
| 1024 x 12 | 121M | 5.5 GB | 95M | 10 |
| 1536 x 16 | 257M | 14.7 GB | 32M | 32 |
| 2048 x 24 | 545M | 37 GB | 12M | 84 |

- The first two rows are measured. The rest are scaled from them.
- The timings predate ROCm and the action layer on the training path. Good to a
  factor of two at best.
- Working memory (activations) is not counted. It grows with `d x L x tokens`.
- A 4-billion-parameter model needs about 64 GB for its trained parts alone.

### Step 4: the episode store

- I first reported it as "reserved for nothing", from the note in `CLAUDE.md` that
  the store is empty in the configuration that runs.
- Reading `KNOWN_INCOMPLETE.md` #5 corrected that. The store is inert, not dead:
  blocks 0 to 4 stored nothing in 54,000 steps at 768 x 8 because the admission
  threshold is far above the measured salience, and the fix (`adaptive_episodes`) is
  decided for a future family. So it should not be removed.
- The model already takes two settings that shrink it: `compressed_episodes` (INT8,
  a quarter of the bytes; the code path is tested in `test_pc_consolidation.py`) and
  `num_episodes`. The training driver passed neither through.
- **Built:** `--compressed-episodes` and `--num-episodes N` on
  `scripts/jepa_pilot_driver.py`. Absent, `model_kwargs` is unchanged, so every
  existing run and checkpoint is untouched. Set, they are recorded in the run's
  `pilot_result.json` with the rest of `model_kwargs`, which is what `load_m9_run`
  rebuilds from.
- With `--compressed-episodes` the store at 1536 x 16 falls from 9.7 GB to 2.4 GB, and
  that row's memory before working memory from 14.7 GB to about 7.4 GB.

## Conclusion

- The card's practical ceiling as built is about 1024 x 12, 121M trained parameters.
- With the smaller store, 1536 x 16 may fit. Not measured.
- A run with either flag is its own configuration. INT8 changes what a layer recalls
  when it has stored something.

Not known: working memory at any size above 768 x 8; speed under ROCm; how many
tokens a Luthi of any size needs.

Limits of what was built: `scripts/resume_queue.py` restarts the driver by stage and
does not pass these flags. A run started with them must be resumed with the same
flags on the command, or its checkpoint will refuse to load (loudly: the store's
shape will not match).

Suggestions, not built:

1. A timed 200-step run at 1024 x 12 on the card, to replace the scaled rows.
2. Mixed precision for the trained parts, if ROCm supports it on this card: it would
   roughly halve working memory. It needs checking against the living layers, which
   must stay in full precision.
3. Allocating the episode store only when a layer first stores something, so an inert
   store costs nothing. That changes checkpoints, so it needs care.

## Artifacts

- `scripts/jepa_pilot_driver.py` (two flags); `tests/test_episode_store_flags.py` (3).
- Counts: built with `MultimodalPredictiveCodingLM(vocab_size=32000, d_model=d,
  n_blocks=L, n_heads=..., max_seq_len=128)` and summed.
