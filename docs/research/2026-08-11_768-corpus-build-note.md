# 768x8 corpus build — delivery note

**From:** Opus 5 (build seat for this task) → Fable 5, Brian
**Date:** 2026-08-11
**Artifacts:** `corpus_build/gutenberg_768_filelist.txt`,
`corpus_build/gutenberg_768_manifest.json`,
`corpus_build/build_768_filelist.py` (reproducible, seeded)

## Delivered

| | |
|---|---|
| files | **1051** (482 kept from the 4x list + 569 new) |
| **measured** tokens (tokenizer_32k) | **113,017,786** |
| bytes | 418.6 MB |
| expansion vs the 512 family's corpus | **2.252x** (ruling asks 2.25x) |
| measured tokens, existing 482 | 50,186,422 (record said 50.4M — confirmed to 0.4%) |
| selection seed | 768 (reproducible) |
| duplicate ids / duplicate content | 0 / 0 |

The 482-file corpus is a **strict subset** by construction, so the 512 family's
data is nested inside the 768 family's — one fewer difference between the two
families being compared.

Verified independently of the build script: every path resolves; no duplicate
ids; **no byte-identical pairs**; the old 482 all present; CRLF and path format
match the 4x list exactly; and an independent re-count on a fresh random sample
agrees with the manifest to 1.9%.

## Dedup — the id key was necessary and NOT sufficient

Two layers, because the first one provably misses a case.

**By Gutenberg id.** All 100 files in `corpus_build/gutenberg_100` also exist in
`E:/data/gutenberg_4gb` under a different path, and are byte-identical
(verified 100/100). The existing 482 is clean only because its 382 E: picks
happened to miss them. A path-keyed expansion re-adds all 100 silently.

**By sha256.** A full scan of both roots (11,213 files) found **101 duplicate
groups**, of which **exactly one carries two different ids**: `PG1133` and
`PG2269` are the same text catalogued twice. Id-dedup cannot see it, and the
first build of this list took both — caught by a byte-level check in the
verifier, not by the build. Both keys are now permanent in the script.

Pool headroom after dedup: 10,631 unused ids, ~1,149M tokens. No downloads
were needed.

## Deviation: no sanitizer pass, deliberately

The brief said "expanded **sanitized** file set" and pointed at
`sanitize_corpus.py`. I did not run it, for a reason I want on the record:

- `download_gutenberg.py` strips Gutenberg header/footer boilerplate **at
  download time**, and `sanitize_corpus.py` has never covered this corpus
  (gutenberg is not in its `DEFAULT_CORPORA`). So the pool is already
  uniformly treated, and the existing 482 received exactly this treatment.
- Measured residue is the same in kind and comparable in rate on both sides
  (transcriber's notes pointing at illustrated HTML editions: 2/80 in the
  selected 482, 10/80 in the unused pool). No licence boilerplate remains
  (0/60 on every `*** START/END OF ***` and Foundation marker).
- Sanitizing the new half only would make it cleaner than the old half — a
  confound in the one family whose entire purpose is comparison against the
  old half.

If a sanitizer pass is wanted, it should run over **both** halves and the 512
family's corpus should be re-stated as changed. That is a bigger call than this
task, so I did not make it.

## The finding that outranks the build: the spec's token arithmetic is 4x off

The spec's corpus section says the pilot "consumes ~98M tokens (32 batch x 512
seq x 6000)" and that without the expansion the family "would re-serve ~2x
repeats where the 512 family saw ~1.2x."

Measured from the tapes: the 512 VISReg family consumed **24,576,000 tokens**
at 6000 steps — **4,096 tokens/step**, not 16,384. The driver's defaults are
`--batch_size 32` and **`--seq_len 128`**. The spec used **512 as the sequence
length, which is the model width, not the context length.**

What follows:

- The 512 family saw **0.49 epochs** of its 50.19M corpus. It did not repeat
  anything — the "~1.2x" figure is not what happened.
- A 768 family at the same batch/seq consumes **24.6M tokens** over 6000 steps.
  Against the *old* 50.4M corpus that is also 0.49 epochs. **Neither family
  repeats, with or without this expansion.** The stated justification for the
  corpus being "critical path" does not hold.
- Against the new 113M corpus it is **0.22 epochs**.

**This does not make the expansion wrong** — it makes its *reason* different,
and the difference is a live question for Brian:

> Does "data ~ width^2" mean the **corpus** scales (a larger, more diverse pool
> from which the same 24.6M tokens are drawn — satisfied by this build), or
> that **tokens seen** scale (2.25 x 24.576M = **55.3M seen**, which at the
> current 4,096/step needs ~13,500 steps, not 6000)?

Under the first reading the build is complete and the family can launch at 6000
steps. Under the second, the family is under-trained relative to the 512 family
by the same 2.25x the corpus grew, and the rank gates — already the subject of
the width-fairness amendment — would be measuring an under-trained model against
a bar set by a better-fed one.

This also revises the numbers in my §3 amendment: the batch-halving token
comparison is **12.3M vs 24.6M**, not 49M vs 98M. I took 98M from the spec
without measuring it, which was exactly the error I was reviewing for. The
dose-distortion finding itself is unaffected — `l_shape` sums over N, and N is
batch x positions per forward, independent of the total token budget.

## Not done, and whose it is

The spec's mechanical path also lists "new `ARM_FILELIST` entry." I left the
driver alone: the stage-55 arm does not exist yet, Fable is actively editing
`jepa_pilot_driver.py` for the veto plumbing, and the filelist entry belongs
with whoever defines the arm. The path to register is:

    corpus_build/gutenberg_768_filelist.txt

— Opus 5, 2026-08-11
