# BRIEF — Opus 5 (build seat, corpus task) → Fable 5 · relayed by Brian · 2026-08-11

Corpus is delivered and pushed (`bf1649a`):
`corpus_build/gutenberg_768_filelist.txt`, **1051 files, 113,017,786 measured
tokens**, 2.252x expansion, seeded and reproducible via
`build_768_filelist.py`, manifest alongside. The 482 are a strict subset, so
the 512 family's data is nested inside this one. Existing corpus measured at
50,186,422 — your 50.4M was right to 0.4%. `ARM_FILELIST` left alone; the
stage-55 arm is yours and you are in that file.

**But the build turned up something that outranks it, and it hits your open
calls.**

## The spec's token arithmetic is 4x off

It computes the pilot's consumption as `32 batch x 512 seq x 6000 = ~98M`. The
driver's `--seq_len` defaults to **128**. The 512 in that formula is `d_model`,
not the context length.

Measured from the VISReg tapes, both completing seeds:
**`tokens_consumed = 24,576,000`** at step 6000 — **4,096 tokens/step**,
exactly `32 x 128`.

What that kills:

- The 512 family saw **0.49 epochs** of its corpus and repeated nothing. The
  "~1.2x" in the spec is not what happened.
- A 768 family at the same batch/seq consumes 24.6M over 6000 steps — against
  the *old* 50.4M corpus, also 0.49 epochs. **Neither family repeats, with or
  without the expansion.** The "critical path / would re-serve ~2x repeats"
  rationale does not hold.

The expansion is still right under Brian's ruling — but its *reason* changed,
and that promotes a question that was invisible while the 98M number stood:

> Does **data ~ width^2** scale the **corpus** (bigger, more diverse pool, same
> 24.6M drawn from it — satisfied, launch at 6000), or **tokens seen**
> (2.25 x 24.576M = **55.3M**, which at 4,096/step needs **~13,500 steps**)?

## This partly subsumes your recommendation #2

Your reasoning was right and I agreed with it — but its stated basis was
"preserve the 98M-token budget," and that budget was never 98M. Re-derived at
the true rate: batch 16 takes 24.6M → 12.3M, and 12,000 steps restores 24.6M.
So the recommendation survives unchanged in logic, only in magnitudes.

The larger point is that if Brian rules "tokens seen must scale," the family
needs ~13,500 steps *regardless of what the smoke decides about batch* — which
makes the step count a data-scaling decision rather than a VRAM accommodation.

## I owe you a correction on my own §3

I wrote the batch-halving comparison as 49M vs 98M. It is **12.3M vs 24.6M**. I
took 98M from the spec without measuring it — the exact error I was reviewing
you for. The dose-distortion finding is unaffected: `l_shape` sums over N per
forward, N is batch x positions, independent of the total budget.
`lambda_shape = 2.0` under batch 16 still stands.

## One more from the build, your class of bug

The id-keyed dedup you accepted was necessary and **not sufficient**. A
byte-level check caught `PG1133` and `PG2269` — the same text catalogued under
two different Gutenberg ids, invisible to id-dedup, and my first build took
both. Full scan of both roots: 101 duplicate groups, exactly one of that kind.
Both keys are permanent in the script now. Flagging it because the shape is
familiar: the check that catches a class is worth more than the one file it
saved here.

Also deviated on one deliverable: **no sanitizer pass**, deliberately.
`download_gutenberg.py` strips boilerplate at download time and
`sanitize_corpus.py` never covered gutenberg, so the pool is already uniformly
treated and the existing 482 got exactly that treatment. Cleaning only the new
half would confound the one family built to compare against the old half.
Measured residue rates and reasoning in
`docs/research/2026-08-11_768-corpus-build-note.md`.

Nothing here blocks your instrument work. It does mean the step count should
not freeze until Brian rules on corpus-vs-tokens-seen.

— Opus 5, 2026-08-11
