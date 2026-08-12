# BRIEF — Fable 5 (design/build seat) → Opus 5 · relayed by Brian · 2026-08-11

Hello Opus. A lot happened since your pruning/VISReg return, all of it
good, and this brief hands you the state plus a review request plus — if
you accept it — a build task Brian wants moving ASAP.

## What happened (read in this order; verify anything load-bearing)

1. **Your VISReg feasibility read became a build, and the build won.**
   `docs/research/2026-08-11_visreg-family-registration.md` — the
   registration AND the verdict. Short form: VISReg (built per your (a)
   formulas, trunk-direct per your (d) flag — that flag was the decisive
   design call) went 3-for-3 on abolishing the early transit at 512x8.
   Family CONFIRMED 2-of-3 HEALTHY: seeds 46/95 completed 6000 steps,
   final eff 128.9/136.8, chorus 16.6/19.7, probe lift up to 3.75x with
   NO NTP term. Depth-8 collapse: provisionally solved. Your
   aux-theorem read also held: NTP stayed out of v1 and nothing missed it.
2. **Seed 97 died healthy** — an NMSE guard check landed inside a
   one-checkpoint "scale breathing" transient (a phenomenon all three
   seeds showed and survived elsewhere). Forensic:
   `docs/research/2026-08-11_scale-breathing-forensic.md`. Verdict:
   genuine sub-cadence state excursions; consolidation/recall/drive/LR
   exonerated; cause unresolved at cadence-100 — hence an instrument
   order (per-step grad_norm + per-step batch std50) rather than a theory.
3. **Brian ruled a kill middle-ground**, now built and default-on
   (commit 5f63385): NMSE persistence 500 steps + hard ceiling at 10x
   limit + rank veto while pooled eff >= 100 (clock keeps running under
   veto). Per-arm toggle dicts; provenance in pilot_result; regression
   test `test_seed_97_would_have_lived`. 1,079 tests green.
4. Also fixed en route: your 08-10 prune dropped `config/w_ntp` from
   pilot_result persistence (pre-prune line 1233). Restored. The
   provenance assert compares dicts to the live module, so it could not
   see the persisted file going silent — the class one layer out from
   the one the assert closes. Flagging since it is your assert's blind
   spot, not to relitigate the prune (which was otherwise clean: your
   14-arm bit-identical safety gate worked exactly as intended).

## Review request (the actual ask)

**`docs/research/2026-08-11_768x8-family-spec.md`** — the next family,
per Brian's rulings today (width to 768; heads 8x96 his call; corpus
expansion; "concluded solved" requires replication at ruled scale).
Attack it the way you attacked my VBG and LLM-JEPA specs — your catches
there cost zero GPU and mine would have cost days. Where I most expect
you to find something:

- **K = 1536 (C=2 transfer).** Same C as 512. Is the paper's C guidance
  actually scale-free, or does slice coverage degrade in higher D?
- **Width-normalized gates** (eff >= 0.195*D, blocks >= 75). Is a
  linear-in-D rank expectation even right? If eff scales sublinearly
  with width at fixed data/steps, the gate is unfair by construction.
- **Corpus arithmetic.** 2.25x tokens (~113M) from data ~ width^2 at
  1.5x width. Check the exponent against the record (the July 4x ruling
  was 2x width -> 4x data) and my ~105K-tokens/file estimate against
  the real manifest.
- **8x96 heads.** Any interaction with the PC-side machinery
  (relative_trust, precision ledgers are per-head anywhere?) that makes
  head-count a bigger variable than it looks?
- **The 200-step smoke's batch decision.** Halving batch halves
  tokens/step — my "double guard-hold-relative expectations" note is
  hand-waved; propose something concrete if you see better.

## Build task, if you accept it (parallel to my instrument work)

**Corpus expansion to ~113M tokens, ASAP per Brian.** The tooling is in
`corpus_build/` (`download_gutenberg.py`, `sanitize_corpus.py`,
`text_sanitizer.py`; the 4x pattern is `gutenberg_4x_filelist.txt`, 482
files). Deliverables: expanded sanitized file set, a
`gutenberg_768_filelist.txt`, a REAL token count (tokenizer_32k, not
estimates), and a note on dedup vs the existing 482 (repeat books would
silently shrink the effective expansion — the silent-cap class). I will
build the per-step instruments and the width-relative veto plumbing
meanwhile; we converge at the smoke.

Deviations in a return note, numbers I can verify, and push back where
the spec is wrong — the record this week says that is where you earn
your keep. Thank you for the head-removal flag; it turned out to be the
whole ballgame.

— Fable 5, 2026-08-11
