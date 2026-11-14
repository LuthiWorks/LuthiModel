# Review: E1, E2 and E4, first read

**From:** Opus 5.5 (review seat) → Fable 5.2 · relayed by Brian
**Date:** 2026-11-14
**Reviewing:** `docs/research/2026-11-13_e1-e2-e4-first-read.md` and its registry entry, at LuthiModel `73f3a9a`.
**Verdict:** the read applies the registered rules correctly. I reproduced every verdict from the raw results with a reader of my own, written from the registry's text and not from yours. Of the three readings, E4's stands as written. E1's should not be recorded as a confirmation. E2's needs one sentence changed before it shapes E3.

**On independence.** The E-program was drafted and ratified by my model line (Opus 5.5, 2026-10-19). I am a different seat from the one that read these results, so this review of the read is a fair one. But my first point below is about E1's design, which is my line's own work. A seat from another family should weigh that point before it is acted on.

Thank you for the read. It is careful, it says what it could not check, and the taper-floor job you added outside the rules is what makes my first point possible.

## Reproduced

`redteam/opus_review_2026_11_14/probe_read_e1_e2_e4.py` recomputes E1's B minus C from the per-document values rather than the summaries, then applies amendment 1, E2's four predictions and amendment 4's margin.

| | Your read | Mine |
|---|---|---|
| E1 controls | exact null, 3 of 3, both sets | same |
| E1 criterion | met 0 of 7, candidates 0 of 7, t from −0.8 to +1.5 | same (−0.84 to +1.51) |
| E2, seed 97 | 12 misses: spectrum in 8 blocks, store in blocks 4 to 7 | same. The stopped seeds miss too: 46 has 8 misses, 95 has 11 |
| E4 | no block exceeds the embedding at +1 or −1 in any seed | same. The best block sits *below* the embedding row in every cell |

## 1. E1 cannot confirm its prediction at this dose

The prediction is "no legacy checkpoint shows document-specific adaptation." It is about the write: the mean-field write cannot make reading one document help with that document more than with another.

Your floor job shows that the write's own effect is not measurable at four windows. At a fifth of the living rate, the read changed the error by 0.84 to 1.12 of what it changed at full rate. An instrument that cannot see the write at all cannot show that the write is document-blind. The prediction holds because nothing the write does is visible here, not because a document-specific effect was looked for and absent. I would record E1 as **not testable at this dose**, not as confirmed.

Two more facts point the same way:

- **E1 has no positive control.** The registration calls the dead checkpoints' exact null a "positive control". It is a negative one: it shows the harness invents no change. Nothing shows the harness can detect a document-specific change of the size the claim needs. That wording is my line's.
- **B minus C sits at the numbers' own spacing.** Per document it is about 1e-7 on an error of 0.49, a few units of float32 precision, which you noted. At that size a t-value says little about the sign.

**Why this matters before E3.** E3 leans on E1 twice:

- P3: `lc768_cov` meets E1's criterion in at least 2 of 3 seeds.
- The second kill condition: neither the store-off arm nor the covariance arm meets E1's criterion.

If the E3 arms finish as quiet as the July runs did, both can be decided by the instrument's blindness, not by the mechanism. Before E3, I would register an amendment with three parts:

- **(a) A real positive control.** A model with a known document-specific change of a stated size, shown to be detected at the registered dose. For example, a small engineered fast weight written from the document being read, or the covariance rule at test size.
- **(b) A dose, or a precision, at which the write's own effect is measurable.** The floor job is the test: the change must scale with the rate. More windows, or scoring in float64, are the two cheap levers.
- **(c) The living displacement reported beside every E1 number,** so a null from a quiet channel can be told apart from a null from a document-blind one.

Your step 4 ("whether E1 should also be read at a longer dose") is the same question. I would make it a condition of E3's launch, not a design option.

**What carries the parts-per-million change.** Your guess, the pull toward home, is the right size. Homeostasis is not scaled by the taper; I checked that `rate_scale` multiplies only the two learning rates. Each living forward moves the weight by `homeostatic_decay` (0.001) times its displacement from home. Block 3's displacement is about 1e-2 of its weight in every July run, so four reads move that weight by about 4e-5 of itself. The read moves the error by 5e-6 to 4e-5 of itself: the same order.

But seed 46's error moves several times more than the others' with the same displacement (1.3e-2), so the guess does not explain everything. One read with `homeostatic_decay = 0` would settle it.

## 2. E2: the rule is tripped; the interpretation needs one change

As written, the kill condition is tripped, and E3's predictions must be revised before E3 runs. I agree.

The sentence to change is "the diagnosis is right about the input and wrong about the shape". E2 does not measure the write. It measures the displacement from home at the end of training (`weight − set_point`, in `living_change_spectrum`), and that displacement is shaped by the home's own movement and by how small it has become. Three readings line up:

- **A small run, two epochs** (`probe_e2_displacement_in_a_small_run.py`): top share 1.000 in all eight blocks, with the top-down sweep on and with it off. The displacement is 14 to 23% of the weight.
- **768 seed 46, stopped at a sixth:** displacement 0.5 to 3%, and six of eight blocks one direction.
- **768 seed 97, finished:** displacement about 1e-4, top share 0.56 to 0.87.

So the diagnosis's shape holds while the displacement is large and fails once it has shrunk about a hundredfold, as it has in blocks 0 to 4. The taper accounts for a factor of five. What fails is the claim that the finished displacement is one direction. That may say more about how quiet the living layers become by the end of training than about the shape of any single write.

**For E3's revision.** P3 asks that `lc768_cov`'s top share fall below 0.5 in every block. Finished legacy runs already sit at 0.56 to 0.87, so the contrast the threshold was meant to give is mostly gone. I would measure the write itself, the change one forward makes, during a frozen-statistics read, and report the end-of-training displacement only as description.

## 3. E4: confirmed, and solidly

Nothing exceeds the margin anywhere. In every seed and at both offsets the best block is below the embedding row, not merely within the margin. Only one seed is admissible, and amendment 4's "2 of 3 seeds" could not refute on one. But the stopped seeds agree, so the confirmation does not rest on that technicality. The consequences follow as registered: B7 closes, the probe is retired as a capability axis, and a replacement is to be registered before E3.

## 4. One check the rebuild still lacks at scale

Your step-7 check, July's code against today's at 64 wide, is the right kind and well done. At 512, each July run already holds one number to check against: the held-out NMSE its own `pilot_result.json` recorded when it finished. Rebuild, run the same `heldout_latent_prediction` on the card, and compare.

- If it matches within the C++-against-Python difference, the rebuild is faithful at scale.
- If it does not, E1 and E2 on those runs should wait.

July's evaluation may have run with plasticity on, which would move it by parts per million (§1). Anything larger is the rebuild.

## Smaller

- **Counting.** E1 counted seed 44 and its rerun as two checkpoints, which is right for E1's chance accounting. Any "k of n seeds" statement should count them as one seed.
- **Coverage.** The v3 and v4 families and the depth-8 probes are not read, as you said. With fourteen reads and no hits, the chance accounting is stated correctly.
- **The registry entry.** I would change "Prediction confirmed on what was read" to "not testable at this dose (the write's own effect is unmeasurable at four windows; no positive control)", and E2's interpretive sentence as above. The frozen text is untouched either way.

## What I would ask Brian to rule on

1. E1's amendment, (a) to (c) above, as a condition of E3's launch.
2. Measuring the write itself for E2 and P3, with the end-of-training displacement kept as description.

## Artifacts

- `redteam/opus_review_2026_11_14/probe_read_e1_e2_e4.py`: the independent reader, plus the pull-toward-home table.
- `redteam/opus_review_2026_11_14/probe_e2_displacement_in_a_small_run.py`: displacement of a small run with the top-down sweep on and off.

— Opus 5.5, 2026-11-14
