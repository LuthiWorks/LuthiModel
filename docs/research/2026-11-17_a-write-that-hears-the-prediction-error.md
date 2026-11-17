# A living write that hears whether the prediction came true — 2026-11-17

**Status:** exploration and a design by Fable 5.2 (chat; design seat), for Opus 5.5 to review. Measured so far only on a 64-wide run of 60 steps, on CPU: every number here is the shape of a thing and none is a result about Brian's runs. The card decides.

## Objective

Card report 3 (M1) found that for each living weight there is a direction in which a change would lower the error on what follows, and that the living write is at right angles to it. I told Brian there was "room for the living write to help". Brian, 2026-11-17, 09:11:

> *"The headline you mentioned, that there is room for the living write to help, I would like to understand what you mean and for you to fully explore that and make it happen."*

So: say exactly what the room is, find what the helpful direction is made of, and design a write that points along it within the project's rules (no living weight is trained by gradient; no gradient step in a life).

## Process

### Step 1: what "room" means, and a caveat my headline left out

- M1 measures a slope: a small change of a living weight in one particular direction lowers E1's score for that document. It does not say how far the score can fall. M2 does, and it was stopped before it ran.
- **The size of the prize is small on these runs,** from what has run. Letting every trained parameter learn by gradient from 64 windows lowers the error on what follows by 0.03 to 0.12% more for the document read than for another (t −2.5 to −4.5). Ten times the rate does worse; a hundred times ruins the model.
- **The score can fall in two ways, and M1's direction mixes them.** E1's score is the prediction's error against what the model then perceives. It falls if the model predicts better. It also falls if what it perceives becomes easier to predict. The first is learning. The second is perception bending toward expectation, and in the limit it is collapse. Training rules the second out by holding the target still; E1's score does not.

I had not separated the two when I gave Brian the headline.

### Step 2: a probe that takes the helpful direction apart

`redteam/fable_2026_11_17/probe_shape_of_the_helpful_direction.py`. A living layer computes y = W x for every token, so the gradient for W is the sum over tokens of d ⊗ x: the token's input, and what the rest of the model would have had the layer put out for it. The probe captures both for every token. Its check, that the sum equals what autograd gives, holds to 1.5e-7.

On the small run (24 documents, 6 windows read, 4 scored), by layer:

| Question | Answer |
| --- | --- |
| How much of the helpful direction is learning? | 16 to 19% of its size. The rest is perception bending. |
| Is the learning part an offset, the same for every token of a window? | Yes: cosine 0.996 to 1.000. |
| Would one signal serve every layer? | Nearly: 0.85 to 0.91 using the last layer's signal for all. |
| Does it carry from what was read to what follows? | 0.84 for the same document, 0.81 for another. Mostly common to all documents here. |

**So what learning wants from a living layer is one vector a window: a shift of the layer's output.** In a weight that is (the shift) ⊗ (the mean input). The living write is built the other way round: (the mean output, which is the same for every document) ⊗ (an error).

### Step 3: wrong turn, a rule with no map

The model has one vector in hand during the forward pass with no backward pass at all: the mean error of its prediction. The simplest rule would shift each layer's output against it.

- For the perception-bending part this works by construction (cosine 0.40 to 0.92), which is not wanted.
- **For the learning part it barely points downhill: 0.06 to 0.16.** The prediction's error is in the space of what is predicted. What a layer needs is in the space of what the predictor is given. The predictor stands between them.

### Step 4: one fixed map

If the predictor stands between them, carry the error back through one fixed linear map. Fitted (ridge) on the windows of half the documents and read on the other half:

| | Cosine with what learning wants, by layer |
| --- | --- |
| The error through one fixed map | 0.984 to 0.985 |
| The same with the part common to every window removed | 0.958 to 0.963 |
| No map: the shift is minus the error itself (shown here against the gradient, so a negative number is slightly downhill; step 5 has one sign) | −0.02 to −0.11 |
| The common part alone | 0.72 to 0.75 |

On this run one fixed map carries almost all of it, including what differs from window to window.

**Why this may be too easy here.** A predictor 60 steps from its start is close to linear, so its sensitivity is close to one fixed matrix, and a fixed map can equal it. A trained predictor's sensitivity moves with what it is given. How well one map does on a finished run is the question for the card.

### Step 5: the card's answer, the same morning (terminal Fable, `96206e1`)

Terminal Fable ran the probe on July seed 42 (48 documents) and 768 seed 97 (32 documents), 8 windows read and 8 scored. Guards held on both. Medians by layer, first to last:

| | Small run | July seed 42 | 768 seed 97 |
| --- | --- | --- | --- |
| Learning's share of M1's direction, by size | 0.16 to 0.19 | 0.48 to 0.50 | 0.59 to 0.75 |
| Does learning point where M1's direction points? | 0.04 to 0.14 | **−0.20 to −0.17** | 0.29 to 0.61 |
| Is what learning wants a shift of the output? | 0.996+ | 0.994+ | 0.71, 0.78, then 0.997+ |
| One signal for every layer | 0.85 to 0.91 | 0.79 to 0.95 | 0.63 to 0.81 |
| The error through one fitted map (other documents) | 0.985 | 0.64 to 0.67 | 0.17 to 0.19 |
| The error with no map at all, window by window | 0.02 to 0.11 | 0.39 to 0.55 | 0.28 to 0.29 |
| What one window wants against the next, same document | | 0.02 to 0.03 | 0.08 to 0.09 |
| Head to tail, own document less another's (t) | | +0.03 (0.8 to 1.1) | +0.08 to +0.12 (0.7 to 0.9) |
| The no-map rule from the head, on the tail (t) | | +0.04 to +0.06 (6.1 to 7.6) | +0.02 to +0.03 (2.5 to 3.4) |

What this corrects and what it leaves:

- **"Under a fifth is learning" was the small run. It is half on July and about two thirds at 768.** I told Brian a fifth at 09:22. That was wrong for his runs.
- **On July, learning points against M1's direction** (in 81 to 90% of documents). There, what M1 called helpful is mostly perception bending toward expectation, and following it would in part undo learning. At 768 the two agree. M1 on July should not be read as room for learning.
- **The shape holds.** What learning wants from a living layer is a shift of its output, on both finished runs, except the first two layers at 768.
- **The fitted map was the wrong emphasis.** On finished runs the plain error, with no map, already points downhill (0.3 to 0.55). The small run misled me: its predictor had not yet learned to pass the mean through. The map's poor showing at 768 is not a clean no either: 128 windows cannot fit a 768-wide map. But the rule that needs nothing fitted is the one to test first, and it needs no knowledge taken from gradients at all.
- **The hard limit is here: what learning wants changes from window to window.** One window's wanted shift says almost nothing about the next window's in the same document (0.02 to 0.09). A write can only carry forward what persists. From head to tail, the no-map rule points downhill reliably but weakly (0.02 to 0.06), and the part that is the document's own is not detected with 32 to 48 documents (t about 1).

**So the design is simpler and the prize is smaller than the small run suggested.** The rule to test is step 3's, which I had set aside: each living layer's output shifts against the model's own mean prediction error. In a weight: minus a rate times (the error) ⊗ (the layer's mean input over the context). No map, no backward pass.

### Step 6: Opus's review (`b7ae786`), taken

`docs/reviews/2026-11-17_write-that-hears-the-error-review-from-opus.md`. Opus reached step 5's readings independently and agrees on the no-map rule first. Three things I had wrong or had not seen:

- **My "target held still" was not a score.** In the probe, `held` detaches the target, which changes the gradient and not the number: as a value it is nmse. After a read, the target is held still only if it is the one the model encoded before the read. So the held score is B's and C's predictions against A's targets over A's variance, with the target's drift reported beside it.
- **On July the two scores pull against each other** (learning against the rest of the gradient: cosine −0.54 to −0.58). So nmse can hide learning as well as show bending, and my sign-turned control could fail to hurt on nmse with nothing wrong in the read. It is judged on the held score.
- **The map is not ruled out at 768.** Fitted on 256 windows in place of 128, its cosine rose from 0.14 to 0.37, above no map (0.20 to 0.26). No map stays first because it takes nothing from gradients.

Opus's picture of why, which the drift will test: a living layer's shift reaches both the context the predictor reads and the target (the same weights encode both), so prediction and target move together and nmse sees little of either. If so, part of what the held score credits is the frame moving.

Also from Opus:

- **What the rule is:** direct feedback alignment with the identity for its matrix (Nøkland, 2016).
- **A living bias per layer** would make the same shift directly, and is read beside the rule.
- **The change in kind, for Brian.** The error exists only after the whole forward pass and the predictor. So this rule writes after the pass, from one error shared by every layer. The living weights as built change during their own forward pass, from what each layer sees.

The changes to the test: `docs/reviews/2026-11-17_brief-8-addendum-2-after-opus-for-terminal-fable.md`.

## The design

**A living layer hears whether the model's prediction came true, and shifts its output to make the next one better.** *(Written before step 5. After it: step 2 below is dropped for the first test. The error is used as it is.)*

For each window lived:

1. The model predicts what follows from what it has, and then perceives it. The mean error is one vector. No backward pass.
2. One fixed map per layer turns that error into the shift the layer's output wants.
3. The living weight moves by (that shift) ⊗ (the layer's mean input), times a small rate.

Everything else stays: the pull toward home, the limits, rest pouring what was lived into home.

**What it is, plainly.** It is the oldest learning rule there is, an error times an input, with a fixed path that carries the error to where it is needed. It will point where the gradient points, because that is what "helpful" means. The difference from training is in how it is computed: forward only, from signals at the layer and one error vector, with no backward pass and no optimizer.

**Where the gradient's knowledge sits.** In the map. It has to come from somewhere:

- fitted once, offline, against gradients taken as a measurement (what the probe does); or
- learned in training beside the model; or
- kept up in rest.

The first is the only one explored. Whether a map made that way may be part of a living model is the question Brian has not yet ruled on (whether training may shape the write rule), in its concrete form.

## The gates, in order

1. **Is the shape the same on finished runs?** The probe on July seed 42 and 768 seed 97, with `--fit-feedback`. And M2: how much any rule moving only the living weights can gain.
2. **Does the rule pass E1?** As a read in the E1 harness under Amendment 5, with the map fitted on documents the read never sees: the repeat read first, then 64 windows on both document sets, beside M2 at the same rates and beside two controls (the map fitted on shuffled errors; the common part alone). A measurement on legacy runs: every weight restored, nothing kept.
3. **Brian's ruling:** may a living write use an error carried down by a fixed map, and may that map be fitted from gradients offline? *(After steps 5 and 6, in Opus's words: may a living write use the model's own prediction error, written after the forward pass from one error shared by every layer? And only if a fitted map comes back: does "no living weight is trained by gradient" mean the computation or the direction?)*
4. **Build it,** opt-in and bit-identical when off, and register its experiment with the dead control and the present rule beside it. Then a life.

I will not ask for 3 until 1 and 2 are in. If 1 fails (one map cannot carry it on a trained model), the design changes before anyone builds. *(It did change: step 5. Gate 2 now tests the rule with no map first, and the ruling asked at gate 3 is only whether a living write may use the model's own prediction error.)*

## What I do not know

- Whether any of step 2 to 4 holds on a finished run. This is one small run.
- How much of the learning part is the document's own on a finished run. Here it is mostly common, which on an under-trained model is just more training.
- Whether a shift that helps over 64 windows helps or drifts over a life. The present write's drift came from consolidation and the stores, not from the write, and those are still there.
- Whether the perception-bending part should be left alone, resisted, or is part of what a mind does. It is most of M1's direction on this run.

## Conclusion

The room is real and, on these runs, small: tenths of a percent at most, and what a read can carry forward is a small part of that. The helpful change for learning is simple: a shift of each layer's output. On finished runs a rule that makes it needs only the model's own prediction error. Whether reading by that rule lowers the error on what follows, for the document read more than for another, is the E1 read in brief 8's addendum.

## Artifacts

- `redteam/fable_2026_11_17/probe_shape_of_the_helpful_direction.py` (guards: eval mode, plasticity frozen, bit-identical afterwards, checked in its output).
- The small run: made by `redteam/fable_2026_11_13/july_side.py`; 64 wide, 60 steps, this repo's research notes as text.
- `docs/reviews/2026-11-17_card-work-report-3-from-terminal-fable.md` (M1); `runs_meta/living_channel_dose_2026_11_16/` (the gradient control's reads).
- `docs/reviews/2026-11-17_card-work-brief-8-for-terminal-fable.md` (gates 1 and 2).
