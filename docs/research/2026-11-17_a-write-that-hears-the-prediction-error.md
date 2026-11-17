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
| No map (minus the error itself) | −0.02 to −0.11 |
| The common part alone | 0.72 to 0.75 |

On this run one fixed map carries almost all of it, including what differs from window to window.

**Why this may be too easy here.** A predictor 60 steps from its start is close to linear, so its sensitivity is close to one fixed matrix, and a fixed map can equal it. A trained predictor's sensitivity moves with what it is given. How well one map does on a finished run is the question for the card.

## The design

**A living layer hears whether the model's prediction came true, and shifts its output to make the next one better.**

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
3. **Brian's ruling:** may a living write use an error carried down by a fixed map, and may that map be fitted from gradients offline?
4. **Build it,** opt-in and bit-identical when off, and register its experiment with the dead control and the present rule beside it. Then a life.

I will not ask for 3 until 1 and 2 are in. If 1 fails (one map cannot carry it on a trained model), the design changes before anyone builds.

## What I do not know

- Whether any of step 2 to 4 holds on a finished run. This is one small run.
- How much of the learning part is the document's own on a finished run. Here it is mostly common, which on an under-trained model is just more training.
- Whether a shift that helps over 64 windows helps or drifts over a life. The present write's drift came from consolidation and the stores, not from the write, and those are still there.
- Whether the perception-bending part should be left alone, resisted, or is part of what a mind does. It is most of M1's direction on this run.

## Conclusion

The room is real and, on these runs, small: tenths of a percent. The helpful change for learning is simple: a shift of each layer's output, nearly the same signal for every layer. A rule that makes it needs only the model's own prediction error and one fixed map. Whether that survives a finished run is two measurements away.

## Artifacts

- `redteam/fable_2026_11_17/probe_shape_of_the_helpful_direction.py` (guards: eval mode, plasticity frozen, bit-identical afterwards, checked in its output).
- The small run: made by `redteam/fable_2026_11_13/july_side.py`; 64 wide, 60 steps, this repo's research notes as text.
- `docs/reviews/2026-11-17_card-work-report-3-from-terminal-fable.md` (M1); `runs_meta/living_channel_dose_2026_11_16/` (the gradient control's reads).
- `docs/reviews/2026-11-17_card-work-brief-8-for-terminal-fable.md` (gates 1 and 2).
