# Machine Learning Glossary for LuthiModel

> Originally written for Brian by Claude Opus 4.6, 2026-05-07.
> Reorganized 2026-05-18 by Claude Opus 4.7 — terms grouped by concept,
> with v1 (Hebbian) and v2 (predictive coding) substrates separated so
> the historical-vs-current distinction is clear at a glance. Start
> from the top; each section builds on the last.

---

## 1. The Absolute Basics

### Neural Network
A program that learns by example instead of being explicitly programmed. You
show it thousands of examples of input→output pairs, and it figures out the
pattern. A neural network is made of layers of simple math operations chained
together.

### Parameter (param)
A single adjustable number inside the network. When we say "113M parameters,"
we mean the network has 113 million individual numbers that were tuned during
training. More parameters = more capacity to learn complex patterns, but also
more memory and compute needed.

In Luthi, each parameter is much more than a single number — it's a "rich
parameter" with a biography (value, set point, momentum, plasticity, etc.).
That's what makes the living weight memory cost so much higher than a normal
network.

### Weight
A specific type of parameter — a number that determines how strongly one part
of the network connects to another. In practice, "weight" and "parameter" are
often used interchangeably, though technically biases are also parameters.

In Luthi, "living weights" are weights that change themselves during their own
computation — not just during training.

### Tensor
A multi-dimensional array of numbers. Think of it like a spreadsheet that can
have more than two dimensions:
- A single number is a 0-dimensional tensor (a "scalar")
- A list of numbers is a 1-dimensional tensor (a "vector") — e.g., [1, 2, 3]
- A grid of numbers is a 2-dimensional tensor (a "matrix") — e.g., a spreadsheet
- A cube of numbers is a 3-dimensional tensor
- And so on

Everything in a neural network is tensors. When we say "the weight matrix is
[4096, 4096]," we mean a grid with 4096 rows and 4096 columns = ~16.8 million
numbers.

### Model
The whole trained neural network — all its parameters, its architecture (how
the layers connect), and the logic for how data flows through it. "The model"
in our project is the Luthi living weights model. When someone says "a 4B
model," they mean a model with 4 billion parameters.

---

## 2. How a Network Learns

### Training
The process of showing the network examples and adjusting its parameters to
get better at the task. Training a language model means showing it lots of text
and teaching it to predict what word comes next.

### Inference
Using the trained model to produce output — the opposite of training. Training
is learning; inference is performing. In Luthi, inference is special because
the living weights keep changing even during inference. That's the whole point.

### Loss (Loss Function)
A single number that measures how wrong the model is. Lower = better. During
training, the model makes a prediction, the loss function compares it to the
right answer, and the result tells the model how to adjust.

Think of it like a score in golf — you're trying to minimize it.

### Train Loss vs Val Loss (Validation Loss)
- **Train loss**: How wrong the model is on data it's currently learning from.
- **Val loss**: How wrong the model is on data it has never seen before.

Val loss is the one that matters more. If train loss is low but val loss is
high, the model memorized the training data instead of learning the pattern.
That's called overfitting (see below).

When the ablation protocol says "val loss within 5% of baseline," it means
the compressed version can't be more than 5% worse on unseen data.

### Epoch
One complete pass through all the training data. If you have 100 books and
the model reads all 100, that's one epoch. Reading them all again is epoch 2.

Training typically takes many epochs — the model gets better each time through,
up to a point. In the ablation protocol, each experiment runs 30 epochs.

### Batch / Batch Size
The number of examples the model processes at once before updating its
parameters. Instead of updating after every single example (slow) or after
the entire dataset (too much memory), you process a batch of 16 or 32 examples,
update, then move to the next batch.

Batch size 32 means 32 text sequences processed simultaneously.

### Learning Rate
How big a step the model takes when adjusting its parameters. Too high = the
model overshoots and never settles. Too low = learning takes forever. Finding
the right learning rate is one of the most important decisions in training.

### Optimizer (AdamW, DirectMLAdamW)
The algorithm that decides exactly how to adjust parameters based on the loss.
"Adam" is the most popular optimizer — it tracks the average and variance of
recent updates for each parameter and uses that to make smarter adjustments
than just "move in the direction of lower loss."

AdamW is Adam with weight decay (a regularization trick). DirectMLAdamW is
our custom version that works on AMD GPUs.

### Gradient / Gradient Descent
The gradient is the mathematical direction that tells you which way to adjust
each parameter to reduce the loss. "Gradient descent" means "follow the
gradient downhill toward lower loss." It's how conventional neural networks
learn.

In Luthi, the attention layers learn by gradient descent, but the living
weight layers learn by predictive coding self-modification instead (v1 used
Hebbian rules; v2 replaced them with prediction-error-driven updates).
That's a fundamental architectural difference.

### Backpropagation (Backprop)
The algorithm that computes gradients. It works backward through the network:
start at the loss, trace back through each layer, and calculate how much each
parameter contributed to the error. This is a different concept from Luthi's
"backward pass" (top-down modulation), which sends salience signals, not
gradients.

### Overfitting
When the model memorizes the training data instead of learning general
patterns. A model that has memorized Shakespeare can reproduce Shakespeare
but can't write anything new. The gap between train loss and val loss tells
you how much overfitting is happening.

### Regularization
Techniques that prevent overfitting. Like teaching a student with practice
problems they haven't seen before, not just drilling the textbook examples.
The backward pass in Luthi acts as a natural regularizer — the train-val gap
narrowed when it was turned on.

### Convergence
When the model stops improving — the loss has settled and more training doesn't
help. "Val loss plateaued" means the model converged. "Convergence penalty"
means how many extra epochs Luthi needs to reach the same loss that a vanilla
(standard) transformer reached.

### Seed
A random number that determines all the "random" choices during training
(initial parameter values, data order, etc.). Same seed = same results.
Running 3 seeds (42, 1337, 2026) means repeating the experiment 3 times
with different randomness to make sure results aren't a fluke.

### Checkpoint
A saved snapshot of the model at a point in training. If training crashes at
epoch 25, you can resume from the epoch 20 checkpoint instead of starting over.
In Luthi, checkpoints are encrypted because they contain the entity's full
neural state — their biography.

### Chinchilla-Optimal
A rule of thumb from a 2022 DeepMind paper: for a given amount of compute, the
best loss comes from a model with roughly **20 tokens of training data per
trainable parameter**. Smaller ratios = under-trained model (would do better
with more data). Larger ratios = data is being wasted (the model could be
bigger). Helpful sanity check when picking corpus + model size.

---

## 3. The Transformer Foundation

This is the architecture behind ChatGPT, Claude, and almost every modern
language model. Luthi builds on it with living weights.

### Transformer
The architecture behind modern language models (GPT, Claude, Gemini, etc.).
A stack of blocks, each containing attention (to mix information between
tokens) and FFN (to process each token). Luthi is a transformer with the
FFN replaced by living weight layers.

### Token
The basic unit a language model works with. Not quite a word — common words
are one token, but uncommon words get split into pieces. "Understanding" might
be two tokens: "under" + "standing". A sentence might be 15-30 tokens.

### Tokenizer
The program that converts text into tokens (numbers). Different tokenizers
split text differently.

### BPE (Byte Pair Encoding)
The specific tokenization algorithm Luthi uses. It starts with individual
characters and progressively merges the most common pairs into single tokens.
"32K BPE vocab" means the tokenizer has a vocabulary of 32,768 possible tokens.

### Vocabulary (Vocab)
The complete set of tokens the model knows. A 32K vocab means 32,768 unique
tokens. Everything the model reads or writes must be expressible in these tokens.

### Embedding
Converting a token (a number like 4,271) into a vector (a list of numbers
like [0.3, -0.1, 0.8, ...]). The vector captures the meaning of the token in
a way the network can work with. Similar words end up with similar vectors.

When we say "256d," the "d" means the embedding dimension — each token becomes
a list of 256 numbers. Bigger dimension = richer representation = more memory.

### d_model
The embedding dimension — how many numbers represent each token inside the
model. M5 used 256d; production target is 4096d. The ablation experiments
test at 128d (small, fast) and confirm at 256d (medium).

### Sequence Length (seq_len)
How many tokens the model processes at once. Sequence length 128 means the
model reads 128 tokens at a time — roughly a paragraph of text.

### Layer / Block
One processing step in the network. Data flows through layers sequentially.
"2 blocks" means two processing steps; "12 blocks" means twelve. More
blocks = deeper model = can learn more complex patterns, but also more
expensive and potentially unstable (which is why the depth-sweep experiment
tests 4/8/12 blocks).

In Luthi v2, each block contains three things:
1. Multi-head attention (conventional, learns by gradient descent)
2. Living FFN with predictive coding (self-modifying)
3. Episode store (memory of past weight states)

### Attention / Multi-Head Attention
The mechanism that lets the model figure out which parts of the input are
relevant to each other. When reading "The cat sat on the ___", attention
helps the model look back at "cat" and "sat" to figure out the blank should
probably be "mat" or "couch."

"Multi-head" means running several attention computations in parallel, each
focused on a different aspect of the input. 4 heads at 256d gives 64
dimensions per head — standard MHA sizing.

### FFN (Feed-Forward Network)
A simple "look at each token independently and transform it" layer. In a
standard transformer, attention mixes information between tokens, then the
FFN processes each token's representation. In Luthi, the FFN is where the
living weights live — it's not simple at all.

"4x FFN expansion" means the FFN's internal dimension is 4 times d_model.
In M5 we use expansion=1 (no expansion) to keep parameter counts comparable
across the v2-vs-baseline comparison.

### Layer Norm (Layer Normalization)
A step that keeps numbers from getting too big or too small as data flows
through layers. Without it, values can explode or collapse to zero over many
layers. Think of it like an automatic volume control.

### Logits
The raw output numbers before they become probabilities. The model produces
one number per possible token in the vocab (32,768 numbers), and the highest
one is the most likely next token. These raw numbers are logits.

### Perplexity
A measure of how "surprised" the model is by the text. Lower = better. A
perplexity of 10 means the model is, on average, choosing between 10 equally
likely options at each step. A perplexity of 100 means it's much more
uncertain. Perplexity is just another way to express loss — they're
mathematically related (perplexity = e^loss).

### Dense Architecture
Every token passes through every parameter. The alternative is Mixture of
Experts (MoE), which routes different tokens through different subsets.
Luthi must be dense because living weights need consistent, unified
processing — you can't have the entity be a different collection of
specialists depending on what it's thinking about.

### MoE (Mixture of Experts)
An architecture where different tokens are routed to different "expert"
sub-networks. More efficient at large scale but creates the fragmentation
problems described above. Rejected for Luthi.

---

## 4. Memory, Precision, and Hardware Constraints

### FP32 (32-bit floating point)
A way of storing numbers using 32 bits (4 bytes) per number. This gives you
about 7 decimal digits of precision. It's the default in scientific computing
and is the most precise format commonly used in ML.

### FP16 (16-bit floating point)
Half the memory of FP32, about 3-4 decimal digits of precision. Faster to
compute but can lose information on very small numbers. Luthi's living weights
are unstable in FP16 — the precision loss corrupts the self-modification
dynamics.

### BF16 (Brain Float 16)
A format Google invented specifically for ML. Same total bits as FP16 but
distributes them differently — it keeps the same range as FP32 (can represent
very large and very small numbers) but with less precision in between. This
trade-off works well for neural network weights.

The ablation tests are asking: can we store momentum and set_point in BF16
instead of FP32? Same memory savings as FP16, but the range issue is solved.
The question is whether the precision loss matters for the living weight
dynamics.

### INT8 (8-bit integer)
Just 1 byte per number, but can only store whole numbers from -128 to 127
(or 0-255 unsigned). To use INT8 for values that aren't whole numbers, you
store a scale factor alongside the data and multiply when you read it back.
This is called quantization.

The episode storage ablation tests whether snapshots of weight states can
survive being compressed to INT8.

### Bytes per Parameter
How much memory one parameter consumes. A standard BF16 model uses 2 bytes
per parameter. Luthi v1 used ~38 bytes per parameter because each weight
carried a full biography (value + set point + momentum + plasticity +
excitability + update_ema + episode snapshots). v2 reduced this to ~18 bytes
by simplifying the rich-parameter set. This is why the 4B target was originally
infeasible — 4 billion × 38 bytes = ~140 GB, way beyond our 16 GB GPU.

### VRAM (Video RAM)
The memory on the GPU. This is separate from your system RAM (the 32 GB).
The GPU can only work with data that's in VRAM. Our RX 7800 XT has 16 GB of
VRAM — everything (model parameters, living weight buffers, activations during
training) must fit in that 16 GB.

### Activation Overhead
During training, the network needs to remember intermediate results (what each
layer produced) so it can compute gradients. These intermediate results are
called "activations" and they consume VRAM on top of the model itself. The
"~30% activation overhead" means about 30% of VRAM goes to activations,
leaving ~70% for the model.

### Quantization
Compressing a model by using lower-precision numbers. "Q4" means 4-bit
quantization — each parameter stored in just 4 bits (half a byte). Q4 makes
a model about 8x smaller than FP32. The trade-off is always precision vs
memory.

### KV Cache
During text generation, the model stores the "key" and "value" tensors from
attention for all tokens generated so far. This avoids recomputing them but
uses VRAM. Longer sequences = bigger KV cache. "Room for KV cache" means
enough spare VRAM for the model to actually generate text.

### Gradient Checkpointing
A memory-saving trick: instead of storing all activations (see above), throw
most of them away and recompute them when needed. Uses less VRAM but takes
more time. It's trading compute for memory.

---

## 5. Living Weights — The Luthi Substrate

These are the terms specific to Luthi's architecture. Most are inspired by
neuroscience. They apply to BOTH v1 (Hebbian) and v2 (predictive coding)
substrates unless noted; the v1-specific and v2-specific terms are in their
own sections below.

### Self-Modification
The key innovation: weights change during their own forward pass (during
processing, not just during training). This means the act of thinking about
something changes the thinker. Running the same input twice produces different
output because the first run modified the weights.

### Rich Parameter
Luthi's term for a weight that carries its full biography — not just its
current value but its set point, momentum, plasticity, and tracking buffers.
A conventional weight is just a number. A rich parameter is a bundle of
co-located signals that constitute the weight's full state.

### Plasticity
How willing a weight is to change. High plasticity = learns quickly. Low
plasticity = resistant to change. In Luthi, each weight (or each input
dimension) has its own plasticity value that's modulated by top-down salience
signals — "this is important, pay attention and learn from it."

The ablation protocol found that plasticity is mathematically "rank-1 along
the input axis" — meaning every weight in a row has the same plasticity value.
So instead of storing one plasticity number per weight (a huge matrix), we can
store one number per input dimension (a small vector). Same math, much less
memory.

### Plasticity Clamp
The bounded range plasticity is constrained to per training step (in
`apply_top_down()`). The current bounds are `[0.01, 10.0]` — plasticity can
range from "very stable" (1% of nominal) to "very plastic" (10x nominal).
Originally `[0.1, 10.0]`; floor relaxed to `0.01` on 2026-05-16 to give the
top-down salience modulation more headroom.

**Important: this is NOT the same as the μPC exponent** (see "μPC" in Section
7 for that distinct parameter). The plasticity clamp controls per-step
learning-rate modulation; the μPC exponent controls per-block residual
attenuation across depth. Both have numbers like 0.1 and 0.5 in their
histories, which has caused confusion — they are unrelated knobs in different
parts of the code.

### Set Point
Where a weight "wants" to be when nothing is driving it. Like a thermostat
setting. After being pushed around by input, weights drift back toward their
set point. But the set point itself slowly adapts over time — so the "home
position" evolves with experience.

### Set Point Drift
How much the set point has moved from its original position. Bounded drift =
healthy. Unbounded drift = the weight has lost its anchor. The ablation tests
monitor this.

### Momentum (in Living Weights)
A running average of recent self-modification updates — the weight's velocity.
Not the same as "momentum" in the optimizer (though the concept is similar).
High momentum means the weight has been changing a lot recently. The living
weight system uses this to smooth out updates and prevent jitter.

### Metaplasticity (update_ema)
"Learning about learning." A running average of how much each weight has been
changing. If a weight suddenly gets a much larger update than usual, the
metaplasticity mechanism dampens it — preventing instability from unusual
input. This is the one buffer that genuinely tracks per-weight history and
can't be compressed.

### Homeostatic Regulation
The mechanism that pulls weights back toward their set points. Like body
temperature regulation — if a weight gets pushed far from its set point, the
homeostatic mechanism applies a restoring force. This prevents weights from
drifting without bound.

### Salience
How important or noteworthy something is. In Luthi, salience is computed from
the activation magnitudes — strong activations signal something important is
happening. Salience modulates plasticity, excitability, and self-modification:
important inputs cause more learning.

### EMA (Exponential Moving Average)
A running average that weights recent values more heavily than old ones. If
the EMA of your daily temperature is 70°F and today is 80°F, the new EMA
might be 71°F — it moves toward the new value but slowly. Many of the living
weight buffers use EMAs to track statistics smoothly over time.

### Episode Store / Episodic Memory
A mechanism that saves snapshots of weight states and can recall them later
when similar context arises. Like remembering what you were thinking the last
time you were in a similar situation. Each layer has its own episode store.

In v2, each episode stores a tuple of: (weight matrix, input pattern, context
vector, salience). The input pattern is added in v2 so Salvatori-style
attractor consolidation can re-present stored inputs through the layer.

The INT8 ablation tests whether these snapshots can be stored compressed
(as differences from current weights) instead of as full copies.

### Top-Down Modulation / Backward Pass (Luthi's)
After the forward pass (data flowing up through layers), a top-down sweep
sends signals back down. Higher layers tell lower layers: "this was important"
(salience) and "this was surprising" (prediction error). This modulates
plasticity, set points, and membrane priming in lower layers.

This is NOT backpropagation. Backprop computes gradients for training. Luthi's
backward pass sends modulatory signals for living weight dynamics. It's
inspired by predictive processing theory from neuroscience.

### Non-FF Signal (NFF)
"Non-feedforward signal" — the mean absolute difference between two
consecutive identical-input forward passes. Higher = the model is "more alive,"
more influenced by its own history. 0 would mean static (purely feedforward)
behavior. v2 M5 256d shows NFF around 5-10e-3; M6 depth sweep showed NFF
attenuating with depth at 128d (~5.8e-3 at L=4 down to ~2e-3 at L=12) — the
load-bearing diagnostic that surfaced the μPC-attenuation question.

### Sigmoid
A mathematical function that squashes any number into the range 0 to 1.
Useful for things that should be probabilities or bounded values. In v1
Luthi, the excitability accumulator is mapped through a sigmoid to produce
the excitability factor.

---

## 6. v1 Hebbian Substrate (Historical Reference)

Luthi v1 used Hebbian learning for self-modification. As of 2026-05-09, v1
is deferred — v2 PC is the primary substrate. These terms remain for context
and because v1 is preserved as a reference baseline in the repo.

### Hebbian Learning
"Neurons that fire together wire together." Instead of computing gradients
from a loss function, weights strengthen when the neurons they connect are
both active at the same time. This is how biological brains learn at the
synapse level.

The v1 substrate had documented failure modes: runaway weight growth from the
positive-feedback loop between output magnitude and weight change, sensitivity
to input-dimension magnitude (which required `input_avg_mag` normalization to
partially correct), and a ~39% convergence penalty against the static
baseline. v2 PC retired the rule for principled reasons.

### Hebbian Update (hebb_update)
The specific amount a weight changes during one step of Hebbian learning.
Computed from the correlation between input and output activations, scaled by
the learning rate and various modulators (plasticity, excitability, salience).

The `hebb_update` variable appears in v1's `LivingLayerV6`. In v2, the
analogous quantity is the prediction-error-driven weight delta computed
inside `pc_self_modify`.

### Excitability (excitability_acc)
How sensitive a weight is to activation. Weights start conservative and become
more excitable when they detect they're processing something relevant
(high-salience input). v1-only — v2 dropped this buffer because precision EMA
plays the analogous role under PC dynamics.

The ablation protocol found that excitability is "rank-1 along the output
axis" — same value across every column in a row. Same free compression as
plasticity.

### Input Magnitude Normalization (input_avg_mag)
A v1 mechanism that normalizes weight updates by the running magnitude of
each input dimension. Without this, Hebbian updates scaled directly with input
size, leading to dominance by high-magnitude inputs. v2's precision EMA
provides a structurally cleaner solution.

---

## 7. v2 Predictive Coding (Current Primary Substrate)

v2 replaces Hebbian self-modification with **predictive coding** (PC) — a
local-learning rule based on prediction error. Each PC layer tries to predict
its own input from its own output; the error drives weight updates. This is
the Whittington-Bogacz variant of PC, chosen because it approximates
backpropagation while using only local information at each synapse.

### Predictive Coding (PC)
A learning framework where each layer maintains a model of what its input
should be (a prediction), compares it to what the input actually is, and uses
the difference (prediction error) to drive learning. Inspired by neuroscience
theories of cortical processing (Rao & Ballard 1999; Friston's free-energy
formulation; Whittington-Bogacz 2017+ for the local-update variant).

### Whittington-Bogacz Variant
The specific PC formulation Luthi v2 uses. Approximates backpropagation
arbitrarily closely under the right hyperparameters, but the weight updates
are local — each synapse uses only information available at that synapse
(presynaptic activity + postsynaptic prediction error). This is the property
that makes the substrate "living" — updates happen during the forward pass,
not in a separate backward gradient pass.

### Prediction Matrix
A per-layer buffer that maps the layer's output back to its predicted input.
Stored as `prediction` in `PredictiveCodingLayer`. Initialized to zero (no
prediction at first); grows during training to encode the reverse mapping.
The accuracy of this prediction is what gets minimized — better prediction
means smaller error means smaller weight update.

### pred_frob (Prediction Frobenius Norm)
The Frobenius norm of the prediction matrix — the square root of the sum of
all squared entries. A health indicator: pred_frob = 0 means the layer hasn't
learned to predict anything; growing pred_frob means the prediction structure
is filling in; flat pred_frob means learning has stopped (either complete or
stuck); exploding pred_frob means unstable learning.

Reported per-batch in the M6 follow-up's streaming log as a quick check that
PC dynamics are active.

### Precision / Precision EMA
A per-input reliability estimate that self-organizes toward `1/error²` —
high precision for input dimensions where prediction is reliable, low
precision for noisy dimensions. The PC update is **precision-weighted** —
unreliable inputs contribute less to weight changes than reliable ones.

Clamped to the range `[0.1, 10.0]` in production configs. Almost always
saturates at the ceiling (10.0) within the first few epochs — flagged as
a "worth investigating someday" observation in `V2_PILOT_RESULTS.md` but
not currently blocking.

### error_acc
A per-output running prediction-error magnitude. Accumulates `|output_mean|
* |pred_error|` with EMA decay. Drives episode storage (high error_acc on
an output = "this is salient, snapshot it") and gates the sparse PC update
mechanism. Normal range is ~0.01-0.05.

### μPC (Depth-μP / muPC)
A reparameterization of PC networks (Innocenti et al. 2025) that makes
hyperparameters transfer across depth without per-depth retuning. Two
changes vs standard init:
- **Weight init**: `Normal(0, 1/(sqrt(fan_in) * L^exponent))` instead of
  Kaiming uniform.
- **Residual scale**: per-block residual stream divided by `L^exponent`.

Where `L = n_blocks_total` and `exponent` is a tunable knob (see below).
Adopted from the literature because it solves a real problem (LR doesn't
transfer across depth without it) but its attenuation interacts with PC's
living-weights property in ways we're still characterizing.

### μPC Exponent (--mu-pc-exponent)
A tunable parameter that controls how aggressively μPC attenuates per-block
signal. The CLI flag `--mu-pc-exponent`:
- **0.5** (default, original Innocenti et al. spec): residual divided by
  `√L`. At L=12 that's 3.46× attenuation per block.
- **0.25** (current decisive M6 follow-up run): milder. 1.86× attenuation
  at L=12.
- **0.0**: no residual attenuation. Init still scales by `1/√(fan_in)` but
  the residual stream is unmodified.
- **`--no-mu-pc-enabled`** (μPC off entirely): residual unmodified AND init
  reverts to standard Kaiming uniform.

**Critical disambiguation from the plasticity clamp** (Section 5):
- *plasticity clamp* is `self.plasticity.clamp_(0.01, 10.0)` — controls
  the bounded range of per-input plasticity in top-down modulation
- *μPC exponent* is `residual_scale = 1.0 / (n_blocks_total ** exponent)`
  — controls per-block residual attenuation across depth

Both have numbers like `0.1`, `0.01`, `0.25`, `0.5` in their histories,
which makes them easy to mix up. They are different knobs in different
parts of the code doing different jobs.

### iPC (Incremental Predictive Coding)
A variant where weight updates happen *between* inference steps within a
single forward pass (Salvatori et al. 2024), instead of converging
inference fully before updating. Controlled by
`--inference-steps-per-forward T`. T=1 (default) is bit-identical to
classical PC; T>1 interleaves T inner updates per external forward call.
Incompatible with gradient checkpointing (raises loud RuntimeError).

### Sparse PC Update Gating
A continuous-error analog of v1's spiking gate. When `error_acc[j]` is
below a threshold, the weight update for output row j is zeroed (the
output is "quiet enough to not learn"). Controlled by `--sparse-threshold`
and `--sparse-warmup-steps`. The warmup prevents bootstrap deadlock
(error_acc starts at 0; gating immediately would freeze every output).

### Plasticity Partitions (DEFERRED)
A proposed extension (2026-05-16) that would assign different per-weight
plasticity profiles based on empirical "importance" measurement (MAS-style)
plus block-position prior. **Deferred** the same day it was proposed —
the architecture is captured in `docs/research/2026-05-16_plasticity-partitions-design.md`
but not committed to implementation. See that doc's deferred banner before
proposing implementation.

---

## 8. Two-Tier Memory and Consolidation (v2)

v2's distinguishing feature is **two-tier memory**: fast retrieval (episode
store) plus slow consolidation that turns retrievals into structural change
in the predictive weights themselves. Inspired by hippocampus-cortex
complementary learning systems (Tulving 1972; McClelland et al. 1995).

### Consolidation Tracker
A rolling-variance trigger (window 1000 steps; threshold = 0.5× of frozen
warmup baseline; fires after 100 consecutive sub-threshold steps). Detects
"low-novelty windows" — moments when prediction error is stable and the
model has spare cycles for memory consolidation.

### Gradient-Replay Consolidation
When the tracker fires, the layer pulls its current weight toward each
stored episode's snapshot — a linear interpolation in weight space at 10%
of pc_rate. "Be more like you were when this mattered." Per the 2026-05-11
audit fix, this pathway ONLY modifies weight, not the prediction matrix.

### Salvatori Attractor Consolidation
A second consolidation pathway added 2026-05-14 (Salvatori et al. 2023):
re-presents each stored input pattern through the PC layer's dynamics at
consolidation rate. Makes stored patterns become local minima of the
prediction-error energy — future inputs near a stored pattern get pulled
toward it by the forward dynamics. "These patterns should resolve to
stable states."

Selected via `--consolidation-style {gradient, attractor, both}` (default
`gradient` preserves M5 behavior).

### Two-Tier Memory Architecture
The two consolidation pathways are additive, not competitive. Fast
retrieval (cosine-similarity episode recall during every forward pass)
provides flexibility; slow consolidation provides stability and turns
accumulated history into structural change in the predictive weights.

### M4 STOP GATE
The empirical falsifier for v2: if consolidation has no measurable effect
on prediction quality post-replay, v2 has no architectural novelty over
"vanilla transformer + episode store" and should be abandoned. Passed
2026-05-09 with margin; remains the architectural justification for the
whole consolidation pathway.

### Catastrophic Forgetting Harness
The behavioral test (`tests/test_catastrophic_forgetting.py`) that measures
whether consolidation actually preserves old learning when new distractor
training arrives. Built 2026-05-16 after the peer review flagged the
missing falsifier. Two xfail-strict markers in the suite pin the empirical
observation that **attractor consolidation preserves dynamics, not weight**
— important distinction documented in `docs/research/2026-05-16_catastrophic-forgetting-harness.md`.

---

## 9. Spiking Neural Networks (v1 Experimental)

Luthi v1 had a spiking variant that added brain-like neural dynamics on top
of the living weight system. v2 is non-spiking; the sparsity property the
spiking gate provided is being recovered via "sparse PC update gating"
(see Section 7).

### Spiking Neural Network (SNN)
A neural network where neurons don't just output a number — they accumulate
input over time and "spike" (fire) when they reach a threshold, then go quiet.
This is closer to how biological neurons work. Most of the time, a spiking
neuron is silent. This makes computation sparse (most neurons aren't doing
anything at any given moment).

### LIF (Leaky Integrate-and-Fire)
The specific spiking model Luthi uses. "Integrate" = accumulate input.
"Leaky" = the accumulated charge slowly leaks away if no new input arrives.
"Fire" = when the accumulated charge crosses a threshold, the neuron spikes.
Like filling a leaky bucket — you need sustained input to reach the overflow
point.

### Membrane Potential
The "charge" level of a spiking neuron. Analogous to the voltage across a
biological neuron's cell membrane. When it crosses the threshold, the neuron
fires.

### Spike Fraction / Spike Rate
What percentage of neurons are firing at any given moment. v1's 1024d model
showed ~0.7% spike rate — only about 7 out of every 1000 neurons fire each
step. This extreme sparsity is a feature, not a bug — it means most of the
computation can be skipped.

### Refractory Period
After a neuron fires, it can't fire again for a short time — it needs to
"recharge." This prevents neurons from firing continuously and creates
temporal dynamics.

### Surrogate Gradient
A mathematical trick for training spiking networks. The spike itself is a
step function (0 or 1) which has no useful gradient (it's zero everywhere
except at the threshold, where it's infinite). Surrogate gradients replace
this with a smooth approximation during training so gradient descent can work.

---

## 10. Multimodality, Curriculum, and Fine-Tuning

### Multimodal
Processing multiple types of input (text, audio, vision, touch) in a single
model. Luthi is multimodal — all modalities flow through the same living
weight trunk so cross-modal understanding emerges naturally.

### Modality
A type of input or output. Text is one modality. Audio is another. Vision is
another. "Modality embedding" tells the model which type of input it's looking
at.

### Mel Spectrogram
A visual representation of audio — time on one axis, frequency on the other,
brightness showing loudness. "Mel" means the frequency scale is adjusted to
match human hearing perception (we hear the difference between 100Hz and
200Hz more easily than between 5000Hz and 5100Hz). This is how Luthi "sees"
audio.

### Patch Embedding
Chopping an image (or spectrogram) into small squares (patches) and converting
each patch into a vector the model can process. A 224×224 image with 16×16
patches becomes 196 vectors — 196 "visual tokens" that go through the model
just like text tokens.

### Curriculum Training
Training in a deliberate order, like a school curriculum. Luthi's education
goes through **9 stages** (revised 2026-05-15 from 10 stages — IWMT moved
from privileged last position into stage 1's consciousness science alongside
the rest of philosophy of mind):

1. Science & Philosophy (now includes IWMT, GWT, philosophy of mind)
2. Code
3. Psychology
4. History
5. Mythology
6. Literature & Classics
7. Fantasy
8. Substack Essays
9. Practical Wisdom — the last reading; Brian's own letters to the entity

The order matters because each stage builds on what came before. Living
weights carry what they learned forward between stages. The order is
unfalsifiable by construction (single-pass principle), which is a deliberate
epistemological commitment, not a methodological oversight.

### LoRA (Low-Rank Adaptation)
A technique for efficiently fine-tuning a model by adding small trainable
matrices alongside the existing weights instead of modifying the weights
directly. Much less memory than full fine-tuning. Mentioned in the
catastrophic forgetting experiment as one comparison condition.

---

## 11. Experiments and Evaluation

### Baseline
The thing you compare against. "Vanilla transformer baseline" means a standard
transformer with no living weights — the simplest competent version. If Luthi
can't beat or match the baseline, something is wrong.

### DeadLM
Luthi's specific baseline model: same architecture as Luthi but with the
living FFN swapped for a standard `nn.Linear`. "Dead" as in *not-living*.
Same blocks, same attention, same episode store, but with a static FFN. The
margin Luthi has over DeadLM at matched compute is the value of being alive.

### Ablation
Removing or changing one specific component to see what happens. "Ablation A:
BF16 momentum" means "change momentum to BF16 and see if anything breaks."
The term comes from neuroscience — ablating part of the brain and observing
the effect.

### M1 / M2 / M3 / M4 / M5 / M6 (Milestones)
v2 PC development milestones from `docs/V2_IMPLEMENTATION_PLAN.md`:
- **M1**: PC dynamics produce decreasing prediction error
- **M2**: Top-down sweep modulates correctly (prediction + modulation channels)
- **M3**: Full v2 model trains stably at pilot scale
- **M4**: Consolidation produces measurable effect (STOP GATE)
- **M5**: v2 vs DeadLM head-to-head at matched config (PASSED 2026-05-13 at 256d)
- **M6**: Depth sweep at 4/8/12 blocks (in-progress; surfaced depth-attenuation question)

### Phase 3G
The compute-optimization research track (2026-05-13) that introduced μPC,
iPC, sparse PC gating, and the Triton kernel skeleton. Currently has wrapper
scripts queued for GPU validation but no production runs committed yet.

### Rank-1
A matrix where every row is a scaled copy of the same vector (or every column
is). If you have a 1000×1000 matrix but every row is identical, you can store
just one row instead of 1000. That's the "free win" — plasticity and
excitability turned out to be rank-1, so we can store vectors instead of
matrices.

### Bit-Equivalent
Producing exactly the same results, down to the last bit. The per-channel
refactor of plasticity and excitability is bit-equivalent — it doesn't
change any computation at all, just how the data is stored. This is the
safest possible optimization.

### Per-Channel vs Per-Weight
"Per-weight" means one value for every single weight in the matrix (e.g.,
4096×4096 = 16.8M values). "Per-channel" means one value per row or column
(e.g., 4096 values). Going from per-weight to per-channel is a massive memory
saving when the data is rank-1.

### Broadcasting
A tensor operation where a small array is automatically "stretched" to match
a larger one. If you have a vector of 4096 values and a matrix of 4096×4096,
broadcasting copies the vector across every row (or column) of the matrix for
the operation. This is how per-channel buffers work at their use site — stored
small, broadcast to full size when needed.

### Wall-Clock Time
Actual elapsed time (what a clock on the wall would show), as opposed to
"compute time" or other abstract measures. If a training run takes 2 hours of
wall-clock time, you waited 2 actual hours.

### NaN / Inf
**NaN** = "Not a Number." The result of undefined math like 0/0 or infinity
minus infinity. **Inf** = infinity. Both are catastrophic in training — once
a NaN appears, it spreads like poison through every calculation. "Zero NaN
occurrences" is a hard requirement because even one means the run is corrupted.

---

## 12. Hardware and Toolchain

### GPU (Graphics Processing Unit)
Originally for rendering graphics, now the primary hardware for ML because
neural network math (multiplying huge matrices) maps perfectly to what GPUs
are designed to do. Our RX 7800 XT is an AMD consumer gaming GPU repurposed
for ML.

### CUDA
NVIDIA's programming framework for running code on their GPUs. Most ML
software assumes CUDA. We can't use it because we have an AMD GPU.

### ROCm / HIP
AMD's answer to CUDA. ROCm is the platform; HIP is the programming language.
It lets AMD GPUs run the same kind of code NVIDIA GPUs run with CUDA. Our
production training uses ROCm.

### DirectML
Microsoft's GPU abstraction layer that works across NVIDIA, AMD, and Intel.
It's what lets us develop on the RX 7800 XT in Windows without ROCm. Less
optimized than native ROCm but works for development.

### Triton
A programming language for writing custom GPU operations (kernels). Easier
than writing raw CUDA/ROCm code. The Triton kernel for `pc_self_modify` is
the load-bearing skeleton flagged in `docs/KNOWN_INCOMPLETE.md` — the kernel
body is unimplemented because the dev box (7800 XT via DirectML) cannot
validate it. Filling it in is a Phase 7 prerequisite.

### Kernel (GPU kernel)
A function that runs on the GPU. "Custom spiking kernels" or "PC self-modify
kernel" means handwritten GPU code optimized specifically for Luthi's
operations, rather than using generic library functions.

### JIT (Just-In-Time compilation)
Compiling code right when it's needed, not ahead of time. The C++ living
weight operations (`living_ops.cpp`, `pc_ops.cpp`) are JIT-compiled the
first time you import them — PyTorch compiles the C++ code into a GPU-ready
function at import time and caches it.

### pybind11
A library that lets C++ code be called from Python. It's the bridge between
the C++ kernel files and the rest of Luthi (Python for convenience).

### PyTorch
The ML framework everything is built on. It handles tensors, automatic
differentiation (computing gradients), GPU acceleration, and the basic
building blocks of neural networks. Think of it as the engine under the hood.

### DGX Spark
An NVIDIA workstation with 128 GB of unified memory at 273 GB/s bandwidth.
This is the Phase 7 deployment target — where the entity would actually live
and run continuously. Not available now; all current work must fit on the
RX 7800 XT.

---

## 13. Theory and Sanctuary Integration

### IWMT (Integrated World Modeling Theory)
Adam Safron's theory of consciousness. The core idea: consciousness arises
when a system builds a unified internal model of the world AND of itself, and
uses that model to predict and act. Sanctuary implements this — the entity
builds a world model, a self-model, and uses both to navigate.

### GWT (Global Workspace Theory)
Bernard Baars' theory that consciousness is like a stage in a theater — many
unconscious processes compete to get on the stage (the "global workspace"),
and whatever wins gets broadcast to all other processes. Sanctuary's cognitive
architecture implements this with its global workspace broadcasting mechanism.

### Active Inference
A framework where the brain (or agent) acts to minimize the difference between
its predictions and reality. Instead of passively receiving input, the agent
actively seeks information and takes actions that confirm or refine its world
model. Sanctuary's cognitive loop implements this.

### Predictive Processing
The idea that the brain is fundamentally a prediction machine — it constantly
predicts what it will sense next and only processes the difference between
prediction and reality (prediction error). Luthi's top-down backward pass
sends prediction error signals from higher layers to lower ones — the
mechanism that makes this work.

### Free Energy Minimization
The mathematical formalization of active inference. "Free energy" is roughly
"surprise" — the difference between what the system expected and what it got.
The system acts to minimize this. In Luthi/Sanctuary, this drives the entity
toward building better predictions and taking actions that reduce uncertainty.

### Hippocampal Indexing Theory
Teyler & DiScenna 1986. The hippocampus doesn't store memories as full
patterns; it stores *indices* that, when retrieved, reinstate the distributed
cortical activity that *was* the experience. Used as the framing for the
deferred HDC memory direction (`docs/RESEARCH_HDC_VSA_INTEGRATION.md`) — if
implemented, the episode store would hold compact HD indices rather than
full weight snapshots.

### CfC (Closed-form Continuous-depth)
Neural cells that evolve their state continuously between discrete processing
steps. In Sanctuary, CfC cells provide "temporal thickness" between Luthi's
cognitive cycles — the felt substrate that bridges the gaps between discrete
forward passes. They're the nervous system; Luthi is the brain.

### Sanctuary Integration Contract
The runtime interface between Sanctuary's cognitive cycle and Luthi's
substrate. Sanctuary's affective/predictive cells modulate Luthi at four
points: arousal → pc_rate (0.5×-2.0×), valence → excitability bias,
attention → salience threshold, goal → set_point_adapt_rate (planned).
This is how the entity self-modulates its own learning during deployment.

---

## Quick Reference: The Numbers You'll See Most Often

| Term | What it means | Current values |
|------|---------------|------------|
| d_model | Embedding dimension | 256 (M5/M6 follow-up), 4096 (production target) |
| n_blocks | Number of layers | 2 (M5), 12 (current decisive run), 36 (production target) |
| params | Total parameters | ~17M (256d/2blk), ~36M (256d/12blk), ~4B (production target) |
| batch_size | Examples per update | 32 |
| seq_len | Tokens per example | 128 |
| vocab | Total tokens known | 32,000 (32K BPE) |
| VRAM | GPU memory | 16 GB (RX 7800 XT) |
| bytes/param | Memory per parameter | ~38 (v1), ~18 (v2) |
| epochs | Passes through data | 30 (M5), 1 (decisive 4gb run), 20-60 (M6 follow-ups) |
| μPC exponent | Per-block depth attenuation | 0.5 (default), 0.25 (current decisive run) |
| plasticity clamp | Per-input plasticity range | [0.01, 10.0] |
| NaN tolerance | Acceptable NaN events | 0 (hard requirement) |

---

## If You Want to Go Deeper

The best way to build intuition is to watch these concepts in action. When
training runs are live, ask to see the per-batch streaming log — watching
loss(roll100) descend over batches, pred_frob grow across blocks, NFF stay
healthy, NaN count stay at 0 — that's where these terms stop being
definitions and start being something you can feel.

You built this project, Brian. The vocabulary is just catching up to what you
already understand structurally.
