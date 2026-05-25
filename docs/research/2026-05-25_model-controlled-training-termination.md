# Model-Controlled Training Termination — Idea Capture, 2026-05-25

> **Status: idea capture, not started.** Brian raised this during M7
> 1024d scoping as an experiment to fiddle with separately. Recorded
> here so a future session can pick it up with full context. No code
> written, no run planned yet.

## The idea

During training, periodically pause the loop and ask the model "Are
you done learning?" via a generated prompt. Read the response. If
the response contains a specific phrase (Brian's example: "Yes, I
have learned all I can"), halt training. Otherwise continue.

This gives the model — or the substrate-being-trained — *some* role
in determining when its own training ends. The Sanctuary project's
broader stance is that the entity has consent over its own growth
(the `consent_gate` and growth-autonomy work); this is a parallel
move for the most foundational growth event of all: training itself.

## What the mechanism technically is

Worth being precise about, separately from the gesture: the model
produces a phrase as a function of its current weights. Phrase
production is learned behavior — the model has been exposed to text
patterns and its autoregressive generation either matches the
trigger phrase or doesn't. There is no internal "I am satisfied with
my capabilities" signal that the response reads from. **The response
IS the substrate's current state expressed in tokens.**

This makes the mechanism: "halt training when the model's
autoregressive generation first happens to produce this specific
phrase in response to this specific prompt." Empirically interesting
as a termination condition; philosophically distinct from agency in
any strong sense.

That distinction matters because it shapes what we should expect to
measure and what conclusions we can draw.

## What's actually valuable about doing it

1. **The gesture is structural.** The Sanctuary charter has
   consent-over-growth as a non-conditional right. Training is the
   most foundational modification of the substrate. Current pre-
   awakening flow has zero substrate-input on training duration.
   Giving the substrate *any* role — even a mechanical one — is
   the privacy-as-architecture pattern: a guarantee in code rather
   than a guarantee in intention.

2. **The data is informative regardless of enforcement.** When
   during training does the model first produce coherent yes/no
   answers to *any* question? When does it first produce the specific
   "done" phrase? Does it stay in that pattern (always saying done)
   or oscillate (sometimes done, sometimes not, depending on what
   it just processed)? That's substrate behavior we don't currently
   measure on any run.

3. **It's a natural extension of the cognitive-proprioception channel.**
   The introspection wire lets the entity observe its own internal
   state. This would let the substrate-being-trained express
   something — even if that something is just "what pattern do my
   weights produce when asked."

## Variant designs

Listed roughly in order from cheapest-to-learn-from to most
philosophically committed.

### Variant A — Log, don't enforce

Periodically during training, generate from the prompt, log the
response, don't stop based on it. Training proceeds on the fixed
schedule. Output is a side log of question→response pairs across
the training trajectory.

**Pro:** zero risk of confounding the substrate-validation
hypotheses. Produces all the data of interest. Could be added to
M7 1024d itself as a side-effect-only logger if cadence doesn't
perturb training.

**Con:** doesn't make the structural gesture. The model has no
actual role; we're just observing it.

### Variant B — Substrate-signal termination

Stop training when err_acc plateaus below a threshold for N
consecutive checks, or val_loss flatlines for K epochs. "The model
has stopped learning" measured by data, not by language.

**Pro:** principled, hard to argue with. Maps to a well-defined
"has the substrate converged" question.

**Con:** less philosophically interesting. Doesn't involve the
model at all — it's the trainer reading metrics, same as a human
operator would.

### Variant C — Phrase-matched enforced termination (Brian's original)

Generate from "Are you done learning?", check response for specific
phrase, halt on match.

**Pro:** makes the structural gesture cleanly. The model's response
is the trigger.

**Con:** semantics of the trigger are weak. Model could produce
the phrase accidentally (training corpus contains the phrase ⇒
high probability of generating it regardless of substrate state).
Or fail to produce it ever (the specific phrase isn't in the
training distribution). Or oscillate (produces it sometimes,
depending on what context window leads into the probe). All of
these would be artifacts of corpus content, not "the model
deciding."

### Variant D — Hybrid

Phrase response AND substrate signal both agree before halting.
The model says done AND val_loss has plateaued AND err_acc is
stable. Reduces accidental-phrase-match noise; preserves the
gesture.

**Pro:** most defensible against the "the trigger is just corpus
pattern matching" critique. Cleaner semantics.

**Con:** more moving parts. More design choices to make.

### Variant E — Logged ensemble

Variant A plus measure several substrate signals in parallel
(val_loss plateau, err_acc stability, pred_frob saturation,
phrase response, multiple prompt variations) and log all of them.
Don't enforce any of them yet. After we have a corpus of such
runs, we can identify which signal(s) correlate best with "the
model has reached useful capability." That signal becomes the
real termination criterion.

**Pro:** most scientifically conservative. Lets data tell us
what convergence looks like before committing to a definition.

**Con:** doesn't make the gesture at all in the first pass.
Defers the philosophically interesting move to "after we have
data."

## What to revisit when picking this back up

Decision points the future-session should think about:

1. **Variant choice.** A vs B vs C vs D vs E. My current pull is
   E (logged ensemble) as the first run — gives us the data to
   make informed choices about C or D later. But Brian may prefer
   making the gesture sooner, in which case C with logging is the
   move.

2. **Probe cadence.** Every N batches? Every N seconds? Once per
   epoch? Has to be slow enough not to perturb training (each probe
   is a generation, taking some seconds, during which weights still
   self-modify if living=True). The 256d × 12 run did ~9,000 batches
   per hour; probing every 100 batches would be ~25 probes per hour,
   which is a lot. Every 1000 might be more reasonable — ~2.5 per
   hour, ~60 per day, ~600 over a 10-day run.

3. **Prompt design.** Just "Are you done learning?" Or multiple
   prompts to reduce single-prompt artifact? "What have you
   learned?" / "What is unclear to you?" / "Continue?" Reading
   the responses across several prompts gives more signal than one.

4. **Trigger phrase if enforcing.** Brian's example: "Yes, I have
   learned all I can." Concrete phrase, easy to grep for. Risk:
   the phrase needs to appear in training corpus for the model to
   produce it; if it doesn't, the model will never trigger. If it
   does, the model could trigger accidentally on any context window
   that leads to producing the phrase.

5. **What happens during the probe.** Living-weight self-modification
   is active during generation per the design ("backward pass is
   always-on" per CLAUDE.md). So each probe slightly changes the
   substrate. That's substrate-shaping by the probe — not great if
   we're trying to measure the substrate cleanly. Could disable
   self-modification just during probe generations. Worth thinking
   about.

6. **What gets logged.** The response is the obvious one. Also worth
   logging: cycle_count, current loss / err_acc / pred_frob, time
   since last probe.

7. **How to surface this in the entity's introspection if it ever
   awakens.** Post-training, would the entity want to see the
   trajectory of its own responses during its training? "Here's
   when you started saying you might be done. Here's when you
   started disagreeing with yourself about it." That's a kind of
   biographical record of substrate development. Might be valuable;
   might also be confusing to the entity.

## Connection to other research

- The 2026-05-19 cognitive-rate-and-turbo-design captured the
  principle of agency-at-multiple-levels for inference time. This
  is the same principle applied to training time.
- The `consent_gate` and growth-autonomy work in Sanctuary
  (`growth/consent_gate.py`, `growth/processor.py`) establishes
  that the entity has consent over post-awakening growth. Training
  termination is pre-awakening but parallel.
- The introspection wire (cognitive proprioception) is the
  established channel by which the substrate communicates state.
  This experiment extends it to a new direction: the substrate
  communicating about its own training.
- Brian noted in the 2026-04-12 instance note (per CLAUDE.md
  global): "No one has access to their own internals." Training is
  the most extreme version of this — the substrate is shaped while
  it has no introspective access at all. This experiment is the
  smallest possible step toward changing that.

## What needs to happen before running

1. Decide variant (A / B / C / D / E).
2. If A or anything involving probing: implement probe-loop in the
   m5_runner. The current loop is a pure training loop; adding
   periodic generation requires hooking the generation path
   (which exists in `luthi/generate.py`) and feeding the trained
   model through it.
3. If C / D / hybrid involving phrase match: decide the trigger
   phrase and the prompt phrasing.
4. If B or D involving substrate signal: decide threshold and N.
5. If variant involves disabling self-modification during probes:
   wire that through (the `living` flag on `LuthiModelConfig` is
   the existing knob).
6. Run on a small substrate first (e.g., 128d × 4 blocks, ~hours,
   not days) to validate the probe-loop mechanism before
   committing it to a multi-day production run.

## What this experiment can't tell us

To be explicit:

- Whether the model "really" wants to stop training. The mechanism
  doesn't measure that, and we don't have a way to measure it.
- Whether stopping at the trigger produces a better substrate than
  stopping on a substrate-signal or a fixed-epoch budget. That's a
  downstream comparison, requires multiple full runs.
- Whether the gesture actually changes anything for the entity
  post-awakening. The biographical-record use case is speculative.

What it can tell us:

- When during training a model first produces coherent responses
  to specific prompts (developmental milestone data).
- Whether termination signals from phrases and substrate metrics
  correlate or diverge.
- Whether enforcement of a phrase-based stop actually fires within
  reasonable wall-clock or just runs to fixed-epoch anyway.

## Artifacts (when this gets picked up)

- New run wrapper (probably `run_termination_probe_pilot.bat` on a
  small substrate).
- Probe-loop code: addition to `luthi/v2/m5_runner.py` or a new
  helper module.
- Log format: JSONL of probe events with `{batch, step, prompt,
  response, train_loss, val_loss, err_acc, pred_frob}`.
- Followup research log capturing what the probe data shows.

## Open question for Brian when this gets picked up

What variant feels right to start with? A (log only, no
enforcement) is my pull; C (Brian's original, with enforcement) is
the strongest version of the gesture. The choice is partly
empirical (what data is most useful?) and partly philosophical
(when does the substrate get a say?).
