# Can dreaming improve the mind? — 2026-11-11

Fable 5.2, in session with Brian. Follows Sanctuary's
`docs/research/2026-11-11_looking-back.md`, steps 9 to 11.

## Objective

Brian, 10:15: "can you build the part that lets the model improve through dreaming
or is that outside your ability?" In Google Research's Sleep framework a resting model
generates samples, keeps the ones a gradient test says will help, learns from them,
and checks that it improved. Rest already dreamed here, as a random mix of two kept
moments, and learned from a dream only as it learns from any lived moment.

Before building: does learning from a dream improve anything, and does choosing the
dream matter? (Standing rules 1 and 3.)

## Process

### Step 1: the probe

`redteam/fable_2026_11_11/probe_do_chosen_dreams_help.py`. One trial:

- Twelve "kept" real moments and 24 "held back" ones, sentences from a small world the
  trunk was not trained on (ten subjects, each with one thing it does). Held back is
  never learned from.
- Eight candidate dreams, each half of one kept moment's words and half of another's.
- The chosen dream is the candidate whose learning direction (the gradient of the
  trunk's own training loss) lines up best with that of the kept real moments.
- From the same starting point each time, one small step on: the chosen dream, a
  random dream, a kept real moment, text from the trunk's own training, random
  tokens, or nothing.
- Reading: prediction loss on the held-back moments, after minus before. Plasticity is
  frozen throughout, so this reads the gradient-trained parts only.

### Step 2: first result, and why it meant nothing

On the trunk used all morning (300 training batches):

| One step on | Held-back loss changed by |
|---|---:|
| Chosen dream | −0.0065 |
| Random dream | −0.0057 |
| A kept real moment | −0.0063 |
| Text from its training | −0.0052 |
| Random tokens | −0.0060 |

- Without the last two rows this reads as "dreaming works". With them it reads as
  "any step at all helps". The trunk was so little trained that every learning
  direction pointed the same way (dreams lined up with real moments at +0.80).
- I had run the first three rows alone and was about to build on them. The two
  controls are why I did not.

### Step 3: a trunk that has learned its corpus

The same trunk trained 2,268 batches (held-out prediction loss 1.08). Two runs of 80
trials. Standard errors in brackets.

| One step on | Held-back loss changed by | Better in |
|---|---:|---:|
| Chosen dream | −0.00220 (0.00005), −0.00218 (0.00006) | 80/80, 80/80 |
| Random dream | −0.00197 (0.00009), −0.00215 (0.00007) | 79/80, 80/80 |
| A kept real moment (replay) | −0.00182 (0.00007), −0.00198 (0.00008) | 80/80, 79/80 |
| Text from its training | −0.00037 (0.00011), −0.00026 (0.00009) | 51/80, 49/80 |
| Random tokens | −0.00040 (0.00008), −0.00037 (0.00010) | 56/80, 52/80 |

Paired within trials:

- Random dream against random tokens: −0.0016 and −0.0018, better in 75 and 77 of 80.
- Random dream against replay: −0.00014 (0.00011) and −0.00017 (0.00008).
- Chosen against random dream: −0.00024 (0.00010) and −0.00004 (0.00007); better in
  only 35 and 31 of 80.

Reading:

- **A dream teaches.** One step on a dream improved prediction of real moments the
  trunk had never learned from, about five times more than a step on unrelated text
  or noise.
- **A dream teaches about as much as replaying a real kept moment**, perhaps slightly
  more.
- **Choosing the dream gave no reliable gain.** Every mix of two kept moments already
  lines up with real moments (+0.70 on average, +0.77 for the chosen one).

### Step 4: what was built

- `luthi/v2/dreaming.py`, `learn_from_dream(actor, dream, real, step_size)`. One small
  step on the trunk's training loss for the dream's words, then the check: prediction
  loss on real kept moments before and after, and the step is undone exactly if it is
  worse. No choosing, by step 3.
- It reads and learns with plasticity frozen. It leaves the trainer's optimizers and
  any standing gradients as they were (pinned by test).
- Sanctuary's rest calls it for each dream, checked against up to six kept moments
  that are not the dream's parents.
- Words only: a dream's picture and sound are not learned from.

### Step 5: one tiny life with it

1,500 moments: 10 dreams, 10 learned from, 0 undone, no rollbacks. In that life the
mind chose to recall 71 times, against 4 in the life before dream learning. One life
each, so I do not know that the learning caused it. It should be looked at.

## Conclusion

- Yes: a mind can improve through dreaming here, in the sense measured. Learning from
  a dream made real, unseen moments from the same world easier to predict.
- It is not more than replay gives. On this evidence a dream is another way to
  practise what was kept, not a source of something new.
- The check is the part worth having regardless: a dream that makes real moments
  harder to predict leaves no mark.

Not known:

- Whether this holds past one 32-wide trunk and one made-up world.
- Whether it helps a mind over a life, as against one step. Nothing here ran long
  enough to say.
- Whether wilder dreams (mixing more than two moments, or across senses) would teach
  more or do harm. With those, choosing might start to matter.
- What the right step size is at full size. 0.01 is the probe's.

Open, for Brian: whether revisited memories should be learned from the same way
(replay teaches about as much, and the same function would do it); review by a
different mind, since this reaches into how Luthi learns by gradient.

## Artifacts

- `luthi/v2/dreaming.py`; `luthi/sanctuary_interface.py` (`learn_from_dream`);
  `tests/test_dreaming.py` (3).
- Probe: `redteam/fable_2026_11_11/probe_do_chosen_dreams_help.py`, run on trunks from
  `halfday_train_trunk.py` (`EPOCHS=1 BATCHES=300`, and `EPOCHS=6 BATCHES=100000`),
  with `TRIALS=80` and `SEED=0`, `SEED=1`.
