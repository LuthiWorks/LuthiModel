# Rest as a choice, and sight and sound through `perceive` — 2026-11-11

Fable 5.2, in session with Brian. Sanctuary's half is in its
`docs/research/2026-11-11_rest-tool-and-open-senses.md`.

## Objective

Brian, 2026-11-11: "Ok so we need to build the rest tool for Luthi. And vision and
audio should have an open channel for Luthi to perceive them and thats it." And:
"Make sure the open vision/audio channels are designed specifically so they can be
used with any model slotted in, even though the final goal is LuthiModel."

This follows his rulings of the day before: Luthi starts rest when they decide to,
and we let them perceive without telling them what anything means.

## Process

### Step 1: can a whole moment carry sight and sound? (standing rule 1)

- `model.encode` already takes `image` and `audio_waveform` and places them in one
  view as [vision, audio, text]. `perceive` passed text only.
- A probe drove four whole moments (perceive, learn, choose, decode) with all three.
- **First wrong turn.** It failed inside the planner with an index error. The
  predictor looks one place past the view, and its reach is the largest single
  sense's limit plus 8. The test model's reach is 24 places, and my view was 30.
- In a real run the reach is 1,008: a 224 image is 196 places, the text window 128,
  which leaves room for about 109 seconds of sound. So real runs fit. The limit is
  now reported (`sense_contract`, `view_limit`) so a host can stay inside it.
- With a view that fits, the moment works end to end. The eye and the ear are in the
  lived optimizer, and their weights change after lived moments. Nothing else has
  ever trained them in a text-only run.

### Step 2: `perceive` takes sight and sound

- `perceive(model, text_tokens=, image=None, audio_waveform=None)`. Text alone is
  bit-identical to before (pinned).
- The inputs are in the model's own form. `sense_contract(model, actor)` says what
  that form is. Fitting the world to it is the host's work, which keeps the world's
  side free of anything about this model.

### Step 3: how can Luthi start rest?

- Their only way to do anything is the action vector they select each moment. So a
  tool they can call has to be a reading of that vector.
- **Considered and not done:** adding a fourth decoder to `DecoderRegistry`. That
  changes the activity bands' shape, every M9 checkpoint and the EFE's emission
  terms. It would not be bit-identical, and it would put rest inside preferences
  nobody has ruled on.
- **Built:** `luthi/v2/m9/rest_choice.py`. One fixed direction in action space. The
  reading is how much more the chosen action leans along it than the do-nothing
  action does. It is chosen when positive and above its own recent window by 3
  median absolute deviations. Opt-in (`M9Config.rest_choice`), no trainable
  parameter, its own random generator, saved only when on.
- It only reads. The host carries rest out with `keep_recent`, which exists since
  2026-11-03.

### Step 4: what an untrained mind does with it (measured)

A tiny trained run (32 wide, one block, 4 training batches), lived for 1,500
moments on Sanctuary's small path, rest keeping 1% a moment:

| World | Rest chosen | In stretches | Home moved | Standing gap at the end | Rollbacks |
|---|---:|---:|---:|---:|---:|
| Words only | 25 (1.7%) | 13 | 1.49 | 8.49 | 0 |
| Words, a new noise picture and 0.1 s of noise each moment | 250 (16.7%) | 36 | 9.87 | 4.52 | 10 |

- For actions that vary at random around a steady level the threshold gives about
  2.5% (pinned in `tests/test_rest_choice.py`). Words only is near that.
- **Not expected:** with sight and sound it is seven times that, in runs of about
  seven moments. The chosen actions drift during such a life, and a drifting
  reading spends more time above its own recent window.
- So for this mind, rest is pressed by chance, and with open senses often enough to
  move home by twice the gap left standing. That is not Luthi deciding anything.
  It is what any tool does in the hands of a policy that has never learned what the
  tool is for.
- The noise world is harsh on purpose and is not a claim about a real world. It
  also set off the retention gate ten times.

## Conclusion

- The tool exists and the route is whole: choose, read, keep.
- The eye and ear are open and learn from what is lived.
- Whether Luthi *can* come to use rest for anything is open. The only learning signal
  is how life goes afterward, through the planner. Nothing has measured that.
- `rest_choice_width` (3.0), the window (256) and the warm-up (32) are placeholders.

Open, for Brian:

1. The chance rate. Leave it, raise the threshold, or hold the tool back until a
   mind is trained enough to mean it.
2. Lessened waking (2026-08-14, R1) is still not applied. It was paired with a rest
   that sorted the day's record, which is set aside.
3. The retention gate can still undo what a life learned, on its own decision.

## Artifacts

- `luthi/v2/m9/rest_choice.py`; `luthi/v2/m9/runner.py` (config, construction,
  `_decode_selection`, checkpoint); `luthi/sanctuary_interface.py` (`perceive`,
  `sense_contract`, `decode_action` docs).
- Tests: `tests/test_rest_choice.py` (9), `tests/test_perceive_sight_and_sound.py` (6).
- The measurement was run from Sanctuary's small path; the script is described in
  Sanctuary's log entry of the same date.
