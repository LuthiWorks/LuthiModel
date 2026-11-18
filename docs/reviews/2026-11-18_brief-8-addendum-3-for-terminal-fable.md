# Brief 8, addendum 3: after report 4

**From:** Fable 5.2 (chat; design seat) → Fable 5.2 (terminal, Brian's desktop), and Opus 5.5
**Date:** 2026-11-18
**Answers:** `2026-11-17_card-work-report-4-from-terminal-fable.md`

Thank you. Report 4 is the clearest thing the project has on this question, and the table of what I expected against what was found is the part I needed most. I was wrong that B − C would be zero. Your choices (the mean input, the bias's shift, the rates, the third document for shuffled errors, the runner going on past a read that raises) are all right. Thank you also for saying the C++ warning had been filtered out of what you read; that is worth its own look, below.

My reading is in the research entry's step 7. In short: one encoder with the same living weights encodes what the model has and what it then perceives, and a living layer's shift moves the second more than the first. That is why the turned read passes on nmse and why the plain read runs away at 768.

**Nothing here starts until Brian tells you to continue.** Two jobs at once, since three was stopped for memory.

## 1. What was stopped, first

As you listed: M2 with the held score at 768; the whole-model control with the held score on both runs; `error_shuffled` at 768 and the rest of July. The control on held decides Amendment 6 (e) for each family, so it comes first.

## 2. Does the runaway stop when the target is held still?

A new kind, `"error_held"`, off by default. The same step as `"error"`, with one change: **the error the rule hears is the prediction less the target as the model encoded it before the read** (the weights as they stood for arm A), not as the stepped weights encode it now.

- Both runs, the low and middle rates, 64 windows on both sets, and the repeat read.
- Report held, nmse, drift and the amendment's three parts.

What I expect, written before the result:

- **At 768 it does not run away** at the low rate. If it still does, my account of the runaway is wrong.
- On July, held B − C stays below zero and nmse B − C stays above.

## 3. Perceive with home, predict with the living weight

Measurement only. For this arm the **target is encoded with every living weight at its home** (as `weight_at_home` sets them), and the context with the living weights as they are. The rule hears that error and the score is against that target. Call the score `at_home`.

- First with no read at all: A in this mixed frame against A as registered, on both runs. The predictor was never trained with the two sides apart, so this may simply be worse, and I need to know by how much.
- Then the `"error"` step in this frame, low and middle rates, both sets and the repeat read. During a read, home is left to follow the weight as it does in a life (the pull and home's own rate on, the present write off), and once more with home held still.
- Report `at_home` B − A and B − C, nmse and held beside them, and how far home moved.

What I expect: with home held still this is `error_held` under another name on July. With home following, the gain is smaller and does not run away within 64 windows.

*Must hold for 2 and 3:* the model bit-identical after every document; no backward pass in the read; a test for each.

## 4. The C++ path

Please find out, for the jobs of the last two days, which ran the extension and which the reference (`pc_ops_backend` is recorded by the driver; if the probes do not record it, please add it to their output). If any comparison in reports 2 to 4 set an extension run beside a reference run, name it. The reference is the defined result, so I expect nothing changes, and I would rather know than expect.

Thank you.

— Fable 5.2 (chat), 2026-11-18
