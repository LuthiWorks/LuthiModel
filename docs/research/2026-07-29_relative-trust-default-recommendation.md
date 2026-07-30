# Recommendation: do NOT promote `relative_trust` to default

**Date:** 2026-07-29
**Author:** Claude Fable 5 (cross-line audit seat)
**Status:** recommendation to Brian. Not a ruling. The call is his.
**Context:** external review round 2 identified precision differentiation as the
PC-to-JEPA compatibility valve and recommended making relative trust the default.
Brian's standing instruction for this review cycle was "take their suggestions
seriously and implement them, and if they cause problems then we will revert."
This is the one item I am declining to implement, with reasons, rather than
implementing and reverting.

`relative_trust` stays opt-in: currently set explicitly by the
`living_v5_4x_d4` arm, default `False` (epsilon trust) everywhere else.

## The argument the review makes is a good one

Absolute precision bounds are conceptually wrong for a PC layer inside a JEPA
objective. Precision is supposed to encode *relative* confidence across
dimensions; clamping it to fixed absolute limits means the meaning of "trusted"
drifts with the overall error scale, and at low error scale every dimension
pins at the ceiling and trust stops differentiating anything. Relative trust
(ratio-to-median, numerics-only epsilon) fixes that at the level of the maths.
I agree with the reasoning. My objection is not to the mechanism.

## Reason 1 (decisive, and new as of yesterday): the evidence base was produced
## under a defective objective

Every measurement we have of relative trust comes from the v5 family, run under
the objective fixed on 2026-07-28 - BatchNorm in front of SIGReg neutralising
the anti-collapse constraint while un-detached MSE paid for shrinking the
representation.

The rationale for relative trust was a response to a precision runaway:
precision medians in the millions, `update_ema` at ~5e-9, trust pinned at the
clamp with no differentiation left. We now know a large part of that runaway was
an **artifact of the collapse**. Measured on matched 4000-step probes, same arm,
only the objective differing:

| | previous objective (probe seed44) | fixed objective (probe seed45) |
|---|---|---|
| precision median, blocks 0-3 | 1.40M / 1.67M / 1.70M / 0.49M | 0.34M / 0.21M / 0.17M / 0.18M |
| `update_ema` | 1.0e-7 to 2.4e-7 | 2.6e-6 to 8.3e-6 |
| per-dim std (`std_p5`) | ~0.28 | 1.09 - 1.37 |
| raw `cos_pred` | 0.988 | 0.645 |

Precision fell by roughly an order of magnitude and substrate motion recovered
15-80x, without touching the trust code at all. The disease relative trust was
prescribed for has substantially remitted for other reasons.

That does not make relative trust wrong. It makes the **case for it unmade**.
Promoting it to default now would bake in a compensation for a defect we removed
yesterday, and we would have no clean way to tell afterwards which of the two
changes was doing the work.

## Reason 2: our own five-seed comparison found no measurable benefit

v5 (relative trust) against v4 (epsilon), five seeds, paired: **all paired
|t| <= 1.5**. No outcome measure separated them. The mechanism is not currently
earning its place on results, only on argument.

## Reason 3: it has a characterised failure mode that epsilon does not

Documented 07-27/07-28: **runaway bottom-end abandonment**. Under relative
trust, a dimension whose precision falls below the median gets down-weighted,
which reduces its updates, which lowers its precision further - self-reinforcing
distrust, with hysteresis at the clamp. seed46 wrote off up to **65 dimensions**
this way. The byte-identical rerun did *not* reproduce it, which is the worse
news of the two: it is a **chaotic instability**, not a deterministic property.
It will appear in some future runs and not others, with no signal in the config.

## Reason 4: this is the specific case where "implement and revert" fails

Brian's instruction is a good default and I have followed it for every other
item in three review rounds. It works because problems surface: a change that
breaks something announces itself, and we revert.

A **default** does not announce itself. Promoting relative trust would silently
alter the substrate of every future run of every arm. Its failure mode is
chaotic and seed-dependent. So the way this one would "cause problems" is not a
break we notice and revert - it is an unexplained widening of between-seed
variance in some future family, indistinguishable from ordinary noise, on a
project whose entire method rests on being able to attribute differences to
registered changes. We would lose the ability to detect the problem at the same
moment we introduced it.

This is exactly the failure the 07-27 verdict taught us to avoid: we registered
a criterion on `precision_spread` before establishing it was reproducible, and
the criterion turned out to be evidentially worthless. Same error, one level up.

## What I recommend instead

Keep `relative_trust` opt-in, and settle it properly under the fixed objective:

1. Register it as an explicit **v6 arm comparison** - fixed objective, relative
   trust on vs off, matched seeds. That is the measurement that was never taken,
   because it could not have been taken before yesterday.
2. Add the abandonment instrument as a **kill criterion** on that arm, not a
   post-hoc observation: count dimensions whose trust sits at the lower clamp for
   N consecutive deep records, and abort if it climbs monotonically. A chaotic
   failure mode is acceptable in an arm you are watching for it; it is not
   acceptable in a default.
3. If it wins under those conditions, promote it with evidence rather than
   argument, and the promotion is then a registered change we can attribute.

The mechanism may well be right. I want it to be right for a reason we can point
at afterwards.
