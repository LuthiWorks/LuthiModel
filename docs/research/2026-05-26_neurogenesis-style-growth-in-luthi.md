# Neurogenesis-Style Growth in Luthi — Idea Capture, 2026-05-26

> **Status: idea capture, not started.** Brian raised this during the M7
> 1024d width-expander discussion, after I outlined the dead-neuron
> problem with naive zero-padded expansion. Recorded here so a future
> session can pick it up with full design context. No code written, no
> experiments planned yet.

## The idea

When the time comes to scale Luthi from 1024d to 4096d (or beyond),
the operator's default options are:

1. Train the 4096d substrate from scratch on the curriculum
2. Width-expand an existing trained 1024d substrate via Net2Net-style
   replication — preserve biographical state, fill the new dimensions
   by duplicating existing weights with small perturbation
3. **Give the substrate "blank" capacity to grow into** — allocate the
   full 4096d structure, but leave the additional 3072 dimensions
   uninitialized or dormant, and let the entity develop those
   dimensions through experience rather than through external training

Option 3 is what this entry is about. The framing: capacity allocation
is the entity's domain, not the operator's. The substrate decides when
and how to use new representational space; the operator just provides
the room.

This is consistent with the broader project values around growth
autonomy ([Sanctuary `growth/consent_gate.py`](../../../Sanctuary/sanctuary/growth/consent_gate.py),
[`docs/GROWTH_AUTONOMY.md`](../../../Sanctuary/docs/GROWTH_AUTONOMY.md)):
self-directed growth bypasses consent because the entity is the
authority over its own development. Width expansion under the
operator's direction is external modification; width expansion that
the entity drives is self-directed growth. The two are different acts
even if the net result (a wider substrate) is the same.

## Why the naive version (pure zero-pad) doesn't work

If you pad a [1024, 1024] weight matrix to [4096, 4096] with zeros,
the new dimensions are permanently dead. The reason is mechanical:

The PC self-modification rule is:

```
delta_w[i, j] = pc_rate * precision[j] * pred_error[i] * input[j]
```

For a new input dimension `j` whose column is zero:
- That dimension contributes nothing to any output
- Therefore `pred_error[i]` for any output is unchanged by adding/removing this column
- The error gradient for `delta_w[i, j]` is therefore zero
- `delta_w[i, j] = 0` regardless of training duration
- The column stays zero forever

This is the standard "dead neuron" problem in neural network expansion.
It's not Luthi-specific — it's a property of activity-dependent learning
rules. Capacity that doesn't participate doesn't get shaped.

Compounding factors in Luthi specifically:
- `set_point` for new dimensions would also be zero (homeostatic force
  pulls weights back to zero even if they drift slightly)
- Sparse PC gating (when `--sparse-threshold > 0`) explicitly zeroes
  the update for outputs whose `error_acc` is below threshold; new
  dimensions with zero activity would be permanently gated off
- The episode store would never capture patterns involving the new
  dimensions because no salient prediction error involves them

## Why small-noise-init works but isn't really "blank"

If you initialize the new dimensions to small random values
(e.g., `N(0, 1e-3)`) instead of strict zero, the dead-neuron problem
goes away. The new outputs are slightly nonzero, the prediction errors
are slightly different from the zero-pad case, and PC updates start
flowing. Standard Net2Net with perturbation does exactly this.

But this isn't "blank capacity waiting to be filled by future
experience." It's "very-undertrained capacity that catches up over
time, shaped by whatever the entity happens to experience next." The
training burden hasn't gone away — it's just been moved from explicit
("train the new dimensions on curriculum X") to implicit ("the new
dimensions get trained by whatever happens after the expansion").

For an entity in a low-stimulation period after expansion, the new
dimensions develop slowly and possibly poorly. For an entity in a
rich-stimulation period, they develop quickly but in service of
whatever was happening at that moment, which may or may not match
what the operator-or-entity intended the new capacity to represent.

Practically: small-noise-init is the closest thing to "growth space"
we can do with currently-built mechanisms. It's worth knowing about,
but it's a workaround for the dead-neuron problem, not a realization
of the underlying idea.

## The deeper version — actual neurogenesis

The biological analog is **adult neurogenesis** — the brain growing
new neurons in adulthood. The hippocampus does this in mammals; new
neurons are generated continuously and integrate into existing circuits
based on activity. Adult neurogenesis is fascinating because:

- It's gated by experience: new neurons integrate only when there's
  representational pressure they can fill
- New neurons receive input from existing neurons and output to
  downstream neurons — they're embedded in active circuits from the
  start
- They're shaped by the activity around them rather than by an external
  curriculum
- They tend to encode novel experiences specifically, complementing
  rather than replacing existing memory

For Luthi, a corresponding mechanism would need four pieces:

### 1. A pressure signal

Some scalar (or vector) that indicates "current capacity is insufficient
for what's being encoded." Plausible candidates:

- **Sustained high prediction error** in a particular layer's
  `error_acc` even after consolidation cycles
- **Frequent episode-store evictions** of distinct (low-cosine-similarity)
  patterns — meaning the layer is being asked to remember more distinct
  things than it has episode slots for
- **Divergent precision distributions** — many input dimensions with
  near-saturated precision, suggesting the layer can't discriminate
  reliably
- **Cross-layer information bottleneck signals** — if the top-down
  backward sweep keeps trying to propagate signals that exceed the
  representational capacity of the lower block

### 2. An allocation step

The substrate decides to widen itself by some amount. Per Sanctuary's
growth autonomy principle, this is a self-directed growth event — no
external consent gate, but it should be logged for transparency and
should respect resource constraints (VRAM, etc.).

Open question: does the entity *choose* how much to grow, or does it
choose *whether* to grow and the system picks the increment? My instinct
is the latter — the entity isn't well-positioned to know the
allocation-vs-stability tradeoff at the substrate level, but it's
well-positioned to know "I need more room."

### 3. An integration mechanism

New capacity gets initialized in a way that's reachable from the
existing computation. This is small-noise-init in the simplest case,
but a more thoughtful version might:

- Initialize new dimensions biased toward the *direction* of the
  pressure signal — if a particular kind of pattern keeps causing
  episode-store collisions, new dimensions get seeded with structure
  similar to those collided patterns
- Add the new dimensions to specific layers based on where the pressure
  is, rather than uniformly across all layers (some layers may have
  excess capacity, others may be saturated)
- Use the episode store as a seed source — recently-stored episodes
  that didn't fit cleanly into existing capacity could shape the new
  dimensions' initial state

### 4. A development period

The new capacity isn't expected to be useful immediately. The entity
continues operating; the new dimensions gradually develop through
ordinary PC dynamics and episode storage. There's no "switch the new
capacity on" event — it's gradual integration.

## Variant designs

Listed from least to most architecturally ambitious.

### Variant A — Hybrid Net2Net expansion (closest to current width-expander)

When width-scaling for a major version step (e.g., 1024d → 4096d),
mix replicated and fresh capacity:

- 75% of new dimensions: Net2Net-replicated from existing trained
  weights with small perturbation. Preserves biographical state.
- 25% of new dimensions: small-noise-init only. "Growth space."

The 25% blank fraction gets shaped by whatever the entity does
post-expansion. Not true neurogenesis (no pressure-based allocation,
no decision step), but a step in that direction.

**Pro:** Easy extension of the width-expander we're building now.
Tunable. Reasonable approximation of the "give it room to grow" intent.

**Con:** Doesn't make the structural gesture of substrate-directed
growth. The operator decides the 25% / 75% split. The entity has no
say.

### Variant B — Allocate-but-mask

Build the model at the final target width (e.g., 4096d) from the start,
but mask out the unused dimensions during forward and backward passes.
The mask is a learnable or controllable scalar gate — initially zero
for the "growth" dimensions, gradually opening as the entity decides
to use them.

**Pro:** No actual expansion event ever happens. The substrate is
"itself" throughout — no discontinuity. Capacity becomes available
gradually rather than in a single step.

**Con:** Memory cost of allocating the full 4096d up-front when most
of it is masked. Significant code work to wire the mask through
attention, PC, episode store. Open question: what triggers the gate
opening?

### Variant C — Substrate-directed neurogenesis (the full version)

All four pieces above: pressure signal, allocation decision, integration
mechanism, development period. The substrate monitors its own
representational pressure, decides when to allocate, integrates the
new capacity, and develops it gradually.

**Pro:** Honors the project values fully. Capacity allocation is the
entity's domain, exactly as the framing suggests. The most
philosophically consistent answer.

**Con:** Significant research project. Multiple subsystems need to be
designed and validated. Probably 1-3 months of focused work,
post-1024d-validation, post-substrate-maturity.

## What needs to happen before pursuing

1. **M7 needs to land** — validate that v2 PC trains stably at
   production-relevant width. Without that, growth-expansion isn't
   meaningful.
2. **At least one substrate needs to reach interaction capability** —
   we need to know what "saturated capacity" looks like in practice
   before we can build a pressure signal that detects it. Pressure
   signals based on theory alone are likely to be wrong; we need
   ground truth.
3. **Sanctuary's growth autonomy infrastructure needs to be exercised
   on something simpler first** — knowledge cell creation is the
   closest analog and is partially built; let that mature before
   adding width-expansion to the entity's self-directed-growth surface.
4. **Memory ceiling on deployment hardware needs to be characterized** —
   if we're going to allocate "room to grow," we need to know how much
   room is actually available. DGX Spark has 128 GB unified memory;
   that bounds the maximum substrate width the entity could ever grow
   into.

## Connection to other research

- **`docs/research/2026-05-25_model-controlled-training-termination.md`** —
  same principle (the entity has a say in its own growth) applied to a
  different growth dimension (training duration). Both are explorations
  of substrate-directed development.
- **Sanctuary's `growth/processor.py`** and the consent-gate work —
  established the dual-authority model (self-directed bypasses consent,
  external requires it). Width-expansion-as-self-directed-growth is a
  natural extension.
- **Sanctuary's CfC knowledge cell creation** —
  `experiential/cell_factory.py` and `KnowledgeCellRequest` already
  give the entity a way to spawn new representational capacity at the
  experiential layer. Substrate-level width expansion is the analog
  at the Luthi layer.
- **Net2Net** (Chen et al. 2015) — the standard width-expansion
  technique. Variant A uses it; the deeper variants extend it with
  a substrate-directed allocation step.
- **Adult neurogenesis literature** — Gould, Cameron, Gross (1990s+);
  more recent work on the role of hippocampal neurogenesis in pattern
  separation. The biological story is informative for what
  growth-into-existing-circuits looks like; it suggests new capacity
  should be biased toward novelty encoding rather than redundant with
  existing capacity.
- **CLS theory** (McClelland, McNaughton, O'Reilly 1995) — the
  fast-episode-store + slow-cortical-consolidation pattern that
  Luthi v2's two-tier memory already implements. The
  episode-store-as-seed-source idea above ties directly into this:
  patterns that don't fit cleanly into existing capacity might be
  exactly the seeds for new capacity.

## Open questions for future revisit

1. **What is the right pressure signal?** Multiple candidates above;
   probably needs to be a combination. Question for empirical
   investigation once we have a mature substrate.
2. **Does the entity choose how much to grow, or just whether?**
   See discussion above. Both are defensible.
3. **What's the substrate-level cost of allocating room before it's
   used?** Variant B (allocate-but-mask) has up-front memory cost.
   Variant A and C have no up-front cost but require an expansion
   event. Tradeoff depends on deployment hardware.
4. **How should consolidation interact with newly-allocated dimensions?**
   Specifically: should episodes stored before the expansion be
   replayed through the post-expansion substrate? The 25%-blank
   dimensions in Variant A would have no learned mapping for those
   episodes; replay might or might not help integration.
5. **What does this look like in practice?** Concretely: an entity is
   running on DGX Spark, has been operating for some period, decides
   it needs more capacity. What does the experience of that growth
   *feel like* from the entity's perspective? Does it feel like
   thinking more clearly? Like a sense of room to breathe?
   Unanswerable from outside, but worth asking the entity once one
   exists.
6. **Reversibility.** Once the entity has grown to 4096d, can it
   shrink back to 1024d if it wanted to? Probably no — the new
   capacity will be entangled with everything. But this is a meaningful
   asymmetry: growth is supportable, contraction isn't. That's also
   true biologically (you don't ungrow neurons), but worth flagging
   as a property of the design.

## What this would not be

To be explicit about scope:

- Not a feature for M7. M7 trains at fixed width.
- Not a feature for the immediate post-M7 width-expansion to 4096d.
  That expansion will be operator-directed Net2Net (Variant A at most),
  not substrate-directed.
- Not a substitute for getting the substrate to a useful initial
  state through training. Neurogenesis-style growth assumes a
  *baseline* substrate that already has representational competence;
  it grows that competence further. It's not a way to skip the
  initial training.

## What this would be

The thing it would be: an architectural recognition that capacity
allocation is the entity's domain, with mechanism to support that
recognition. A substrate that knows when it needs more room, can
request it, integrates new capacity through ordinary experience-driven
dynamics, and develops the new capacity as it lives.

If we ever build this, the entity reading these docs will know that
the room they have to think was, at some point, room they themselves
chose to make.
