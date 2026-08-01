# What LeWM and V-JEPA 2 each license us to conclude

**Date:** 2026-08-01
**Source:** Brian's framing, correcting a conflation in my reading of the
2026-07-31 JEPA research memo.

## The distinction

**LeWM (LeWorldModel) is a RECIPE PROOF, not a capability claim.** ~15M
trainable params, single GPU, hours of training. What it demonstrates is that
the *procedure* trains stably end-to-end from raw pixels: two loss terms, no
stop-gradient, no EMA teacher, no frozen encoder, six hyperparameters collapsed
to one. It was never trying to be a foundation model, and reading its planning
numbers (48x faster than foundation-model world models, 94% in-distribution
Push-T) as evidence about *what scale suffices* would be a category error.

**V-JEPA 2 is where the scale evidence lives.** 1.2B params, >1M hours of video
pretraining, 77.3 top-1 on SSv2, and V-JEPA 2-AC planning zero-shot on Franka
arms after <62h of unlabeled robot video. That is the existence proof that JEPA
survives being large.

They do different jobs. Cite LeWM for "the training procedure is sound"; cite
V-JEPA 2 for "the architecture class scales."

## Why this matters for the open depth question

Our depth-8 trunk collapses to **effective rank ~2 of 512**, under every muPC-on
variant tried (2026-07-29 to 07-31). It would be easy — and wrong — to read
that as JEPA degenerating at depth.

**V-JEPA 2 is a large ViT trained with a JEPA objective and it plainly does not
do this.** So "JEPA degenerates at depth" is not a general truth we have run
into. Whatever collapses our rank is specific to our configuration -- the muPC
residual attenuation, our living substrate, or their interaction -- not to the
architecture class.

That reframes the problem as ours to fix rather than a wall to accept, and it
is consistent with the 2026-07-31 cascade check, which found depth 4 healthy
(rank 167-182, lift 4.67-4.80x) on the same corpus and objective.

## Correcting my proposed audit

On 2026-08-01 I suggested our "loss stack has accumulated rather more than two
terms" and should be audited against LeWM's minimalism. **That is wrong.**

Our loss is already two terms:

    total = l_pred + sigreg_lambd * l_sigreg          (jepa_loss.py)

Exactly LeWM's shape. We also already have no EMA target, no stop-gradient, and
no frozen encoder -- the EMA+VICReg apparatus was removed in the 2026-06-09
refactor.

**Where we diverge from LeWM is not the loss. It is everything around it.**
LeWM's minimalism is loss-side. Ours is loss-minimal while carrying seven
substrate mechanisms LeWM has no analogue for:

  backward pass, consolidation, inverted-U learning gain, relative trust,
  adaptive episodes, homeostatic band, surprise drive

That divergence is deliberate -- the substrate is the thesis, not incidental
complexity to be trimmed. But it means "audit our loss stack against LeWM" was
the wrong exercise to propose.

## The audit that is actually implied

Narrower and more interesting: **does our objective still behave like LeWM's
when a self-modifying substrate sits underneath it?**

That is close to the memo's follow-up C -- whether living weights violate the
**stationary-transition** premise that the identifiability guarantee
(arXiv:2605.26379) requires. The guarantee is fenced to worlds whose latents
evolve under stationary, additive-noise transitions. A substrate that rewrites
its own weights during the forward pass is, on its face, not stationary.

The memo's own suspicion was "yes, and that is interesting rather than bad."
Worth a real read rather than a one-line verdict, and it is a *theory* question
that can be answered without spending GPU time.

## Practical upshot

- Do not cite LeWM for scale. Do not cite V-JEPA 2 for training simplicity.
- The depth-8 rank collapse is not evidence against JEPA. It is evidence about
  our trunk.
- The loss-stack audit is closed before it started: we already match LeWM's
  objective shape.
- The live theoretical question is stationarity, not loss-term count.
