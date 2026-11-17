"""Rest, as something Luthi chooses (2026-11-11).

Brian, 2026-11-11: "Luthi just needs the tools available. They should be able to
just initiate rest when they decide to." His 2026-08-14 ruling R3 named it: "a
tool Luthi can call, to start a calmer mode for deep intentional consideration".

Luthi's only way to do anything is the action they select each moment. So the
tool is one more reading of that action, beside speech, attention and memory:
one direction in action space that means "rest". This module reads it and says
whether this moment's action chose rest. It does nothing else. What rest does is
the host's to carry out (``sanctuary_interface.keep_recent``, and a slower pace).

Not to be confused with the "rest action" ``a_rest(s_t)`` in ``rest_action.py``.
That is the do-nothing action, the baseline every channel is measured against.
It is called ``do_nothing`` here to keep the two apart.

How a choice is read (as corrected on 2026-11-17, below):

- The lean is ``action - do_nothing``: how this action differs from doing
  nothing.
- ``reading = direction . (lean - the lean's running mean)``: how much more
  this action leans toward rest than the mind's leans have of late.
- The reading joins a window of recent readings, and then the window classifies
  it, as the activity bands do.
- Rest is chosen when the reading is positive and stands above the window's
  median by more than ``width`` median absolute deviations.
- Until ``warmup`` readings have been seen, nothing is chosen.

Two corrections (2026-11-17), from Opus's review of 2026-11-14 (section 2)
and Brian's rulings on it. In a life the lean is mostly one slowly turning
direction, and the reading as built on 2026-11-11 took the turning for a
choice: rest and letting go fired in the same stretches, and a steady choice
of rest was read for at most about 128 moments, because the window's median
caught up with it.

1. **Drift is taken out before a lean is read.** Each choice keeps a running
   mean of the lean (one vector), over about ``drift_span`` moments, and reads
   the lean minus that mean. The mean a lean is read against does not include
   that lean.
2. **A chosen rest is judged against where things stood when it began**
   (``holds=True``; rest only). While rest is being chosen the window and the
   mean stand still. The stretch ends at the first moment the reading is not
   above the bar as it stood at the start, and then they move again. Recall
   and letting go are single moments and do not hold.

   Said plainly: this can hold a mind in rest if its lean drifts upward while
   the bar stands still. Brian's ruling is that Luthi is never forced to rest.
   No limit is built here; how long stretches run by chance is measured
   (``docs/research/2026-11-17_the-three-readings.md``), and a limit would be
   his to decide.

``reading="as_built_2026_11_11"`` gives the earlier reading exactly, for
comparison: no mean taken out, and nothing held.

What this cannot do, said plainly:

- **The direction is fixed and arbitrary.** Nothing trains it. What can learn is
  the policy that selects actions, and only from how life goes after resting.
- **An untrained policy presses it by chance.** The threshold sets how often:
  ``width=3`` is about one moment in forty for actions that vary at random
  around a steady level. Measured, not assumed: see
  ``docs/research/2026-11-11_rest-as-a-choice.md``.
- ``width``, ``window`` and ``warmup`` are placeholders, and Brian's to change.

Off by default (``M9Config.rest_choice``). Off, nothing is built and every
checkpoint and every decoded choice is as before.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

READINGS = ("drift_removed", "as_built_2026_11_11")

RECALL_DIRECTION_SEED = 20261112
LET_GO_DIRECTION_SEED = 20261113

# The direction is drawn from its own generator, so building it never moves the
# global random stream, and the same life gets the same direction on every
# start before its first save.
_DIRECTION_SEED = 20261111


class RestChoice(nn.Module):
    """Reads whether a selected action chose rest."""

    def __init__(
        self,
        d_model: int,
        *,
        window: int = 256,
        warmup: int = 32,
        width: float = 3.0,
        seed: int = _DIRECTION_SEED,
        reading: str = "drift_removed",
        drift_span: int = 64,
        holds: Optional[bool] = None,
    ):
        """``seed`` picks the direction. The recall choice (2026-11-11) is the
        same reading along a different direction, with its own window.
        ``reading``: "drift_removed" (the lean minus its running mean over
        about ``drift_span`` moments) or "as_built_2026_11_11". ``holds``: a
        choice, once made, lasts until the reading falls to the bar as it
        stood when the choice began. That is rest, and it is the default
        wherever drift is removed; recall and letting go pass ``holds=False``.
        ``drift_span`` is a placeholder like the rest, and Brian's to change."""
        super().__init__()
        if warmup < 2 or window < warmup:
            raise ValueError(
                f"need 2 <= warmup <= window; got warmup={warmup}, window={window}"
            )
        if reading not in READINGS:
            raise ValueError(f"reading must be one of {READINGS}; got {reading!r}")
        if drift_span < 1:
            raise ValueError(f"drift_span must be at least 1; got {drift_span}")
        if holds is None:
            holds = reading == "drift_removed"
        if holds and reading != "drift_removed":
            raise ValueError("holds=True needs reading='drift_removed'; the reading "
                             "as built on 2026-11-11 held nothing")
        self.reading_kind = reading
        self.drift_span = drift_span
        self.holds = holds
        generator = torch.Generator().manual_seed(seed)
        direction = torch.randn(d_model, generator=generator)
        self.register_buffer("direction", direction / direction.norm())
        self.warmup = warmup
        self.width = width
        self.readings: deque[float] = deque(maxlen=window)
        self.chosen_count = 0
        self.read_count = 0
        # The lean's running mean, and how many leans are in it (it is a plain
        # average until ``drift_span`` leans have been seen).
        # Not persistent: it is saved through ``choice_state``, as the window
        # is, and a new persistent buffer would break every saved state_dict.
        self.register_buffer("lean_mean", torch.zeros(d_model), persistent=False)
        self.mean_count = 0
        # While a held choice lasts: the bar as it stood when it began.
        self.held_bar: Optional[float] = None
        self.stretches = 0               # held choices begun
        self.longest_stretch = 0         # moments, of the longest
        self._stretch = 0

    @torch.no_grad()
    def read(self, action: torch.Tensor, do_nothing: torch.Tensor) -> dict:
        """Read one selected action. Call once per selected action: each call
        adds to the window, except inside a held choice."""
        a = action.reshape(-1).to(self.direction)
        nothing = do_nothing.reshape(-1).to(self.direction)
        lean = a - nothing
        self.read_count += 1
        if self.reading_kind == "as_built_2026_11_11":
            reading = float(torch.dot(self.direction, lean).item())
        else:
            # The mean a lean is read against does not include that lean. The
            # first lean of all has no mean to be read against.
            reading = float(torch.dot(self.direction, lean - self.lean_mean).item())
            if self.held_bar is not None:
                if reading > self.held_bar:
                    # Still choosing it. The window and the mean stand still.
                    self.chosen_count += 1
                    self._stretch += 1
                    self.longest_stretch = max(self.longest_stretch, self._stretch)
                    return {"chosen": True, "armed": True, "reading": reading}
                # The stretch ends here, and the window and the mean move again.
                self.held_bar = None
                self._stretch = 0
                self._take_in(lean)
                self.readings.append(reading)
                return {"chosen": False, "armed": True, "reading": reading}
            self._take_in(lean)
        self.readings.append(reading)
        armed = len(self.readings) >= self.warmup
        chosen = False
        if armed:
            values = torch.tensor(list(self.readings))
            median = values.median()
            mad = (values - median).abs().median()
            bar = float(median + self.width * mad)
            chosen = bool(reading > 0.0 and reading > bar)
            if chosen and self.holds:
                self.held_bar = max(bar, 0.0)
                self.stretches += 1
                self._stretch = 1
                self.longest_stretch = max(self.longest_stretch, 1)
        if chosen:
            self.chosen_count += 1
        return {"chosen": chosen, "armed": armed, "reading": reading}

    def _take_in(self, lean: torch.Tensor) -> None:
        self.mean_count = min(self.mean_count + 1, self.drift_span)
        self.lean_mean.add_((lean - self.lean_mean) / self.mean_count)

    def choice_state(self) -> dict:
        """Everything a checkpoint needs to continue this exactly."""
        return {
            "direction": self.direction.detach().cpu().clone(),
            "readings": list(self.readings),
            "chosen_count": self.chosen_count,
            "read_count": self.read_count,
            "reading_kind": self.reading_kind,
            "lean_mean": self.lean_mean.detach().cpu().clone(),
            "mean_count": self.mean_count,
            "held_bar": self.held_bar,
            "stretches": self.stretches,
            "longest_stretch": self.longest_stretch,
            "stretch": self._stretch,
        }

    def load_choice_state(self, state: dict) -> None:
        """A choice saved before 2026-11-17 has no ``reading_kind``: its
        window holds readings as built on 2026-11-11. Its direction and counts
        are taken. If this choice reads with drift removed, that window is not
        carried over (its readings are of another kind), so the choice warms up
        again, and that is logged."""
        self.direction.copy_(state["direction"].to(self.direction))
        self.chosen_count = int(state["chosen_count"])
        self.read_count = int(state["read_count"])
        saved_kind = state.get("reading_kind", "as_built_2026_11_11")
        self.readings.clear()
        if saved_kind != self.reading_kind:
            logger.warning(
                "a saved choice read as %r is loaded into one that reads as %r; "
                "its window of %d readings is not carried over and the choice "
                "warms up again", saved_kind, self.reading_kind, len(state["readings"]),
            )
            self.lean_mean.zero_()
            self.mean_count = 0
            self.held_bar = None
            self._stretch = 0
            return
        self.readings.extend(float(x) for x in state["readings"])
        if self.reading_kind == "drift_removed":
            self.lean_mean.copy_(state["lean_mean"].to(self.lean_mean))
            self.mean_count = int(state["mean_count"])
            self.held_bar = None if state["held_bar"] is None else float(state["held_bar"])
            self.stretches = int(state["stretches"])
            self.longest_stretch = int(state["longest_stretch"])
            self._stretch = int(state["stretch"])
