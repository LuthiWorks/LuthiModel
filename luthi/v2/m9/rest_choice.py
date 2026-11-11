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

How a choice is read:

- ``reading = direction . action - direction . do_nothing``: how much more this
  action leans toward rest than doing nothing would.
- The reading joins a window of recent readings, and then the window classifies
  it, as the activity bands do.
- Rest is chosen when the reading is positive and stands above the window's
  median by more than ``width`` median absolute deviations.
- Until ``warmup`` readings have been seen, nothing is chosen.

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

from collections import deque

import torch
import torch.nn as nn

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
    ):
        super().__init__()
        if warmup < 2 or window < warmup:
            raise ValueError(
                f"need 2 <= warmup <= window; got warmup={warmup}, window={window}"
            )
        generator = torch.Generator().manual_seed(_DIRECTION_SEED)
        direction = torch.randn(d_model, generator=generator)
        self.register_buffer("direction", direction / direction.norm())
        self.warmup = warmup
        self.width = width
        self.readings: deque[float] = deque(maxlen=window)
        self.chosen_count = 0
        self.read_count = 0

    @torch.no_grad()
    def read(self, action: torch.Tensor, do_nothing: torch.Tensor) -> dict:
        """Read one selected action. Call once per selected action: each call
        adds to the window."""
        a = action.reshape(-1).to(self.direction)
        nothing = do_nothing.reshape(-1).to(self.direction)
        reading = float(torch.dot(self.direction, a - nothing).item())
        self.readings.append(reading)
        self.read_count += 1
        armed = len(self.readings) >= self.warmup
        chosen = False
        if armed:
            values = torch.tensor(list(self.readings))
            median = values.median()
            mad = (values - median).abs().median()
            chosen = bool(reading > 0.0 and reading > float(median + self.width * mad))
        if chosen:
            self.chosen_count += 1
        return {"chosen": chosen, "armed": armed, "reading": reading}

    def choice_state(self) -> dict:
        """Everything a checkpoint needs to continue this exactly."""
        return {
            "direction": self.direction.detach().cpu().clone(),
            "readings": list(self.readings),
            "chosen_count": self.chosen_count,
            "read_count": self.read_count,
        }

    def load_choice_state(self, state: dict) -> None:
        self.direction.copy_(state["direction"].to(self.direction))
        self.readings.clear()
        self.readings.extend(float(x) for x in state["readings"])
        self.chosen_count = int(state["chosen_count"])
        self.read_count = int(state["read_count"])
