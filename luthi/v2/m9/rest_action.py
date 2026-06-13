"""§6.i `a_rest`: context-dependent rest action.

Per the M9 step-1 spec (§6.i, paraphrased): *rest is computed, not a
fixed constant.* `a_rest(s_t)` = "predict minimal self-change" =
the identity-continuation action for the current self-state. A
fixed learned constant can't track context; the entity's
silence-baseline depends on what state it's currently in.

This module parameterizes `a_rest(s_t)` as a tiny MLP and provides
the M9 head training objective for it -- minimize
`‖predict_next(s_t, a_rest(s_t)) - s_t‖_internal`. Once trained,
two cycle-time references fall out for free:

- `decode(a_rest(s_t))` -> the per-modality activity at rest. This
  is the **context-dependent silent reference** that
  `ActivityBands.external_stasis(..., rest_activity=...)` reads:
  current activity at-or-below rest_activity = stasis. Replaces
  the round-2 absolute_silent_floor (which was the interim
  context-independent backstop) for the K-M9-5 dark-room kill.
- `‖predict_next(s_t, a_rest(s_t)) - s_t‖_internal` -> the
  minimal `‖Δs_internal‖` the entity can produce. Feeds
  `DeltaSBand.is_silent_per_batch(..., rest_delta_s=...)` as the
  context-dependent internal-stasis floor.

Stop-grad contract (per spec §1): the rest forward goes through
the M8 predictor, but the rest loss must not update the predictor.
The runner zeros loss_module grads after the M9 backward so the
predictor receives M8 gradient only. The rest_action net itself is
in the M9 optimizer.

Init: the final layer is zero-init so `a_rest(s_t) ~= 0` at launch
(a plausible "do nothing" prior). Training reshapes it into the
predicted-minimal-self-change action per state.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RestActionNet(nn.Module):
    """State -> a_rest(s_t). Point-estimate MLP (no log_std)."""

    def __init__(self, d_model: int, hidden_dim: int | None = None):
        super().__init__()
        self.d_model = d_model
        if hidden_dim is None:
            hidden_dim = d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        # Zero-init the output projection so a_rest starts at 0.
        with torch.no_grad():
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """[B, D] -> [B, D] -- the predicted minimal-self-change action."""
        return self.net(s)
