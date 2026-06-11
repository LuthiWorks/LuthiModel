"""Habit network -- Fountas-style amortized proposal distribution.

`HabitNet`: state `s_t` -> K candidate actions `a_t` per the spec
§2.3 / spec §3 build list item 2 ("`s_t -> K candidate a_t`;
amortized policy; distilled from MCTS"). The habit net does the
sample-efficiency heavy lifting that lets the 256-d continuous
action space be searched at all: instead of sampling actions
uniformly (curse of dimensionality), the habit net concentrates
proposals near plausible actions and MCTS expands from there with
progressive widening.

Architecture (step 1 simple form):
- Trunk: 2-layer MLP from [B, D] state to [B, hidden] features.
- Mean head: [hidden] -> [B, D] action mean per state.
- Log-std head: [hidden] -> [B, D] per-dim log-std per state.
- Sample K times via the reparameterization trick: eps ~ N(0, 1),
  a = mean + std * eps. The samples are differentiable w.r.t.
  mean / log_std so distillation (KL toward MCTS-selected actions
  or behaviour cloning on top-Q candidates) flows gradients back.

Training story (downstream; spec doesn't pin a single method):
- Behaviour-cloning regression on top-EFE candidates from MCTS,
  OR
- Cross-entropy distillation toward the MCTS visit distribution
  over candidates,
  OR
- Direct policy gradient using V(s) as critic.

At launch the habit net starts under-trained, so std init should
be wide enough for early MCTS expansion to see useful variety
without overwhelming the optimizer. Default `log_std_init = 0.0`
gives unit-variance samples; tunable.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class HabitNet(nn.Module):
    """State -> per-state Gaussian over actions; samples K candidates."""

    def __init__(
        self,
        d_model: int,
        hidden_dim: int | None = None,
        log_std_init: float = 0.0,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.d_model = d_model
        if hidden_dim is None:
            hidden_dim = d_model * 2
        self.trunk = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
        )
        self.mean_head = nn.Linear(hidden_dim, d_model)
        self.log_std_head = nn.Linear(hidden_dim, d_model)

        # Stable init: log_std_head bias to log_std_init (state-
        # independent prior on std); weights small so state has
        # little leverage on std at start.
        with torch.no_grad():
            self.log_std_head.bias.fill_(log_std_init)
            nn.init.normal_(self.log_std_head.weight, std=1e-2)

        # Clamp bounds to keep the Gaussian well-conditioned.
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """s: [B, D] -> (mean [B, D], log_std [B, D]).

        log_std is hard-clamped to [log_std_min, log_std_max] so the
        sampler doesn't blow up under bad parameter updates. The
        clamp is differentiable inside the open interval.
        """
        h = self.trunk(s)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(
            min=self.log_std_min, max=self.log_std_max
        )
        return mean, log_std

    def sample(
        self,
        s: torch.Tensor,
        K: int,
        generator: torch.Generator | None = None,
    ) -> dict:
        """Reparameterized samples + log-probs.

        Returns dict:
          mean         : [B, D]
          log_std      : [B, D]
          candidates   : [B, K, D] differentiable w.r.t. mean/log_std
          log_prob     : [B, K] log Gaussian density per sample
        """
        mean, log_std = self.forward(s)
        std = log_std.exp()
        B, D = mean.shape

        if generator is None:
            eps = torch.randn(B, K, D, device=s.device)
        else:
            eps = torch.randn(
                B, K, D, device=s.device, generator=generator
            )
        # Reparameterized sample.
        candidates = mean.unsqueeze(1) + std.unsqueeze(1) * eps  # [B, K, D]

        # Gaussian log-density:
        # log p(x) = -0.5 * sum_i[ eps_i^2 + log(2*pi) + 2*log(std_i) ]
        # Per-sample [B, K]:
        log_prob = (
            -0.5 * (eps.pow(2)).sum(dim=-1)
            - 0.5 * D * math.log(2.0 * math.pi)
            - log_std.sum(dim=-1).unsqueeze(1)
        )

        return {
            "mean": mean,
            "log_std": log_std,
            "candidates": candidates,
            "log_prob": log_prob,
        }

    def entropy(self, s: torch.Tensor) -> torch.Tensor:
        """Differential entropy of the per-state Gaussian.

        H[N(mu, sigma^2 I)] = 0.5 * D * log(2*pi*e) + sum_i log(sigma_i).
        Returns: [B] entropy.
        """
        _, log_std = self.forward(s)
        D = log_std.shape[-1]
        constant = 0.5 * D * math.log(2.0 * math.pi * math.e)
        return constant + log_std.sum(dim=-1)
