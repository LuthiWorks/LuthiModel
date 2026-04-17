"""DirectML-safe AdamW optimizer.

PyTorch's built-in AdamW uses aten::lerp.Scalar_out for its exponential
moving average updates. DirectML does not support this op, causing every
optimizer step to fall back to CPU — a GPU→CPU→GPU round-trip that
destroys training throughput and can trigger TDR (GPU driver timeouts).

This module provides DirectMLAdamW: a functionally identical AdamW that
replaces lerp(a, b, t) with the equivalent mul_(1-t).add_(b, alpha=t),
which DirectML handles natively.

Math equivalence:
    lerp(a, b, t) = a + t * (b - a) = a * (1 - t) + b * t
    a.lerp_(b, t) <==> a.mul_(1 - t).add_(b, alpha=t)
"""

import math
from typing import Optional

import torch
from torch.optim.optimizer import Optimizer


class DirectMLAdamW(Optimizer):
    """AdamW optimizer without aten::lerp — safe for DirectML.

    Implements the same algorithm as torch.optim.AdamW with decoupled
    weight decay (Loshchilov & Hutter, 2019). All tensor operations use
    only mul_, add_, addcmul_, sqrt, and div — no lerp.

    Args:
        params: Iterable of parameters or param groups.
        lr: Learning rate (default: 1e-3).
        betas: Coefficients for computing running averages of gradient
            and its square (default: (0.9, 0.999)).
        eps: Term added to denominator for numerical stability (default: 1e-8).
        weight_decay: Decoupled weight decay coefficient (default: 1e-2).
        amsgrad: Whether to use the AMSGrad variant (default: False).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss
                (optional).

        Returns:
            Loss value from closure if provided, else None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            amsgrad = group["amsgrad"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("DirectMLAdamW does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                # Decoupled weight decay (applied to params directly)
                if weight_decay != 0:
                    p.mul_(1.0 - lr * weight_decay)

                # Update biased first moment estimate
                # lerp equivalent: exp_avg = exp_avg * beta1 + grad * (1 - beta1)
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                # Update biased second raw moment estimate
                # lerp equivalent: exp_avg_sq = exp_avg_sq * beta2 + (grad^2) * (1 - beta2)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Bias correction
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step

                step_size = lr / bias_correction1

                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                # Parameter update
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
