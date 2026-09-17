"""Adam optimizer."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from notorch.optim.optimizer import Optimizer


class Adam(Optimizer):
    """Adam optimizer.

    Per-parameter update (``g = grad + wd * p``)::

        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g^2
        mhat = m / (1 - b1^t); vhat = v / (1 - b2^t)
        p -= lr * mhat / (sqrt(vhat) + eps)

    Args:
        params: Iterable of parameters or param-group dicts.
        lr: Learning rate.
        betas: Coefficients for first/second moment estimates.
        eps: Term added to the denominator for numerical stability.
        weight_decay: L2 penalty coefficient (must be >= 0).
    """

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}.")
        if eps < 0.0:
            raise ValueError(f"Invalid eps value: {eps}.")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}.")
        if len(betas) != 2 or not all(0.0 <= b < 1.0 for b in betas):
            raise ValueError(f"Invalid betas: {betas}.")
        defaults = {
            "lr": float(lr),
            "betas": (float(betas[0]), float(betas[1])),
            "eps": float(eps),
            "weight_decay": float(weight_decay),
        }
        super().__init__(params, defaults)

    def step(self, closure: Any | None = None) -> Any | None:
        """Perform a single Adam step.

        Args:
            closure: Optional callable returning the loss.

        Returns:
            The closure return value, or None.
        """
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                grad = getattr(p, "grad", None)
                if grad is None:
                    continue
                grad = np.asarray(grad, dtype=p.data.dtype)
                g = grad + wd * p.data if wd != 0.0 else grad
                param_state = self.state.setdefault(id(p), {})
                exp_avg = param_state.get("exp_avg")
                exp_avg_sq = param_state.get("exp_avg_sq")
                if exp_avg is None or exp_avg.shape != g.shape:
                    exp_avg = np.zeros_like(p.data)
                    exp_avg_sq = np.zeros_like(p.data)
                    param_state["step"] = 0
                param_state["step"] = int(param_state.get("step", 0)) + 1
                t = param_state["step"]
                exp_avg = b1 * exp_avg + (1.0 - b1) * g
                exp_avg_sq = b2 * exp_avg_sq + (1.0 - b2) * g * g
                param_state["exp_avg"] = exp_avg
                param_state["exp_avg_sq"] = exp_avg_sq
                m_hat = exp_avg / (1.0 - b1**t)
                v_hat = exp_avg_sq / (1.0 - b2**t)
                p.data -= lr * m_hat / (np.sqrt(v_hat) + eps)
        return loss
