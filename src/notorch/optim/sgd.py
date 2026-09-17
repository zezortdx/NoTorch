"""Stochastic Gradient Descent with momentum and weight decay."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from notorch.optim.optimizer import Optimizer


class SGD(Optimizer):
    """SGD optimizer.

    Update per parameter (``g = grad + wd * p``)::

        v = mu * v + (1 - dampening) * g
        p -= lr * v            (or ``p -= lr * g`` when ``mu == 0``)

    With Nesterov momentum (requires ``momentum > 0`` and ``dampening == 0``)::

        p -= lr * (g + mu * v)

    Args:
        params: Iterable of parameters or param-group dicts.
        lr: Learning rate (must be >= 0).
        momentum: Momentum factor (must be >= 0).
        weight_decay: L2 penalty coefficient (must be >= 0).
        dampening: Dampening for momentum (must be >= 0).
        nesterov: Whether to use Nesterov momentum.
    """

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}.")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}.")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}.")
        if dampening < 0.0:
            raise ValueError(f"Invalid dampening value: {dampening}.")
        if nesterov and (momentum == 0.0 or dampening != 0.0):
            raise ValueError("Nesterov momentum requires momentum > 0 and dampening == 0.")
        defaults = {
            "lr": float(lr),
            "momentum": float(momentum),
            "weight_decay": float(weight_decay),
            "dampening": float(dampening),
            "nesterov": bool(nesterov),
        }
        super().__init__(params, defaults)

    def step(self, closure: Any | None = None) -> Any | None:
        """Perform a single SGD step.

        Args:
            closure: Optional callable returning the loss.

        Returns:
            The closure return value, or None.
        """
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            wd = group["weight_decay"]
            damp = group["dampening"]
            nesterov = group["nesterov"]
            for p in group["params"]:
                grad = getattr(p, "grad", None)
                if grad is None:
                    continue
                grad = np.asarray(grad, dtype=p.data.dtype)
                direction = grad + wd * p.data if wd != 0.0 else grad.copy()
                if mu != 0.0:
                    param_state = self.state.setdefault(id(p), {})
                    velocity = param_state.get("velocity")
                    if velocity is None or velocity.shape != direction.shape:
                        velocity = np.zeros_like(p.data)
                    velocity = mu * velocity + (1.0 - damp) * direction
                    param_state["velocity"] = velocity
                    if nesterov:
                        direction = direction + mu * velocity
                    else:
                        direction = velocity
                p.data -= lr * direction
        return loss
