"""Base optimizer for NoTorch.

All optimizers operate directly on :class:`Tensor`-like objects exposing
``.data`` (np.ndarray), ``.grad`` (np.ndarray | None) and ``.zero_grad()``.
Only ``numpy`` + stdlib are used. This module never imports tensor/nn code;
parameters are duck-typed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

import numpy as np


class Optimizer(ABC):
    """Abstract base class for gradient-based optimizers.

    Args:
        params: Iterable of parameters, or iterable of param-group dicts.
            A param-group dict must contain a ``"params"`` key holding a
            parameter or iterable of parameters; other keys override defaults.
        defaults: Default hyperparameters stored on each param group.
    """

    def __init__(self, params: Iterable[Any], defaults: dict[str, Any]) -> None:
        self.defaults: dict[str, Any] = dict(defaults)
        self.state: dict[int, dict[str, Any]] = {}
        self.param_groups: list[dict[str, Any]] = []

        param_list = list(params)
        if len(param_list) == 0:
            raise ValueError("Optimizer got an empty parameter list.")
        if isinstance(param_list[0], dict):
            for group in param_list:
                self.add_param_group(group)
        else:
            self.add_param_group({"params": param_list})

    def zero_grad(self) -> None:
        """Clear gradients of all parameters."""
        for group in self.param_groups:
            for p in group["params"]:
                if getattr(p, "grad", None) is None:
                    continue
                zero_grad = getattr(p, "zero_grad", None)
                if callable(zero_grad):
                    zero_grad()
                else:  # Fallback for minimal tensor-likes.
                    p.grad = None

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        """Add a parameter group to the optimizer.

        Args:
            param_group: Dict with a ``"params"`` key (a parameter or
                iterable of parameters). Missing hyperparameters are filled
                from :attr:`defaults`.

        Raises:
            ValueError: If the group is malformed or a parameter is duplicated.
        """
        if not isinstance(param_group, dict):
            raise ValueError("param_group must be a dict.")
        if "params" not in param_group:
            raise ValueError("param_group must contain a 'params' key.")
        raw = param_group["params"]
        if isinstance(raw, dict):
            raise ValueError("'params' must be a parameter or iterable, not a dict.")
        if hasattr(raw, "data"):
            params = [raw]
        else:
            try:
                params = list(raw)
            except TypeError:
                raise ValueError("'params' must be a parameter or iterable of parameters.")
        if len(params) == 0:
            raise ValueError("param_group 'params' is empty.")
        for p in params:
            if not hasattr(p, "data"):
                raise ValueError("Each parameter must expose a '.data' attribute.")
            for group in self.param_groups:
                if any(p is q for q in group["params"]):
                    raise ValueError("A parameter appears in more than one param group.")

        group = dict(self.defaults)
        group.update({k: v for k, v in param_group.items() if k != "params"})
        group["params"] = params
        self.param_groups.append(group)

    @abstractmethod
    def step(self, closure: Any | None = None) -> Any | None:
        """Perform a single optimization step.

        Args:
            closure: Optional callable that reevaluates the model and
                returns the loss.

        Returns:
            The closure return value, or None.
        """
        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        """Return optimizer state keyed by ``id(param)``.

        Returns:
            Dict with ``"state"`` (``{id(param): per-param dict}`` with
            ndarrays deep-copied) and ``"param_groups"`` (hyperparameters
            plus ``id`` lists so :meth:`load_state_dict` can remap).
        """
        state = {
            pid: {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in s.items()}
            for pid, s in self.state.items()
        }
        param_groups = []
        for group in self.param_groups:
            packed = {k: v for k, v in group.items() if k != "params"}
            packed["params"] = [id(p) for p in group["params"]]
            param_groups.append(packed)
        return {"state": state, "param_groups": param_groups}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load optimizer state produced by :meth:`state_dict`.

        Param-group hyperparameters are restored in place; per-param state
        is remapped positionally onto the current parameters.

        Raises:
            ValueError: If the state is malformed or group counts mismatch.
        """
        if not isinstance(state_dict, dict) or "state" not in state_dict:
            raise ValueError("Invalid optimizer state_dict.")
        saved_groups = state_dict.get("param_groups", [])
        if len(saved_groups) != len(self.param_groups):
            raise ValueError("param_groups length mismatch in state_dict.")
        saved_state = state_dict["state"]
        new_state: dict[int, dict[str, Any]] = {}
        for group, saved in zip(self.param_groups, saved_groups):
            for key in list(group.keys()):
                if key != "params" and key in saved:
                    group[key] = saved[key]
            saved_ids = saved.get("params", [])
            if len(saved_ids) != len(group["params"]):
                raise ValueError("param count mismatch in state_dict.")
            for p, pid in zip(group["params"], saved_ids):
                if pid in saved_state:
                    s = saved_state[pid]
                    new_state[id(p)] = {
                        k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in s.items()
                    }
        self.state = new_state
