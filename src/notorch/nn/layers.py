"""Basic neural network layers: :class:`Linear` and :class:`Flatten`."""

from __future__ import annotations

import numpy as np

from ..tensor import Tensor
from .init import xavier_uniform_
from .module import Module
from .parameter import Parameter


class Linear(Module):
    """Fully-connected layer computing ``y = x @ W + b``.

    The weight ``W`` has shape ``(in_features, out_features)`` so inputs of
    shape ``(..., in_features)`` map to ``(..., out_features)``; the bias
    ``b`` has shape ``(out_features,)`` and broadcasts over leading dims.

    ``W`` is initialised with Xavier-uniform values and ``b`` with zeros.
    Xavier initialisation keeps activation variances stable for linear,
    tanh and sigmoid layers; for deep ReLU networks consider re-initialising
    with :func:`~notorch.nn.init.he_uniform_` instead.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: Whether to include the additive bias ``b``.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        self.in_features: int = int(in_features)
        self.out_features: int = int(out_features)
        self.W: Parameter = Parameter(
            np.empty((self.in_features, self.out_features), dtype=np.float32)
        )
        xavier_uniform_(self.W)
        self.b: Parameter | None = (
            Parameter(np.zeros((self.out_features,), dtype=np.float32)) if bias else None
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the affine transform ``x @ W + b``."""
        out = x @ self.W
        if self.b is not None:
            out = out + self.b
        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.b is not None}"
        )


class Flatten(Module):
    """Flatten the input from ``start_dim`` to the end into a single axis.

    Args:
        start_dim: First dim to flatten (may be negative to count from the
            end). With the default ``start_dim=1``, an input of shape
            ``(N, *)`` becomes ``(N, -1)``.
    """

    def __init__(self, start_dim: int = 1) -> None:
        super().__init__()
        self.start_dim: int = int(start_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Flatten all dims from ``start_dim`` onwards, preserving the graph."""
        shape = tuple(x.shape)
        ndim = len(shape)
        dim = self.start_dim + ndim if self.start_dim < 0 else self.start_dim
        if not 0 <= dim <= ndim:
            raise ValueError(f"start_dim={self.start_dim} out of range for {ndim}-d input")
        if dim >= ndim:
            return x
        kept = shape[:dim]
        flat = 1
        for size in shape[dim:]:
            flat *= size
        new_shape = (*kept, int(flat))
        return x.reshape(new_shape)

    def extra_repr(self) -> str:
        return f"start_dim={self.start_dim}"
