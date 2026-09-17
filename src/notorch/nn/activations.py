"""Elementwise activation modules."""

from __future__ import annotations

from ..tensor import Tensor
from .module import Module


class ReLU(Module):
    """Applies the rectified linear unit, ``relu(x) = max(0, x)``, elementwise."""

    def forward(self, x: Tensor) -> Tensor:
        """Apply ReLU elementwise, preserving the autograd graph."""
        return x.relu()


class Sigmoid(Module):
    """Applies the logistic sigmoid, ``sigmoid(x) = 1 / (1 + exp(-x))``, elementwise."""

    def forward(self, x: Tensor) -> Tensor:
        """Apply sigmoid elementwise, preserving the autograd graph."""
        return x.sigmoid()


class Tanh(Module):
    """Applies the hyperbolic tangent, ``tanh(x)``, elementwise."""

    def forward(self, x: Tensor) -> Tensor:
        """Apply tanh elementwise, preserving the autograd graph."""
        return x.tanh()
