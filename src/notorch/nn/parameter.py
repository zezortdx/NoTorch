"""Trainable parameter type.

:class:`Parameter` is a :class:`~notorch.tensor.Tensor` with
``requires_grad=True`` by default, marking it as a leaf tensor that
optimizers update and that :class:`~notorch.nn.Module` collects in
:meth:`~notorch.nn.Module.parameters`.
"""

from __future__ import annotations

from typing import Any

from ..tensor import Tensor


class Parameter(Tensor):
    """A trainable tensor; a :class:`Tensor` defaulting to ``requires_grad=True``.

    Args:
        data: Array-like data used to initialise the parameter.
        requires_grad: Whether autograd tracks operations on this parameter.
        **kwargs: Forwarded to :class:`~notorch.tensor.Tensor`.
    """

    def __init__(self, data: Any, requires_grad: bool = True, **kwargs: Any) -> None:
        super().__init__(data, requires_grad=requires_grad, **kwargs)

    def __repr__(self) -> str:
        return f"Parameter({self.data!r}, requires_grad={self.requires_grad!r})"
