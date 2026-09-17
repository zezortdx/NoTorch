"""Loss modules: :class:`MSELoss`, :class:`BinaryCrossEntropyLoss`, :class:`CrossEntropyLoss`."""

from __future__ import annotations

from typing import Union

import numpy as np

from ..tensor import Tensor
from .module import Module

__all__ = [
    "BCELoss",
    "BinaryCrossEntropyLoss",
    "CrossEntropyLoss",
    "MSELoss",
]

Target = Union[Tensor, np.ndarray, list, tuple]

_REDUCTIONS = ("mean", "sum", "none")


def _as_target_tensor(target: Target) -> Tensor:
    """Return ``target`` as a non-trainable :class:`Tensor`."""
    if isinstance(target, Tensor):
        return target
    return Tensor(np.asarray(target), requires_grad=False)


def _apply_reduction(value: Tensor, reduction: str) -> Tensor:
    """Apply a ``mean``/``sum``/``none`` reduction, preserving the graph."""
    if reduction == "mean":
        return value.mean()
    if reduction == "sum":
        return value.sum()
    return value


def _check_reduction(reduction: str) -> str:
    """Validate a reduction mode and return it."""
    if reduction not in _REDUCTIONS:
        raise ValueError(f"reduction must be one of {_REDUCTIONS}, got {reduction!r}")
    return reduction


def _log_softmax_rows(logits: Tensor) -> Tensor:
    """Numerically stable log-softmax over the class axis of 2-D logits."""
    return logits.log_softmax(dim=1)


class MSELoss(Module):
    """Mean-squared-error loss, ``(input - target) ** 2`` with a reduction.

    Args:
        reduction: ``"mean"`` (default), ``"sum"`` or ``"none"``. ``"none"``
            returns the elementwise squared errors with the graph intact.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction: str = _check_reduction(reduction)

    def forward(self, input: Tensor, target: Target) -> Tensor:
        """Compute the (reduced) squared error between ``input`` and ``target``."""
        diff = input - _as_target_tensor(target)
        return _apply_reduction(diff * diff, self.reduction)

    def extra_repr(self) -> str:
        return f"reduction={self.reduction!r}"


class BinaryCrossEntropyLoss(Module):
    """Binary cross-entropy over probabilities in ``[0, 1]``.

    Probabilities are clamped to ``[eps, 1 - eps]`` (default ``eps=1e-12``)
    with differentiable ``relu`` operations so the autograd graph is preserved
    and ``log(0)`` is avoided.

    Args:
        reduction: ``"mean"`` (default), ``"sum"`` or ``"none"``.
        eps: Clamping epsilon; probabilities are confined to
            ``[eps, 1 - eps]``.
        from_logits: If ``True``, inputs are treated as raw logits and passed
            through ``sigmoid`` first; otherwise inputs are probabilities.
    """

    def __init__(
        self,
        reduction: str = "mean",
        eps: float = 1e-12,
        from_logits: bool = False,
    ) -> None:
        super().__init__()
        self.reduction: str = _check_reduction(reduction)
        if not 0.0 < eps < 0.5:
            raise ValueError(f"eps must lie in (0, 0.5), got {eps}")
        self.eps: float = float(eps)
        self.from_logits: bool = bool(from_logits)

    def forward(self, input: Tensor, target: Target) -> Tensor:
        """Compute the (reduced) binary cross-entropy."""
        t = _as_target_tensor(target)
        probs = input.sigmoid() if self.from_logits else input
        lo, hi = self.eps, 1.0 - self.eps
        clamped = hi - (hi - ((probs - lo).relu() + lo)).relu()
        loss = -((t * clamped.log()) + ((1 - t) * (1 - clamped).log()))
        return _apply_reduction(loss, self.reduction)

    def extra_repr(self) -> str:
        return f"reduction={self.reduction!r}, eps={self.eps!r}, from_logits={self.from_logits!r}"


BCELoss = BinaryCrossEntropyLoss


class CrossEntropyLoss(Module):
    """Softmax cross-entropy for integer class targets.

    Applies a numerically stable ``log_softmax`` to ``logits`` of shape
    ``(N, C)`` and takes the negative log-likelihood of the true classes
    given as integer targets of shape ``(N,)`` (a :class:`Tensor` or an
    array-like). Gathering uses a one-hot mask with ``mul``/``sum`` so the
    graph back to ``logits`` is preserved.

    Args:
        reduction: ``"mean"`` (default), ``"sum"`` or ``"none"``. ``"none"``
            returns per-sample losses of shape ``(N,)``.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction: str = _check_reduction(reduction)

    def forward(self, input: Tensor, target: Target) -> Tensor:
        """Compute the (reduced) cross-entropy between ``logits`` and class indices."""
        if tuple(input.data.shape) == () or input.data.ndim != 2:
            raise ValueError(
                f"expected logits of shape (N, C), got shape {tuple(input.data.shape)}"
            )
        num_samples, num_classes = (int(d) for d in input.data.shape)
        raw = target.data if isinstance(target, Tensor) else np.asarray(target)
        indices = np.asarray(raw).astype(np.int64).reshape(-1)
        if indices.size != num_samples:
            raise ValueError(f"expected {num_samples} targets, got {indices.size}")
        if bool(((indices < 0) | (indices >= num_classes)).any()):
            raise ValueError(f"target indices must lie in [0, {num_classes})")
        log_probs = _log_softmax_rows(input)
        one_hot = np.zeros((num_samples, num_classes), dtype=input.data.dtype)
        one_hot[np.arange(num_samples), indices] = 1.0
        mask = Tensor(one_hot, requires_grad=False)
        gathered = (log_probs * mask).sum(axis=1)
        return _apply_reduction(-gathered, self.reduction)

    def extra_repr(self) -> str:
        return f"reduction={self.reduction!r}"
