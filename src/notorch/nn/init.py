"""In-place weight initialisation routines.

Every function takes a :class:`~notorch.tensor.Tensor`, writes sampled values
into its ``.data`` in place (via :func:`numpy.copyto` or in-place fill, so the
tensor object and its autograd metadata are preserved), and returns the tensor.

Weight tensors are assumed to have shape ``(..., fan_in, fan_out)``, matching
:class:`~notorch.nn.Linear` whose weight ``W`` has shape
``(in_features, out_features)``. ``fan_in``/``fan_out`` are derived from the
tensor shape unless explicitly overridden. Randomness comes from a
:class:`numpy.random.Generator` (a seed integer is also accepted for
convenience); a fresh entropy-seeded generator is used when ``rng`` is
``None``.
"""

from __future__ import annotations

import math

import numpy as np

from ..tensor import Tensor

__all__ = [
    "he_normal_",
    "he_uniform_",
    "kaiming_normal_",
    "kaiming_uniform_",
    "normal_",
    "ones_",
    "uniform_",
    "xavier_normal_",
    "xavier_uniform_",
    "zeros_",
]

RngLike = np.random.Generator | np.random.RandomState | int | None


def _coerce_rng(rng: RngLike) -> np.random.Generator | np.random.RandomState:
    """Return a usable RNG from ``None``, a seed integer, or an RNG instance."""
    if rng is None:
        from ..random import get_rng

        return get_rng()
    if isinstance(rng, (int, np.integer)):
        return np.random.default_rng(int(rng))
    return rng


def _resolve_fan(
    shape: tuple[int, ...],
    fan_in: int | None,
    fan_out: int | None,
) -> tuple[int, int]:
    """Derive ``(fan_in, fan_out)`` from a weight shape.

    Assumes shape ``(..., fan_in, fan_out)``; 1-D tensors use their single dim
    for both fans and 0-D tensors fall back to ``(1, 1)``. Explicit
    ``fan_in``/``fan_out`` arguments override the derived values.
    """
    if len(shape) >= 2:
        derived_in, derived_out = int(shape[-2]), int(shape[-1])
    elif len(shape) == 1:
        derived_in = derived_out = int(shape[0])
    else:
        derived_in = derived_out = 1
    if fan_in is not None:
        derived_in = int(fan_in)
    if fan_out is not None:
        derived_out = int(fan_out)
    return max(derived_in, 1), max(derived_out, 1)


def _fill_in_place(tensor: Tensor, sample: np.ndarray) -> Tensor:
    """Copy ``sample`` into ``tensor.data`` in place and return the tensor."""
    np.copyto(tensor.data, sample.astype(tensor.data.dtype, copy=False))
    return tensor


def xavier_uniform_(
    tensor: Tensor,
    gain: float = 1.0,
    rng: RngLike = None,
    fan_in: int | None = None,
    fan_out: int | None = None,
) -> Tensor:
    """Fill with ``Uniform(-limit, limit)`` where ``limit = gain * sqrt(6 / (fan_in + fan_out))``.

    Args:
        tensor: Tensor whose ``.data`` is modified in place.
        gain: Multiplicative factor (use ``sqrt(2)`` for ReLU-adjacent layers).
        rng: Generator, seed integer, or ``None`` for fresh entropy.
        fan_in: Override for the fan-in derived from the shape.
        fan_out: Override for the fan-out derived from the shape.
    """
    fi, fo = _resolve_fan(tuple(tensor.data.shape), fan_in, fan_out)
    limit = gain * math.sqrt(6.0 / (fi + fo))
    sample = _coerce_rng(rng).uniform(-limit, limit, size=tensor.data.shape)
    return _fill_in_place(tensor, sample)


def xavier_normal_(
    tensor: Tensor,
    gain: float = 1.0,
    rng: RngLike = None,
    fan_in: int | None = None,
    fan_out: int | None = None,
) -> Tensor:
    """Fill with ``Normal(0, std)`` where ``std = gain * sqrt(2 / (fan_in + fan_out))``.

    Args:
        tensor: Tensor whose ``.data`` is modified in place.
        gain: Multiplicative factor.
        rng: Generator, seed integer, or ``None`` for fresh entropy.
        fan_in: Override for the fan-in derived from the shape.
        fan_out: Override for the fan-out derived from the shape.
    """
    fi, fo = _resolve_fan(tuple(tensor.data.shape), fan_in, fan_out)
    std = gain * math.sqrt(2.0 / (fi + fo))
    sample = _coerce_rng(rng).normal(0.0, std, size=tensor.data.shape)
    return _fill_in_place(tensor, sample)


def he_uniform_(
    tensor: Tensor,
    gain: float = math.sqrt(2.0),
    mode: str = "fan_in",
    rng: RngLike = None,
    fan_in: int | None = None,
    fan_out: int | None = None,
) -> Tensor:
    """Fill with ``Uniform(-limit, limit)`` where ``limit = gain * sqrt(3 / fan)``.

    With the default ``gain=sqrt(2)`` this is standard He initialisation for
    ReLU networks (``limit = sqrt(6 / fan)``).

    Args:
        tensor: Tensor whose ``.data`` is modified in place.
        gain: Multiplicative factor (``sqrt(2)`` for ReLU).
        mode: Which fan to use, ``"fan_in"`` (forward pass) or ``"fan_out"``.
        rng: Generator, seed integer, or ``None`` for fresh entropy.
        fan_in: Override for the fan-in derived from the shape.
        fan_out: Override for the fan-out derived from the shape.
    """
    if mode not in ("fan_in", "fan_out"):
        raise ValueError(f"mode must be 'fan_in' or 'fan_out', got {mode!r}")
    fi, fo = _resolve_fan(tuple(tensor.data.shape), fan_in, fan_out)
    fan = fi if mode == "fan_in" else fo
    limit = gain * math.sqrt(3.0 / fan)
    sample = _coerce_rng(rng).uniform(-limit, limit, size=tensor.data.shape)
    return _fill_in_place(tensor, sample)


def he_normal_(
    tensor: Tensor,
    gain: float = math.sqrt(2.0),
    mode: str = "fan_in",
    rng: RngLike = None,
    fan_in: int | None = None,
    fan_out: int | None = None,
) -> Tensor:
    """Fill with ``Normal(0, std)`` where ``std = gain * sqrt(1 / fan)``.

    With the default ``gain=sqrt(2)`` this is standard He initialisation for
    ReLU networks (``std = sqrt(2 / fan)``).

    Args:
        tensor: Tensor whose ``.data`` is modified in place.
        gain: Multiplicative factor (``sqrt(2)`` for ReLU).
        mode: Which fan to use, ``"fan_in"`` (forward pass) or ``"fan_out"``.
        rng: Generator, seed integer, or ``None`` for fresh entropy.
        fan_in: Override for the fan-in derived from the shape.
        fan_out: Override for the fan-out derived from the shape.
    """
    if mode not in ("fan_in", "fan_out"):
        raise ValueError(f"mode must be 'fan_in' or 'fan_out', got {mode!r}")
    fi, fo = _resolve_fan(tuple(tensor.data.shape), fan_in, fan_out)
    fan = fi if mode == "fan_in" else fo
    std = gain * math.sqrt(1.0 / fan)
    sample = _coerce_rng(rng).normal(0.0, std, size=tensor.data.shape)
    return _fill_in_place(tensor, sample)


kaiming_uniform_ = he_uniform_
kaiming_normal_ = he_normal_


def zeros_(tensor: Tensor) -> Tensor:
    """Fill ``tensor.data`` with zeros in place and return the tensor."""
    tensor.data.fill(0)
    return tensor


def ones_(tensor: Tensor) -> Tensor:
    """Fill ``tensor.data`` with ones in place and return the tensor."""
    tensor.data.fill(1)
    return tensor


def normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0, rng: RngLike = None) -> Tensor:
    """Fill ``tensor.data`` with samples from ``Normal(mean, std)`` in place.

    Args:
        tensor: Tensor whose ``.data`` is modified in place.
        mean: Mean of the normal distribution.
        std: Standard deviation of the normal distribution (must be >= 0).
        rng: Generator, seed integer, or ``None`` for fresh entropy.
    """
    if std < 0:
        raise ValueError(f"std must be non-negative, got {std}")
    sample = _coerce_rng(rng).normal(mean, std, size=tensor.data.shape)
    return _fill_in_place(tensor, sample)


def uniform_(tensor: Tensor, a: float = 0.0, b: float = 1.0, rng: RngLike = None) -> Tensor:
    """Fill ``tensor.data`` with samples from ``Uniform(a, b)`` in place.

    Args:
        tensor: Tensor whose ``.data`` is modified in place.
        a: Lower bound of the uniform distribution.
        b: Upper bound of the uniform distribution (must be >= ``a``).
        rng: Generator, seed integer, or ``None`` for fresh entropy.
    """
    if b < a:
        raise ValueError(f"b must be >= a, got a={a}, b={b}")
    sample = _coerce_rng(rng).uniform(a, b, size=tensor.data.shape)
    return _fill_in_place(tensor, sample)
