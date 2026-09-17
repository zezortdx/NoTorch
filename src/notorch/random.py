"""Seedable global RNG for NoTorch (numpy Generator only)."""

from __future__ import annotations

import numpy as np

_rng: np.random.Generator | None = None


def manual_seed(seed: int) -> np.random.Generator:
    """Seed the global RNG and return it.

    Args:
        seed: Integer seed for ``np.random.default_rng``.

    Returns:
        The newly created global ``Generator``.
    """
    global _rng
    _rng = np.random.default_rng(seed)
    return _rng


def get_rng() -> np.random.Generator:
    """Return the global RNG, creating an unseeded one if needed.

    Returns:
        The global ``np.random.Generator``.
    """
    global _rng
    if _rng is None:
        _rng = np.random.default_rng()
    return _rng


def normal(
    loc: float = 0.0,
    scale: float = 1.0,
    size=None,
    dtype=np.float64,
) -> np.ndarray:
    """Draw samples from a normal distribution using the global RNG.

    Args:
        loc: Mean of the distribution.
        scale: Standard deviation of the distribution.
        size: Output shape (``None`` for a scalar).
        dtype: ``np.float32`` or ``np.float64``.

    Returns:
        Array of normal samples.
    """
    return get_rng().normal(loc=loc, scale=scale, size=size).astype(dtype, copy=False)


def uniform(
    low: float = 0.0,
    high: float = 1.0,
    size=None,
    dtype=np.float64,
) -> np.ndarray:
    """Draw samples from a uniform distribution using the global RNG.

    Args:
        low: Lower bound (inclusive).
        high: Upper bound (exclusive).
        size: Output shape (``None`` for a scalar).
        dtype: ``np.float32`` or ``np.float64``.

    Returns:
        Array of uniform samples.
    """
    return get_rng().uniform(low=low, high=high, size=size).astype(dtype, copy=False)


def randn(*shape, dtype=np.float64) -> np.ndarray:
    """Standard-normal samples of shape ``shape`` (convenience wrapper)."""
    size = shape if shape else None
    return normal(0.0, 1.0, size=size, dtype=dtype)


def rand(*shape, dtype=np.float64) -> np.ndarray:
    """Uniform ``[0, 1)`` samples of shape ``shape`` (convenience wrapper)."""
    size = shape if shape else None
    return uniform(0.0, 1.0, size=size, dtype=dtype)
