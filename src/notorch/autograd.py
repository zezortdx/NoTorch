"""Broadcast-aware gradient reduction for the NoTorch autograd engine.

When a forward op broadcasts an input (e.g. ``a(32, 10) + b(10,)``), the
upstream gradient has the *broadcast* shape and must be summed back to the
input's original shape before accumulation.
"""

from __future__ import annotations

import numpy as np


def sum_to_shape(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum a broadcasted gradient back to ``shape``.

    Args:
        grad: Upstream gradient with the broadcast (output) shape.
        shape: Original (pre-broadcast) input shape. ``()`` denotes scalar.

    Returns:
        Gradient reduced to exactly ``shape``.

    Handles:
        - Same shape (no-op, returns ``grad`` unchanged).
        - Scalar target ``()`` (sums everything to a 0-d array).
        - Target ``(1,)`` (sums everything, reshapes to ``(1,)``).
        - Leading-dim broadcasting (extra leading dims in ``grad`` are
          summed away, e.g. ``(32, 10) -> (10,)``).
        - Size-1 dims (axes where the target has 1 but ``grad`` has N
          are summed with ``keepdims=True``, e.g. ``(4, 5) -> (4, 1)``).

    Vectorized: no Python loops over elements (one small loop over axes
    only, each step a single ``numpy.sum`` call).
    """
    shape = tuple(shape)
    grad = np.asarray(grad)

    # No-op fast path.
    if grad.shape == shape:
        return grad

    # Scalar target: collapse everything to a 0-d array.
    if shape == ():
        return np.array(grad.sum())

    # 1) Collapse extra leading dims introduced by broadcasting.
    #    e.g. grad (32, 10) -> target (10,): sum axis 0.
    ndiff = grad.ndim - len(shape)
    if ndiff > 0:
        grad = grad.sum(axis=tuple(range(ndiff)))
        if grad.shape == shape:
            return grad

    # 2) Collapse axes where the target has size 1 (broadcast dims).
    #    e.g. grad (4, 5) -> target (4, 1): sum axis 1 with keepdims.
    for i, (g_dim, s_dim) in enumerate(zip(grad.shape, shape)):
        if s_dim == 1 and g_dim != 1:
            grad = grad.sum(axis=i, keepdims=True)

    # 3) Reshape handles leftovers such as target (1,) vs grad shape (1,)
    #    already fine, or scalar-kept dims. Guarantees exact shape.
    if grad.shape != shape:
        grad = grad.reshape(shape)
    return grad
