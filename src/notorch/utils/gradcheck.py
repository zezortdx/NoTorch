"""Gradient checker for the NoTorch autograd engine (numpy only)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def gradcheck(
    f: Callable[[list], object],
    inputs: list,
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-3,
) -> tuple[bool, dict]:
    """Compare autograd gradients against central finite differences.

    Args:
        f: Callable taking a list of Tensors and returning a scalar Tensor.
        inputs: Tensors with ``requires_grad=True`` (cast to float64 copies).
        eps: Finite-difference perturbation size.
        atol: Absolute tolerance for ``numpy.allclose``.
        rtol: Relative tolerance for ``numpy.allclose``.

    Returns:
        ``(passed, details)`` where ``passed`` is True if every input's
        analytic gradient matches the numeric one, and ``details`` holds
        per-input ``analytic``/``numeric`` arrays, ``max_abs_err``,
        ``max_rel_err`` and up to 10 ``mismatches`` entries.
    """
    from ..tensor import Tensor

    # Fresh float64 copies requiring grad for the analytic pass.
    ts = [Tensor(np.asarray(t.data, dtype=np.float64), requires_grad=True) for t in inputs]
    out = f(ts)
    if not isinstance(out, Tensor) or out.data.size != 1:
        raise ValueError("gradcheck: f must return a scalar (single-element) Tensor")
    out.backward()
    analytics = [t.grad.copy() for t in ts]

    passed = True
    per_input = []
    for k, t in enumerate(inputs):
        x0 = np.asarray(t.data, dtype=np.float64)
        analytic = analytics[k]
        numeric = np.zeros_like(x0)
        # Central differences: (f(x+e) - f(x-e)) / 2e per element.
        for idx in np.ndindex(x0.shape):
            xp = x0.copy()
            xm = x0.copy()
            xp[idx] += eps
            xm[idx] -= eps
            fp = f(
                [
                    Tensor(xp if i == k else np.asarray(v.data, dtype=np.float64))
                    for i, v in enumerate(inputs)
                ]
            ).data.item()
            fm = f(
                [
                    Tensor(xm if i == k else np.asarray(v.data, dtype=np.float64))
                    for i, v in enumerate(inputs)
                ]
            ).data.item()
            numeric[idx] = (fp - fm) / (2.0 * eps)
        ok = np.allclose(analytic, numeric, atol=atol, rtol=rtol)
        passed = passed and bool(ok)
        abs_err = np.abs(analytic - numeric)
        denom = np.maximum(np.abs(numeric), 1e-12)
        rel_err = abs_err / denom
        bad = np.argwhere(~np.isclose(analytic, numeric, atol=atol, rtol=rtol))
        mismatches = [
            {
                "index": tuple(i.tolist()),
                "analytic": float(analytic[tuple(i.tolist())]),
                "numeric": float(numeric[tuple(i.tolist())]),
            }
            for i in bad[:10]
        ]
        per_input.append(
            {
                "input": k,
                "passed": bool(ok),
                "analytic": analytic,
                "numeric": numeric,
                "max_abs_err": float(abs_err.max(initial=0.0)),
                "max_rel_err": float(rel_err.max(initial=0.0)),
                "mismatches": mismatches,
            }
        )
    return passed, {"inputs": per_input}


def check_op(op: Callable, inputs: list, **kwargs) -> tuple[bool, dict]:
    """Check an op by summing its output to a scalar first.

    Args:
        op: Callable taking ``*inputs`` (Tensors) and returning a Tensor.
        inputs: Tensors with ``requires_grad=True``.
        **kwargs: Forwarded to :func:`gradcheck` (``eps``/``atol``/``rtol``).

    Returns:
        Same ``(passed, details)`` tuple as :func:`gradcheck`.
    """
    return gradcheck(lambda ts: op(*ts).sum(), inputs, **kwargs)
