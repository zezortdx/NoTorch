"""Scalar/vector reverse-mode autograd Tensor for NoTorch (numpy only)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .autograd import sum_to_shape

_LOG_EPS = 1e-12


def _as_array(data, dtype=None) -> np.ndarray:
    """Convert user input to a float32/float64 ndarray."""
    if isinstance(data, Tensor):
        arr = data.data.copy()
    else:
        arr = np.asarray(data)
    if dtype is None:
        if arr.dtype == np.float64 or arr.dtype == np.float32:
            pass
        elif np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float64)
        else:
            # ints, bools, lists -> float64
            arr = arr.astype(np.float64)
    else:
        dt = np.dtype(dtype)
        if dt not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError(f"Tensor dtype must be float32/float64, got {dt}")
        arr = arr.astype(dt, copy=False)
    return arr


def _ensure_tensor(other, ref: Tensor | None = None) -> Tensor:
    """Wrap a scalar/array as a non-differentiable Tensor."""
    if isinstance(other, Tensor):
        return other
    dtype = ref.data.dtype if ref is not None else None
    return Tensor(other, requires_grad=False, dtype=dtype)


class Tensor:
    """Numpy-backed value in a dynamically built autograd graph.

    Attributes:
        data: Underlying ``np.ndarray`` (float32 or float64).
        requires_grad: Whether to track ops for :meth:`backward`.
        grad: Accumulated gradient (``None`` until :meth:`backward` runs).
    """

    def __init__(
        self,
        data,
        requires_grad: bool = False,
        dtype=None,
        name: str | None = None,
        label: str | None = None,
    ) -> None:
        """Create a Tensor.

        Args:
            data: Scalar, list, or ``np.ndarray``. Ints/lists become float64;
                float32 arrays stay float32 unless ``dtype`` overrides.
            requires_grad: Track operations for differentiation.
            dtype: Optional ``np.float32``/``np.float64`` override.
            name: Optional label for debugging/plots. ``label`` is an alias.
        """
        self.data: np.ndarray = _as_array(data, dtype)
        self.requires_grad: bool = bool(requires_grad)
        self.grad: np.ndarray | None = None
        self._prev: set[Tensor] = set()
        self._op: str = ""
        self._backward: Callable[[], None] = lambda: None
        self.name: str | None = name if name is not None else label

    # -- properties -----------------------------------------------------
    @property
    def shape(self) -> tuple[int, ...]:
        """Array shape of :attr:`data`."""
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        """Dtype of :attr:`data`."""
        return self.data.dtype

    @property
    def ndim(self) -> int:
        """Number of dimensions of :attr:`data`."""
        return self.data.ndim

    @property
    def label(self) -> str | None:
        """Alias of :attr:`name`."""
        return self.name

    @label.setter
    def label(self, value: str | None) -> None:
        self.name = value

    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, shape={self.shape}, requires_grad={self.requires_grad})"

    # -- helpers --------------------------------------------------------
    def zeros_like(self) -> Tensor:
        """Return a zero Tensor with the same shape/dtype (no grad)."""
        return Tensor(np.zeros_like(self.data))

    def ones_like(self) -> Tensor:
        """Return a ones Tensor with the same shape/dtype (no grad)."""
        return Tensor(np.ones_like(self.data))

    @classmethod
    def zeros(cls, shape: Sequence[int] | int, **kwargs) -> Tensor:
        """Create a zero Tensor of ``shape``."""
        return cls(np.zeros(shape), **kwargs)

    @classmethod
    def ones(cls, shape: Sequence[int] | int, **kwargs) -> Tensor:
        """Create a ones Tensor of ``shape``."""
        return cls(np.ones(shape), **kwargs)

    def zero_grad(self) -> None:
        """Reset :attr:`grad` to ``None``."""
        self.grad = None

    def detach(self) -> Tensor:
        """Return a copy that shares no graph (``requires_grad=False``)."""
        return Tensor(self.data.copy(), requires_grad=False)

    def numpy(self) -> np.ndarray:
        """Return a copy of :attr:`data` as ``np.ndarray``."""
        return self.data.copy()

    def item(self) -> float:
        """Return the scalar value (only for single-element Tensors)."""
        return float(self.data.item())

    def tolist(self):
        """Return :attr:`data` as a (nested) Python list."""
        return self.data.tolist()

    # -- graph mechanics ------------------------------------------------
    def _accumulate(self, t: Tensor, g: np.ndarray) -> None:
        """Add upstream grad ``g`` into leaf/branch tensor ``t``."""
        if not t.requires_grad:
            return
        if g.shape != t.shape:
            g = sum_to_shape(g, t.shape)
        g = np.array(g, dtype=t.data.dtype, copy=True)
        t.grad = g if t.grad is None else t.grad + g

    @staticmethod
    def _wrap(
        out_data: np.ndarray,
        parents: tuple[Tensor, ...],
        op: str,
        backward: Callable[[], None],
    ) -> Tensor:
        """Build the output Tensor node of an op."""
        out = Tensor(
            out_data,
            requires_grad=any(p.requires_grad for p in parents),
        )
        out._prev = set(parents)
        out._op = op
        out._backward = backward
        return out

    def backward(self, gradient=None) -> None:
        """Run reverse-mode autograd from this Tensor.

        Args:
            gradient: Explicit upstream gradient for non-scalar outputs.
                Must be broadcastable to :attr:`shape`. If ``None``, this
                Tensor must be scalar (single element) and the seed is ones.

        Raises:
            RuntimeError: If called on a Tensor with ``requires_grad=False``,
                or on a non-scalar without a ``gradient`` argument.
            ValueError: If ``gradient`` cannot broadcast to :attr:`shape`.
        """
        if not self.requires_grad:
            raise RuntimeError("backward() called on a Tensor with requires_grad=False")
        if gradient is None:
            if self.data.size != 1:
                raise RuntimeError("backward() on non-scalar Tensor requires a 'gradient' argument")
            seed = np.ones_like(self.data)
        else:
            g = np.asarray(gradient, dtype=self.data.dtype)
            try:
                seed = np.broadcast_to(g, self.shape).copy()
            except ValueError:
                raise ValueError(
                    f"gradient shape {g.shape} cannot broadcast to Tensor shape {self.shape}"
                )
        self.grad = seed if self.grad is None else self.grad + seed

        topo: list[Tensor] = []
        visited: set[int] = set()

        def _dfs(v: Tensor) -> None:
            if id(v) in visited:
                return
            visited.add(id(v))
            for p in v._prev:
                _dfs(p)
            topo.append(v)

        _dfs(self)
        for v in reversed(topo):
            if v.grad is None:
                continue
            v._backward()

    # -- elementwise binary ops (broadcasting via sum_to_shape) ---------
    def __add__(self, other) -> Tensor:
        """Elementwise add with broadcasting. d(a+b)/da = 1, d/db = 1."""
        other = _ensure_tensor(other, self)
        out_data = self.data + other.data
        out = Tensor._wrap(out_data, (self, other), "add", lambda: None)

        def _backward() -> None:
            g = out.grad
            out._accumulate(self, g)
            out._accumulate(other, g)

        out._backward = _backward
        return out

    __radd__ = __add__

    def __sub__(self, other) -> Tensor:
        """Elementwise subtract. d(a-b)/da = 1, d/db = -1."""
        other = _ensure_tensor(other, self)
        out_data = self.data - other.data
        out = Tensor._wrap(out_data, (self, other), "sub", lambda: None)

        def _backward() -> None:
            g = out.grad
            out._accumulate(self, g)
            out._accumulate(other, -g)

        out._backward = _backward
        return out

    def __rsub__(self, other) -> Tensor:
        """Reflected subtract: ``other - self``."""
        return _ensure_tensor(other, self).__sub__(self)

    def __mul__(self, other) -> Tensor:
        """Elementwise multiply. d(a*b)/da = b, d/db = a."""
        other = _ensure_tensor(other, self)
        out_data = self.data * other.data
        out = Tensor._wrap(out_data, (self, other), "mul", lambda: None)

        def _backward() -> None:
            g = out.grad
            out._accumulate(self, g * other.data)
            out._accumulate(other, g * self.data)

        out._backward = _backward
        return out

    __rmul__ = __mul__

    def __truediv__(self, other) -> Tensor:
        """Elementwise divide. d(a/b)/da = 1/b, d/db = -a/b^2."""
        other = _ensure_tensor(other, self)
        out_data = self.data / other.data
        out = Tensor._wrap(out_data, (self, other), "div", lambda: None)

        def _backward() -> None:
            g = out.grad
            b = other.data
            out._accumulate(self, g / b)
            out._accumulate(other, -g * self.data / (b * b))

        out._backward = _backward
        return out

    def __rtruediv__(self, other) -> Tensor:
        """Reflected divide: ``other / self``."""
        return _ensure_tensor(other, self).__truediv__(self)

    def __neg__(self) -> Tensor:
        """Negate. d(-x)/dx = -1."""
        out_data = -self.data
        out = Tensor._wrap(out_data, (self,), "neg", lambda: None)

        def _backward() -> None:
            out._accumulate(self, -out.grad)

        out._backward = _backward
        return out

    def __pow__(self, exponent) -> Tensor:
        """Power with scalar exponent only. d(x^p)/dx = p*x^(p-1)."""
        if isinstance(exponent, Tensor) or not np.isscalar(exponent):
            raise TypeError("pow exponent must be a Python scalar (int/float)")
        p = exponent
        out_data = self.data**p
        out = Tensor._wrap(out_data, (self,), f"pow({p})", lambda: None)

        def _backward() -> None:
            out._accumulate(self, out.grad * p * self.data ** (p - 1))

        out._backward = _backward
        return out

    def __matmul__(self, other) -> Tensor:
        """Matrix multiply (``@``), supports batched N-D via numpy."""
        other = _ensure_tensor(other, self)
        a, b = self.data, other.data
        out_data = a @ b
        out = Tensor._wrap(out_data, (self, other), "matmul", lambda: None)

        def _backward() -> None:
            g = out.grad
            if a.ndim == 1 and b.ndim == 1:
                # dot -> scalar: d/da = g*b, d/db = g*a
                ga, gb = g * b, g * a
            elif a.ndim == 1:
                # (K,) @ (..., K, N) -> (..., N)
                # dL/da_k = sum_batch,j B_kj * g_j ; dL/dB_kj = a_k * g_j
                ga = (b @ g[..., None])[..., 0]
                gb = a[..., :, None] * g[..., None, :]
            elif b.ndim == 1:
                # (..., M, K) @ (K,) -> (..., M)
                # dL/dA_mk = g_m * b_k ; dL/db_k = sum A_mk * g_m
                ga = g[..., :, None] * b
                gb = (a.swapaxes(-2, -1) @ g[..., None])[..., 0]
            else:
                # (..., M, K) @ (..., K, N): standard adjoint with
                # axes swapped over the last two dims; batch broadcast
                # dims collapse later via sum_to_shape.
                ga = g @ b.swapaxes(-1, -2)
                gb = a.swapaxes(-1, -2) @ g
            out._accumulate(self, ga)
            out._accumulate(other, gb)

        out._backward = _backward
        return out

    # -- reductions -----------------------------------------------------
    def _reduced_grad(self, out: Tensor, axis, keepdims: bool) -> np.ndarray:
        """Broadcast a reduction's upstream grad back to input shape."""
        g = out.grad
        if axis is None:
            return np.broadcast_to(g, self.shape).copy()
        axes = (axis,) if isinstance(axis, int) else tuple(axis)
        axes = tuple(ax % self.ndim for ax in axes)
        if not keepdims:
            # Re-insert the dropped axes as size-1 for broadcasting.
            shape = list(g.shape)
            for ax in sorted(axes):
                shape.insert(ax, 1)
            g = g.reshape(shape)
        return np.broadcast_to(g, self.shape).copy()

    def sum(self, axis=None, keepdims: bool = False) -> Tensor:
        """Sum elements over ``axis``. Backward broadcasts the grad."""
        out_data = self.data.sum(axis=axis, keepdims=keepdims)
        out = Tensor._wrap(out_data, (self,), "sum", lambda: None)

        def _backward() -> None:
            out._accumulate(self, self._reduced_grad(out, axis, keepdims))

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims: bool = False) -> Tensor:
        """Mean over ``axis``. Backward is ``sum``-backward divided by N."""
        out_data = self.data.mean(axis=axis, keepdims=keepdims)
        out = Tensor._wrap(out_data, (self,), "mean", lambda: None)

        def _backward() -> None:
            g = self._reduced_grad(out, axis, keepdims)
            if axis is None:
                n = self.data.size
            else:
                axes = (axis,) if isinstance(axis, int) else tuple(axis)
                n = 1
                for ax in axes:
                    n *= self.shape[ax % self.ndim]
            out._accumulate(self, g / n)

        out._backward = _backward
        return out

    def max(self, axis=None, keepdims: bool = False) -> Tensor:
        """Max over ``axis``; ties split the gradient evenly.

        Forward is ``numpy.max``. Backward routes upstream grad to the
        argmax entries; tied maxima share it (mask / tie-count), which
        keeps ``max`` usable for softmax numerical stability.
        """
        out_data = self.data.max(axis=axis, keepdims=keepdims)
        out = Tensor._wrap(out_data, (self,), "max", lambda: None)

        def _backward() -> None:
            x = self.data
            m = x.max(axis=axis, keepdims=True)  # keepdims for broadcasting
            mask = (x == m).astype(x.dtype)
            if axis is None:
                w = mask / mask.sum()
                g = np.broadcast_to(out.grad, x.shape).copy()
            else:
                count = mask.sum(axis=axis, keepdims=True)
                w = mask / np.maximum(count, 1)
                g = out.grad
                if not keepdims:
                    axes = (axis,) if isinstance(axis, int) else tuple(axis)
                    axes = tuple(ax % x.ndim for ax in axes)
                    shape = list(g.shape)
                    for ax in sorted(axes):
                        shape.insert(ax, 1)
                    g = g.reshape(shape)
                g = np.broadcast_to(g, x.shape).copy()
            out._accumulate(self, g * w)

        out._backward = _backward
        return out

    # -- shape ops ------------------------------------------------------
    def reshape(self, *shape) -> Tensor:
        """Reshape (accepts ``reshape((2, 3))`` or ``reshape(2, 3)``)."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        else:
            shape = tuple(shape)
        out_data = self.data.reshape(shape)
        in_shape = self.shape
        out = Tensor._wrap(out_data, (self,), "reshape", lambda: None)

        def _backward() -> None:
            out._accumulate(self, out.grad.reshape(in_shape))

        out._backward = _backward
        return out

    def transpose(self, *axes) -> Tensor:
        """Permute axes (no args reverses them, like ``ndarray.T``)."""
        if len(axes) == 0:
            perm = None
            out_data = self.data.T
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            perm = tuple(axes[0])
            out_data = self.data.transpose(perm)
        else:
            perm = tuple(axes)
            out_data = self.data.transpose(perm)
        out = Tensor._wrap(out_data, (self,), "transpose", lambda: None)

        def _backward() -> None:
            if perm is None:
                out._accumulate(self, out.grad.T)
            else:
                inv = tuple(np.argsort(perm))
                out._accumulate(self, out.grad.transpose(inv))

        out._backward = _backward
        return out

    @property
    def T(self) -> Tensor:
        """Transposed view (axes reversed), differentiable."""
        return self.transpose()

    def flatten(self) -> Tensor:
        """Flatten to 1-D (differentiable reshape to ``(-1,)``)."""
        return self.reshape(-1)

    def __getitem__(self, idx) -> Tensor:
        """Slicing/indexing; backward scatters grad into zeros."""
        out_data = self.data[idx]
        in_shape = self.shape
        out = Tensor._wrap(out_data, (self,), "getitem", lambda: None)

        def _backward() -> None:
            g = np.zeros(in_shape, dtype=self.data.dtype)
            # add.at accumulates correctly for repeated (fancy) indices.
            np.add.at(g, idx, out.grad)
            out._accumulate(self, g)

        out._backward = _backward
        return out

    # -- elementwise nonlinearities -------------------------------------
    def exp(self) -> Tensor:
        """Elementwise exp. d(exp x)/dx = exp x."""
        out_data = np.exp(self.data)
        out = Tensor._wrap(out_data, (self,), "exp", lambda: None)

        def _backward() -> None:
            out._accumulate(self, out.grad * out_data)

        out._backward = _backward
        return out

    def log(self) -> Tensor:
        """Elementwise log with input clamped to ``>= 1e-12``."""
        safe = np.maximum(self.data, _LOG_EPS)
        out_data = np.log(safe)
        out = Tensor._wrap(out_data, (self,), "log", lambda: None)

        def _backward() -> None:
            out._accumulate(self, out.grad / safe)

        out._backward = _backward
        return out

    def tanh(self) -> Tensor:
        """Elementwise tanh. d(tanh x)/dx = 1 - tanh^2 x."""
        out_data = np.tanh(self.data)
        out = Tensor._wrap(out_data, (self,), "tanh", lambda: None)

        def _backward() -> None:
            out._accumulate(self, out.grad * (1.0 - out_data * out_data))

        out._backward = _backward
        return out

    def relu(self) -> Tensor:
        """Elementwise ReLU. Subgradient 1 for x > 0, else 0."""
        out_data = np.maximum(self.data, 0.0)
        out = Tensor._wrap(out_data, (self,), "relu", lambda: None)

        def _backward() -> None:
            out._accumulate(self, out.grad * (self.data > 0).astype(self.data.dtype))

        out._backward = _backward
        return out

    def sigmoid(self) -> Tensor:
        """Elementwise sigmoid. d(s)/dx = s*(1-s)."""
        out_data = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor._wrap(out_data, (self,), "sigmoid", lambda: None)

        def _backward() -> None:
            out._accumulate(self, out.grad * out_data * (1.0 - out_data))

        out._backward = _backward
        return out

    # -- softmax family ---------------------------------------------------
    @staticmethod
    def _norm_dim(dim: int, ndim: int) -> int:
        """Normalize possibly-negative ``dim``."""
        d = dim % ndim
        if not 0 <= d < ndim:
            raise ValueError(f"dim {dim} out of range for {ndim}-D Tensor")
        return d

    def softmax(self, dim: int = -1) -> Tensor:
        """Numerically stable softmax along ``dim`` (max-shift trick)."""
        d = self._norm_dim(dim, self.ndim)
        m = self.data.max(axis=d, keepdims=True)
        e = np.exp(self.data - m)
        s = e / e.sum(axis=d, keepdims=True)
        out = Tensor._wrap(s, (self,), "softmax", lambda: None)

        def _backward() -> None:
            g = out.grad
            # Jacobian-vector product: s * (g - sum(g*s)).
            gs = (g * s).sum(axis=d, keepdims=True)
            out._accumulate(self, s * (g - gs))

        out._backward = _backward
        return out

    def log_softmax(self, dim: int = -1) -> Tensor:
        """Numerically stable log-softmax along ``dim``."""
        d = self._norm_dim(dim, self.ndim)
        m = self.data.max(axis=d, keepdims=True)
        shifted = self.data - m
        lse = np.log(np.exp(shifted).sum(axis=d, keepdims=True))
        out_data = shifted - lse
        out = Tensor._wrap(out_data, (self,), "log_softmax", lambda: None)

        def _backward() -> None:
            g = out.grad
            # dL/dx = g - exp(out) * sum(g).
            out._accumulate(self, g - np.exp(out_data) * g.sum(axis=d, keepdims=True))

        out._backward = _backward
        return out
