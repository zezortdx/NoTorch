import numpy as np

from notorch import Tensor, manual_seed, nn
from notorch.utils.gradcheck import gradcheck


def _inputs(*shapes, seed=0):
    rng = np.random.default_rng(seed)
    return [Tensor(rng.normal(size=s).astype(np.float64), requires_grad=True) for s in shapes]


def test_gradcheck_mul():
    a, b = _inputs((2, 3), (2, 3))
    passed, _ = gradcheck(lambda ts: (ts[0] * ts[1]).sum(), [a, b])
    assert passed


def test_gradcheck_add():
    a, b = _inputs((2, 3), (2, 3), seed=1)
    passed, _ = gradcheck(lambda ts: (ts[0] + ts[1]).sum(), [a, b])
    assert passed


def test_gradcheck_matmul():
    a, b = _inputs((2, 3), (3, 2), seed=2)
    passed, _ = gradcheck(lambda ts: (ts[0] @ ts[1]).sum(), [a, b])
    assert passed


def test_gradcheck_mean():
    (a,) = _inputs((2, 3), seed=3)
    passed, _ = gradcheck(lambda ts: ts[0].mean(), [a])
    assert passed


def test_gradcheck_softmax():
    (a,) = _inputs((2, 3), seed=4)
    const = np.arange(6, dtype=np.float64).reshape(2, 3) + 0.5
    passed, _ = gradcheck(lambda ts: (ts[0].softmax(dim=-1) * const).sum(), [a])
    assert passed


def test_gradcheck_relu():
    (a,) = _inputs((2, 3), seed=5)
    passed, _ = gradcheck(lambda ts: ts[0].relu().sum(), [a])
    assert passed


def test_gradcheck_tanh():
    (a,) = _inputs((2, 3), seed=6)
    passed, _ = gradcheck(lambda ts: ts[0].tanh().sum(), [a])
    assert passed


def test_gradcheck_sigmoid():
    (a,) = _inputs((2, 3), seed=7)
    passed, _ = gradcheck(lambda ts: ts[0].sigmoid().sum(), [a])
    assert passed


def test_gradcheck_linear():
    manual_seed(0)
    x, w, b = _inputs((2, 4), (4, 3), (3,), seed=8)
    passed, _ = gradcheck(lambda ts: (ts[0] @ ts[1] + ts[2]).sum(), [x, w, b])
    assert passed


def test_gradcheck_nn_linear_module_forward():
    manual_seed(0)
    lin = nn.Linear(4, 3)
    lin.W.data = lin.W.data.astype(np.float64)
    lin.b.data = lin.b.data.astype(np.float64)
    (x,) = _inputs((2, 4), seed=9)
    passed, _ = gradcheck(lambda ts: lin(ts[0]).sum(), [x])
    assert passed
