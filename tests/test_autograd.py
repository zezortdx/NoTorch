import numpy as np
import pytest

from notorch import Tensor


def _g(t):
    assert t.grad is not None
    return t.grad


def test_add_grad():
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = Tensor([4.0, 5.0], requires_grad=True)
    (x + y).sum().backward()
    np.testing.assert_allclose(_g(x), [1.0, 1.0])
    np.testing.assert_allclose(_g(y), [1.0, 1.0])


def test_sub_grad():
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = Tensor([4.0, 5.0], requires_grad=True)
    (x - y).sum().backward()
    np.testing.assert_allclose(_g(x), [1.0, 1.0])
    np.testing.assert_allclose(_g(y), [-1.0, -1.0])


def test_mul_grad():
    x = Tensor([2.0, 3.0], requires_grad=True)
    y = Tensor([4.0, 5.0], requires_grad=True)
    (x * y).sum().backward()
    np.testing.assert_allclose(_g(x), [4.0, 5.0])
    np.testing.assert_allclose(_g(y), [2.0, 3.0])


def test_div_grad():
    x = Tensor([4.0], requires_grad=True)
    y = Tensor([2.0], requires_grad=True)
    (x / y).sum().backward()
    np.testing.assert_allclose(_g(x), [0.5])
    np.testing.assert_allclose(_g(y), [-1.0])


def test_pow_grad():
    x = Tensor([2.0], requires_grad=True)
    (x**3).sum().backward()
    np.testing.assert_allclose(_g(x), [12.0])


def test_pow_tensor_exponent_raises():
    x = Tensor([2.0], requires_grad=True)
    with pytest.raises(TypeError):
        x ** Tensor([3.0])


def test_neg_grad():
    x = Tensor([1.0, -2.0], requires_grad=True)
    (-x).sum().backward()
    np.testing.assert_allclose(_g(x), [-1.0, -1.0])


def test_branched_graph():
    x = Tensor([3.0], requires_grad=True)
    y = x * x + x
    assert y.data.item() == pytest.approx(12.0)
    y.backward()
    np.testing.assert_allclose(_g(x), [7.0])


def test_repeated_usage_accumulates():
    x = Tensor([2.0], requires_grad=True)
    y = x * x
    y.backward()
    np.testing.assert_allclose(_g(x), [4.0])
    y.backward()
    np.testing.assert_allclose(_g(x), [12.0])


def test_zero_grad_resets():
    x = Tensor([2.0], requires_grad=True)
    (x * x).backward()
    assert x.grad is not None
    x.zero_grad()
    assert x.grad is None


def test_multi_op_chain():
    a = Tensor([2.0], requires_grad=True)
    b = Tensor([3.0], requires_grad=True)
    c = Tensor([4.0], requires_grad=True)
    f = a * b + c
    f.backward()
    np.testing.assert_allclose(_g(a), [3.0])
    np.testing.assert_allclose(_g(b), [2.0])
    np.testing.assert_allclose(_g(c), [1.0])


def test_matmul_forward_backward():
    A = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    B = Tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], requires_grad=True)
    C = A @ B
    np.testing.assert_allclose(C.data, np.array([[4.0, 5.0], [10.0, 11.0]]))
    C.sum().backward()
    np.testing.assert_allclose(_g(A), np.ones((2, 2)) @ B.data.T)
    np.testing.assert_allclose(_g(B), A.data.T @ np.ones((2, 2)))


def test_sum_grad():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    x.sum().backward()
    np.testing.assert_allclose(_g(x), np.ones((2, 2)))


def test_sum_axis_grad():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    x.sum(axis=0).backward(gradient=np.ones(2))
    np.testing.assert_allclose(_g(x), np.ones((2, 2)))


def test_mean_grad():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    x.mean().backward()
    np.testing.assert_allclose(_g(x), np.full((2, 2), 0.25))


def test_mean_axis_grad():
    x = Tensor([[1.0, 3.0], [2.0, 4.0]], requires_grad=True)
    x.mean(axis=1).backward(gradient=np.ones(2))
    np.testing.assert_allclose(_g(x), np.full((2, 2), 0.5))


def test_reshape_grad():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = x.reshape(4)
    assert y.shape == (4,)
    y.backward(gradient=np.ones(4))
    np.testing.assert_allclose(_g(x), np.ones((2, 2)))


def test_transpose_grad():
    x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    y = x.transpose()
    assert y.shape == (3, 1)
    y.backward(gradient=np.ones((3, 1)))
    np.testing.assert_allclose(_g(x), np.ones((1, 3)))


def test_relu():
    d = np.array([-2.0, 0.5, 3.0])
    x = Tensor(d, requires_grad=True)
    y = x.relu()
    np.testing.assert_allclose(y.data, [0.0, 0.5, 3.0])
    y.sum().backward()
    np.testing.assert_allclose(_g(x), [0.0, 1.0, 1.0])


def test_sigmoid():
    d = np.array([0.0])
    x = Tensor(d, requires_grad=True)
    y = x.sigmoid()
    np.testing.assert_allclose(y.data, [0.5])
    y.sum().backward()
    np.testing.assert_allclose(_g(x), [0.25])


def test_tanh():
    d = np.array([0.0, 1.0])
    x = Tensor(d, requires_grad=True)
    y = x.tanh()
    np.testing.assert_allclose(y.data, np.tanh(d))
    y.sum().backward()
    np.testing.assert_allclose(_g(x), 1 - np.tanh(d) ** 2)


def test_exp_log():
    d = np.array([0.0, 1.0, 2.0])
    x = Tensor(d, requires_grad=True)
    (x.exp().sum()).backward()
    np.testing.assert_allclose(_g(x), np.exp(d))
    z = Tensor(d + 1.0, requires_grad=True)
    (z.log().sum()).backward()
    np.testing.assert_allclose(_g(z), 1.0 / (d + 1.0))


def test_backward_requires_grad_false_raises():
    with pytest.raises(RuntimeError):
        Tensor([1.0]).backward()


def test_backward_nonscalar_needs_gradient():
    with pytest.raises(RuntimeError):
        Tensor([1.0, 2.0], requires_grad=True).backward()


def test_getitem_and_max_smoke():
    x = Tensor([[1.0, 5.0], [3.0, 2.0]], requires_grad=True)
    row = x[0]
    np.testing.assert_allclose(row.data, [1.0, 5.0])
    row.sum().backward()
    np.testing.assert_allclose(_g(x), [[1.0, 1.0], [0.0, 0.0]])
    m = Tensor([[1.0, 5.0]], requires_grad=True).max()
    assert m.data.item() == pytest.approx(5.0)
