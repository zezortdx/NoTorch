import numpy as np

from notorch import Tensor, manual_seed
from notorch.utils.gradcheck import gradcheck


def test_add_row_vector_broadcast():
    manual_seed(0)
    rng = np.random.default_rng(0)
    a = Tensor(rng.normal(size=(32, 10)), requires_grad=True)
    b = Tensor(rng.normal(size=(10,)), requires_grad=True)
    c = a + b
    assert c.shape == (32, 10)
    c.sum().backward()
    np.testing.assert_allclose(a.grad, np.ones((32, 10)))
    np.testing.assert_allclose(b.grad, np.full((10,), 32.0))


def test_mul_column_row_broadcast():
    a = Tensor(np.ones((4, 1)), requires_grad=True)
    b = Tensor(np.ones((1, 5)), requires_grad=True)
    c = a * b
    assert c.shape == (4, 5)
    c.sum().backward()
    np.testing.assert_allclose(a.grad, np.full((4, 1), 5.0))
    np.testing.assert_allclose(b.grad, np.full((1, 5), 4.0))


def test_scalar_broadcast():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    (x + 5.0).sum().backward()
    np.testing.assert_allclose(x.grad, np.ones(3))
    z = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    (z * 2.0).sum().backward()
    np.testing.assert_allclose(z.grad, np.full(3, 2.0))


def test_mean_broadcast_gradcheck():
    manual_seed(0)
    rng = np.random.default_rng(1)
    a = Tensor(rng.normal(size=(4, 3)).astype(np.float64), requires_grad=True)
    b = Tensor(rng.normal(size=(3,)).astype(np.float64), requires_grad=True)
    passed, _ = gradcheck(lambda ts: (ts[0] + ts[1]).mean(), [a, b])
    assert passed
