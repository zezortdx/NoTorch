import numpy as np
import pytest

from notorch import Tensor
from notorch.nn import Parameter


def test_construct_from_list_defaults_float64():
    t = Tensor([1, 2, 3])
    assert t.shape == (3,)
    assert t.dtype == np.dtype(np.float64)
    assert np.array_equal(t.data, np.array([1.0, 2.0, 3.0]))


def test_construct_matrix_shape():
    t = Tensor([[1, 2], [3, 4]])
    assert t.shape == (2, 2)
    assert t.ndim == 2


def test_construct_scalar():
    t = Tensor(3.0)
    assert t.shape == ()
    assert t.data.size == 1
    assert t.item() == 3.0


def test_float32_preserved():
    t = Tensor(np.ones((2, 2), dtype=np.float32))
    assert t.dtype == np.dtype(np.float32)


def test_float64_preserved():
    t = Tensor(np.ones((2, 2), dtype=np.float64))
    assert t.dtype == np.dtype(np.float64)


def test_dtype_override():
    t = Tensor([1, 2], dtype=np.float32)
    assert t.dtype == np.dtype(np.float32)


def test_invalid_dtype_raises():
    with pytest.raises(TypeError):
        Tensor([1, 2], dtype=np.int32)


def test_requires_grad_default_false():
    assert Tensor([1.0]).requires_grad is False
    assert Tensor([1.0]).grad is None


def test_requires_grad_true():
    t = Tensor([1.0], requires_grad=True)
    assert t.requires_grad is True
    assert t.grad is None


def test_parameter_requires_grad_by_default():
    p = Parameter([1.0, 2.0])
    assert p.requires_grad is True
    assert p.shape == (2,)


def test_repr_contents():
    t = Tensor([[1.0]], requires_grad=True)
    r = repr(t)
    assert "Tensor" in r
    assert "shape" in r
    assert "requires_grad=True" in r


def test_zeros_ones_classmethods():
    z = Tensor.zeros((2, 3))
    o = Tensor.ones((2, 3))
    assert z.shape == (2, 3)
    assert np.array_equal(z.data, np.zeros((2, 3)))
    assert np.array_equal(o.data, np.ones((2, 3)))


def test_zeros_ones_like():
    t = Tensor([[1.0, 2.0]])
    assert t.zeros_like().shape == (1, 2)
    assert t.ones_like().shape == (1, 2)
    assert (t.zeros_like().data == 0).all()


def test_detach():
    t = Tensor([1.0, 2.0], requires_grad=True)
    d = t.detach()
    assert d.requires_grad is False
    assert np.array_equal(d.data, t.data)
    assert d.data is not t.data


def test_numpy_returns_copy():
    t = Tensor([1.0])
    arr = t.numpy()
    arr[0] = 99.0
    assert t.data[0] == 1.0


def test_item_and_tolist():
    assert Tensor(2.5).item() == 2.5
    assert Tensor([[1, 2]]).tolist() == [[1.0, 2.0]]


def test_transpose_and_flatten_shapes():
    t = Tensor(np.ones((2, 3)))
    assert t.T.shape == (3, 2)
    assert t.flatten().shape == (6,)


def test_name_label_alias():
    t = Tensor([1.0], name="w")
    assert t.label == "w"
    t.label = "b"
    assert t.name == "b"
