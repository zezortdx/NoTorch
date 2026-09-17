import math

import numpy as np

from notorch import Tensor, manual_seed, nn


def test_linear_shapes_and_forward():
    manual_seed(0)
    lin = nn.Linear(4, 3)
    assert lin.W.shape == (4, 3)
    assert lin.b.shape == (3,)
    out = lin(Tensor(np.ones((5, 4))))
    assert out.shape == (5, 3)


def test_linear_no_bias():
    lin = nn.Linear(4, 3, bias=False)
    assert lin.b is None
    assert len(lin.parameters()) == 1
    out = lin(Tensor(np.ones((2, 4))))
    assert out.shape == (2, 3)


def test_parameters_recursive_nested_sequential():
    model = nn.Sequential(nn.Linear(2, 3), nn.Sequential(nn.Linear(3, 1), nn.ReLU()))
    params = model.parameters()
    assert len(params) == 4
    names = [name for name, _ in model.named_parameters()]
    assert "0.W" in names
    assert "0.b" in names
    assert "1.0.W" in names
    assert "1.0.b" in names


def test_train_eval_propagates():
    model = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
    model.eval()
    assert model.training is False
    assert all(m.training is False for m in model.modules())
    model.train()
    assert model.training is True
    assert all(m.training is True for m in model.modules())


def test_xavier_uniform_bound():
    t = Tensor(np.zeros((10, 20), dtype=np.float64))
    nn.init.xavier_uniform_(t, rng=0)
    limit = math.sqrt(6.0 / (10 + 20))
    assert (np.abs(t.data) <= limit).all()


def test_he_uniform_bound():
    t = Tensor(np.zeros((8, 4), dtype=np.float64))
    nn.init.he_uniform_(t, rng=0)
    limit = math.sqrt(6.0 / 8)
    assert (np.abs(t.data) <= limit).all()


def test_zeros_normal_uniform_init():
    t = Tensor(np.ones((3, 3), dtype=np.float64))
    nn.init.zeros_(t)
    assert (t.data == 0).all()
    nn.init.normal_(t, mean=0.0, std=1.0, rng=0)
    assert t.shape == (3, 3)
    nn.init.uniform_(t, a=-1.0, b=1.0, rng=0)
    assert ((t.data >= -1.0) & (t.data <= 1.0)).all()


def test_flatten():
    flat = nn.Flatten(start_dim=1)
    x = Tensor(np.ones((2, 3, 4)), requires_grad=True)
    y = flat(x)
    assert y.shape == (2, 12)
    y.sum().backward()
    np.testing.assert_allclose(x.grad, np.ones((2, 3, 4)))
    assert nn.Flatten(start_dim=0)(Tensor(np.ones((2, 3)))).shape == (6,)
