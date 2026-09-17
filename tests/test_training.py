import numpy as np

from notorch import Tensor, load, manual_seed, nn, optim, save
from notorch.data import DataLoader


def _seeded_init(model, seed):
    rng = np.random.default_rng(seed)
    for _, p in model.named_parameters():
        nn.init.xavier_uniform_(p, rng=rng)
    return model


def test_mlp_bce_learns_or():
    manual_seed(0)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y = np.array([[0], [1], [1], [1]], dtype=np.float64)
    model = _seeded_init(
        nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid()), seed=0
    )
    loss_fn = nn.BinaryCrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=0.05)
    for _ in range(500):
        opt.zero_grad()
        loss = loss_fn(model(Tensor(X)), Tensor(y))
        loss.backward()
        opt.step()
    final = loss_fn(model(Tensor(X)), Tensor(y)).data.item()
    pred = (model(Tensor(X)).data > 0.5).astype(int)
    assert final < 0.1
    assert (pred == y.astype(int)).all()


def test_linear_regression_mse_learns():
    manual_seed(0)
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(20, 1))
    y = 2 * X + 1
    model = _seeded_init(nn.Linear(1, 1), seed=0)
    loss_fn = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=0.05)
    for _ in range(400):
        opt.zero_grad()
        loss = loss_fn(model(Tensor(X)), Tensor(y))
        loss.backward()
        opt.step()
    final = loss_fn(model(Tensor(X)), Tensor(y)).data.item()
    assert final < 1e-4
    assert abs(float(model.W.data[0, 0]) - 2.0) < 0.05
    assert abs(float(model.b.data[0]) - 1.0) < 0.05


def test_crossentropy_3class_learns():
    manual_seed(42)
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [2.0, 3.0],
            [3.0, 2.0],
            [3.0, 3.0],
            [0.0, 3.0],
            [1.0, 3.0],
            [0.0, 2.0],
            [1.0, 2.0],
        ]
    )
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    model = _seeded_init(nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 3)), seed=42)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=0.05)
    for _ in range(400):
        opt.zero_grad()
        loss = loss_fn(model(Tensor(X)), y)
        loss.backward()
        opt.step()
    final = loss_fn(model(Tensor(X)), y).data.item()
    pred = np.argmax(model(Tensor(X)).data, axis=1)
    assert final < 0.01
    assert (pred == y).all()


def test_save_load_roundtrip(tmp_path):
    manual_seed(0)
    model = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1))
    path = str(tmp_path / "model.npz")
    save(model, path)
    restored = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1))
    restored.load_state_dict(load(None, path))
    for (k1, v1), (k2, v2) in zip(
        sorted(model.state_dict().items()), sorted(restored.state_dict().items())
    ):
        assert k1 == k2
        np.testing.assert_array_equal(v1, v2)
    x = Tensor(np.array([[0.5, -1.2]]))
    np.testing.assert_array_equal(model(x).data, restored(x).data)


def test_dataloader_batching_deterministic():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    first = [(xb.copy(), yb.copy()) for xb, yb in DataLoader(X, y, batch_size=4, seed=0)]
    second = [(xb.copy(), yb.copy()) for xb, yb in DataLoader(X, y, batch_size=4, seed=0)]
    assert len(first) == 3
    for (xa, ya), (xb, yb) in zip(first, second):
        np.testing.assert_array_equal(xa, xb)
        np.testing.assert_array_equal(ya, yb)
    seen = np.sort(np.concatenate([yb for _, yb in first]))
    np.testing.assert_array_equal(seen, np.arange(10))
    assert [len(yb) for _, yb in first] == [4, 4, 2]
