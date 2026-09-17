import numpy as np
import pytest

from notorch import Tensor, manual_seed, nn
from notorch.utils.gradcheck import gradcheck


def test_mse_values_and_reductions():
    loss = nn.MSELoss()
    out = Tensor([[1.0, 2.0, 3.0]])
    tgt = Tensor([[1.0, 2.0, 4.0]])
    assert loss(out, tgt).data.item() == pytest.approx(1.0 / 3.0)
    assert nn.MSELoss(reduction="sum")(out, tgt).data.item() == pytest.approx(1.0)
    none = nn.MSELoss(reduction="none")(out, tgt)
    np.testing.assert_allclose(none.data, [[0.0, 0.0, 1.0]])
    with pytest.raises(ValueError):
        nn.MSELoss(reduction="bogus")


def test_bce_values():
    loss = nn.BinaryCrossEntropyLoss()
    probs = Tensor([[0.9, 0.1]])
    tgt = [[1.0, 0.0]]
    got = loss(probs, tgt).data.item()
    assert got == pytest.approx(-np.log(0.9))
    assert nn.BinaryCrossEntropyLoss(reduction="sum")(probs, tgt).data.item() == pytest.approx(
        -2 * np.log(0.9)
    )
    none = nn.BinaryCrossEntropyLoss(reduction="none")(probs, tgt)
    assert none.shape == (1, 2)


def test_crossentropy_matches_manual_log_softmax_nll():
    manual_seed(0)
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(5, 4)).astype(np.float64)
    targets = np.array([0, 1, 2, 3, 1])
    ce = nn.CrossEntropyLoss()
    got = ce(Tensor(logits), targets).data.item()
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_sm = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    expected = -log_sm[np.arange(5), targets].mean()
    assert got == pytest.approx(expected)


def test_crossentropy_reductions():
    logits = Tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])
    targets = [0, 1]
    mean = nn.CrossEntropyLoss(reduction="mean")(logits, targets).data.item()
    total = nn.CrossEntropyLoss(reduction="sum")(logits, targets).data.item()
    assert total == pytest.approx(2 * mean)
    none = nn.CrossEntropyLoss(reduction="none")(logits, targets)
    assert none.shape == (2,)
    assert none.data.sum() == pytest.approx(total)


def test_crossentropy_bad_targets_raise():
    with pytest.raises(ValueError):
        nn.CrossEntropyLoss()(Tensor([[1.0, 2.0]]), [7])


def test_gradcheck_mse():
    manual_seed(0)
    rng = np.random.default_rng(2)
    x = Tensor(rng.normal(size=(3, 2)).astype(np.float64), requires_grad=True)
    t = rng.normal(size=(3, 2)).astype(np.float64)
    passed, _ = gradcheck(lambda ts: nn.MSELoss()(ts[0], t), [x])
    assert passed


def test_gradcheck_bce():
    manual_seed(0)
    rng = np.random.default_rng(3)
    x = Tensor(rng.uniform(0.2, 0.8, size=(4, 1)).astype(np.float64), requires_grad=True)
    t = (rng.uniform(size=(4, 1)) > 0.5).astype(np.float64)
    passed, _ = gradcheck(lambda ts: nn.BinaryCrossEntropyLoss()(ts[0], t), [x])
    assert passed


def test_gradcheck_crossentropy():
    manual_seed(0)
    rng = np.random.default_rng(4)
    x = Tensor(rng.normal(size=(4, 3)).astype(np.float64), requires_grad=True)
    t = np.array([0, 1, 2, 1])
    passed, _ = gradcheck(lambda ts: nn.CrossEntropyLoss()(ts[0], t), [x])
    assert passed
