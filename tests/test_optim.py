import numpy as np

from notorch import Tensor, optim


def _quadratic(p, target=3.0):
    return ((p - target) ** 2).sum()


def test_sgd_single_step_quadratic():
    p = Tensor([5.0], requires_grad=True)
    opt = optim.SGD([p], lr=0.1)
    loss = _quadratic(p)
    loss.backward()
    np.testing.assert_allclose(p.grad, [4.0])
    opt.step()
    np.testing.assert_allclose(p.data, [4.6])


def test_sgd_momentum_state_accelerates():
    opt_steps = []
    for _ in range(2):
        pass
    p = Tensor([5.0], requires_grad=True)
    opt = optim.SGD([p], lr=0.1, momentum=0.9)
    _quadratic(p).backward()
    first_before = p.data.copy()
    opt.step()
    first_move = first_before - p.data.copy()
    opt.zero_grad()
    _quadratic(p).backward()
    before = p.data.copy()
    opt.step()
    second_move = before - p.data.copy()
    assert len(opt.state) == 1
    state = next(iter(opt.state.values()))
    assert "velocity" in state
    assert (second_move > first_move).all()
    opt_steps.append((first_move, second_move))


def test_adam_step_decreases_loss():
    p = Tensor([5.0], requires_grad=True)
    opt = optim.Adam([p], lr=0.1)
    before = _quadratic(p).data.item()
    _quadratic(p).backward()
    opt.step()
    after = _quadratic(p).data.item()
    assert after < before
    np.testing.assert_allclose(p.data, [4.9], atol=1e-6)


def test_zero_grad_clears():
    p = Tensor([5.0], requires_grad=True)
    q = Tensor([1.0], requires_grad=True)
    opt = optim.SGD([p, q], lr=0.01)
    _quadratic(p).backward()
    (q * q).sum().backward()
    opt.zero_grad()
    assert p.grad is None
    assert q.grad is None
