# NoTorch

Neural networks from scratch.

No PyTorch.
No TensorFlow.
Just NumPy and pain.

![tests](https://github.com/zezortdx/NoTorch/actions/workflows/tests.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

## What is this

NoTorch is a tiny educational deep learning framework built on top of NumPy. Let's be honest about the division of labor: NumPy does array storage and number crunching. NoTorch implements the deep-learning machinery on top — computation graph, autodiff, gradient propagation, layers, losses, optimizers, training loops.

It exists for one reason: so you actually understand what `loss.backward()` does.

## Why

Because `import torch` teaches you API. Rebuilding autograd teaches you the thing itself — computation graphs, chain rule, broadcasting gradients, Adam bias correction. Every line is readable. No C++ backend. No magic.

## Features

- `Tensor` with dynamic autograd (add, mul, matmul, pow, exp, log, sum, reshape, ... )
- `nn.Module`, `Linear`, `Sequential`, ReLU / Sigmoid / Tanh
- MSELoss, BinaryCrossEntropyLoss, CrossEntropyLoss (fused softmax)
- SGD (+ momentum), Adam
- `DataLoader` with shuffle/batching (`TensorDataset`, minibatches, `batches_per_epoch`)
- `gradcheck`, `manual_seed`, save/load (`.npz`)
- Test suite + docs that show the math, not just the API

## Install

```bash
pip install -e ".[dev]"
```

or a plain local install:

```bash
pip install .
```

Requires Python >= 3.11, `numpy>=1.24`. Not on PyPI — install from source. (The distribution is named `notorch-zero`; you still `import notorch`. Don't ask.)

## 60-second example

```python
import numpy as np
import notorch
from notorch import Tensor, nn

notorch.manual_seed(0)

model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
opt = notorch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()

X = np.random.randn(64, 2)
y = (X[:, :1] * X[:, 1:])  # learn the product. good luck.

for epoch in range(200):
    opt.zero_grad()
    pred = model(Tensor(X))
    loss = loss_fn(pred, Tensor(y))
    loss.backward()
    opt.step()
    if epoch % 50 == 0:
        print(f"epoch {epoch} loss {loss.data:.4f}")
```

## Autograd in 30 seconds

Forward ops build a graph. `backward()` walks it in reverse, combining local derivatives with the chain rule. Grads accumulate into `.grad` (that's why you `zero_grad()`). Broadcasting is undone with `sum_to_shape`. Full writeup: `docs/autograd.md`.

```python
x = Tensor(2.0, requires_grad=True)
f = x * x + x   # 6.0
f.backward()    # df/dx = 2x + 1 = 5.0
print(x.grad)   # 5.0
```

Math reference (Linear, MSE, activations, softmax/CE, SGD/Adam): `docs/math.md`.

## Architecture

```mermaid
flowchart LR
    T[Tensor<br/>autograd engine] --> M[nn.Module<br/>Linear + activations]
    M --> L[Loss<br/>MSE / BCE / CrossEntropy]
    L --> O[Optim<br/>SGD / Adam]
    O --> T
    D[Data<br/>DataLoader] --> M
```

Package map and design rules: `docs/architecture.md`.

## Training examples

```bash
python examples/xor.py        # MLP learns XOR. the hello-world of nonlinear pain.
python examples/spiral.py     # 2-layer MLP on three spirals. watch decision boundary bend.
python examples/mnist.py      # MLP on MNIST. digits, from scratch, no torchvision.
```

- **XOR**: 2-8-8-1 MLP (Tanh hidden, Sigmoid out), BCELoss, Adam, full-batch. Converges in seconds.
- **Spiral**: 3-class spiral, 2-64-64-3 MLP (ReLU) + CrossEntropyLoss + Adam. The reason hidden layers exist.
- **MNIST**: 784-128-64-10 MLP, Adam, DataLoader batches. Downloads once via stdlib, caches under `data/mnist/`. Typical results land in the high 90s on test accuracy depending on seed and config — your mileage may vary.

## Project structure

```
src/notorch/      # tensor, nn, optim, data, utils
tests/            # pytest suite
examples/         # xor, spiral, mnist
docs/             # autograd, math, architecture
```

## Limitations

- CPU + NumPy only. No GPU in v0.1.0. No speed records will be broken.
- Educational by design. It will not dethrone PyTorch, and it isn't trying to.
- No Conv2D, no RNNs, no BatchNorm, no Dropout. Yet.
- Second-order grads? Never heard of them.
- Large models will be slow. That's the tuition fee for understanding.

## Roadmap

- [ ] Conv2D + MaxPool
- [ ] BatchNorm / LayerNorm
- [ ] Dropout
- [ ] LR schedulers (Step, Cosine)
- [ ] GPU backend (or at least CuPy-shaped dreams)
- [ ] More examples (CNN on MNIST, char-RNN)

## Contributing

Small PRs. NumPy only. Every op needs a backward and a test. See `CONTRIBUTING.md`.

## License

MIT. See `LICENSE`.

## Acknowledgments

- karpathy/micrograd — proved a weekend and a DAG is all you need.
- Every autograd tutorial that lied about broadcasting gradients being easy.
- NumPy, for doing all the actual work.
