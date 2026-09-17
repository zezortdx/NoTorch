# Math reference

All ops NumPy-backed. Shapes assume row-major batches: `X: (N, in)`.

## Linear layer

```
Y = X @ W + b
```

- `X: (N, in)`, `W: (in, out)`, `b: (out,)` broadcast over rows.
- Gradients: `dL/dW = X^T @ dL/dY`, `dL/db = sum(dL/dY, axis=0)`, `dL/dX = dL/dY @ W^T`.

## Mean squared error

```
L = (1/N) * sum((pred - target)^2)
```

Gradient: `dL/dpred = 2 * (pred - target) / N`.

## Activations

Sigmoid:

```
s(x) = 1 / (1 + exp(-x)),   s'(x) = s(x) * (1 - s(x))
```

Tanh:

```
t(x) = tanh(x),             t'(x) = 1 - t(x)^2
```

ReLU:

```
r(x) = max(0, x),           r'(x) = 1 if x > 0 else 0
```

Softmax (row-wise, with max-subtraction for stability):

```
p_i = exp(z_i - max(z)) / sum_j exp(z_j - max(z))
```

Cross-entropy (combined softmax + NLL, mean over batch):

```
L = -(1/N) * sum_n log(p[n, y_n])
```

Gradient w.r.t. logits: `dL/dz = (p - one_hot(y)) / N`. Clean, no log-of-zero if you fuse them.

## Chain rule

```
dL/dx = dL/dy * dy/dx
```

For vectors/matrices, replace products with Jacobian-vector products.
NoTorch never forms full Jacobians — each `_backward` applies the
local VJP directly.

## Optimizers

SGD:

```
theta = theta - lr * g
```

SGD with momentum:

```
v = mu * v + g
theta = theta - lr * v
```

Adam:

```
m = b1 * m + (1 - b1) * g
v = b2 * v + (1 - b2) * g^2
m_hat = m / (1 - b1^t),   v_hat = v / (1 - b2^t)
theta = theta - lr * m_hat / (sqrt(v_hat) + eps)
```

Defaults: `lr=1e-3`, `b1=0.9`, `b2=0.999`, `eps=1e-8`.
