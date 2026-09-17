# Autograd: how NoTorch learns

NoTorch builds a **dynamic computation graph**. Every forward op records
its parents. `backward()` walks that graph in reverse and applies the
chain rule. Nothing static. Nothing compiled. Just a DAG and pain.

## The graph is a DAG

Each `Tensor` stores:

- `data`: the NumPy array.
- `requires_grad`: whether to track.
- `grad`: accumulated gradient (None until backward).
- `_prev`: parent tensors.
- `_backward`: closure that pushes grad to parents.
- `_op`: label for debugging.

Leaf tensors (weights, inputs) have no parents. Every op creates a new
node pointing back. The result is a directed acyclic graph rooted at
the loss.

## Topological sort

`backward()` first collects nodes via DFS from the loss, then visits
them in reverse topological order. This guarantees a node's upstream
gradient is final before its `_backward` runs.

```python
# sketch
order = []
visited = set()

def dfs(node):
    if node not in visited:
        visited.add(node)
        for parent in node._prev:
            dfs(parent)
        order.append(node)

dfs(loss)
loss.grad = ones_like(loss)
for node in reversed(order):
    node._backward()
```

## Chain rule

Each `_backward` implements the local Jacobian-vector product:

```
grad_parent += upstream_grad * local_derivative
```

Example — multiply `c = a * b`:

```python
a.grad += upstream * b.data
b.grad += upstream * a.data
```

Add, matmul, pow, exp, log, reshape, sum — all the same pattern:
take the incoming gradient, multiply by the local derivative, pass it down.

## Gradient accumulation

Grads **accumulate** (`+=`), they don't overwrite. A tensor used twice
(e.g. `x * x`) gets contributions from every path. That's why you call
`optim.zero_grad()` each step — otherwise yesterday's gradients leak
into today's update.

## Broadcasting and sum_to_shape

NumPy broadcasts silently. Gradients can't. If `a: (3, 1)` broadcasts
against `b: (3, 4)` to produce `(3, 4)`, the upstream grad has shape
`(3, 4)` but `a.grad` must be `(3, 1)`.

Fix: `sum_to_shape(grad, shape)` — sum the upstream gradient along every
axis that was broadcast until it matches the parent's shape. Every
binary `_backward` in NoTorch does this. Forget it once and you'll get
a shape error at 2am. Ask me how I know.

## Worked example: f(x) = x * x + x at x = 2

Forward:

```
a = x * x   # 4
f = a + x   # 6
```

Backward (seed `df/df = 1`):

```
df/da = 1            -> grad_a = 1
df/dx (via +) = 1    -> grad_x += 1
df/dx (via *) = 2*x  -> grad_x += 1 * 2 * 2 = 4
```

Total: `df/dx = 2x + 1 = 5` at `x = 2`. Two paths, one accumulation.
That's the whole trick.
