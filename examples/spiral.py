"""Spiral classification demo for NoTorch.

3-class spiral generated with numpy. MLP: 2 -> hidden -> hidden -> 3
with ReLU, trained with CrossEntropyLoss + Adam.

Run:
    python examples/spiral.py --epochs 2000 --lr 1e-3 --hidden 64 --samples 100
    python examples/spiral.py --plot   # needs matplotlib (optional)
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from notorch import Tensor, nn, optim

try:
    import notorch
except ImportError:  # pragma: no cover
    notorch = None  # type: ignore


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    if notorch is not None and hasattr(notorch, "manual_seed"):
        try:
            notorch.manual_seed(seed)
        except Exception:
            pass


def make_spiral(
    samples_per_class: int = 100, n_classes: int = 3, noise: float = 0.2, seed: int = 0
):
    rng = np.random.RandomState(seed)
    n = samples_per_class
    X = np.zeros((n * n_classes, 2), dtype=np.float32)
    y = np.zeros(n * n_classes, dtype=np.int64)
    for j in range(n_classes):
        ix = range(n * j, n * (j + 1))
        r = np.linspace(0.0, 1.0, n, dtype=np.float64)
        t = np.linspace(j * 4.0, (j + 1) * 4.0, n) + rng.randn(n) * noise
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)].astype(np.float32)
        y[ix] = j
    return X, y


def to_numpy(t):
    if hasattr(t, "numpy"):
        try:
            return np.asarray(t.numpy())
        except Exception:
            pass
    if hasattr(t, "data"):
        try:
            return np.asarray(t.data)
        except Exception:
            pass
    return np.asarray(t)


def to_scalar(t) -> float:
    if hasattr(t, "item"):
        try:
            return float(t.item())
        except Exception:
            pass
    return float(to_numpy(t).reshape(-1)[0])


class SpiralNet(nn.Module):
    def __init__(self, hidden: int = 64, n_classes: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden, hidden)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden, n_classes)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.fc3(x)


def accuracy(logits_np: np.ndarray, targets: np.ndarray) -> float:
    return float((logits_np.argmax(axis=1) == targets).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description="NoTorch spiral demo")
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--samples", type=int, default=100, help="samples per class")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.2)
    ap.add_argument("--batch-size", type=int, default=0, help="0 = full-batch, else minibatch size")
    ap.add_argument(
        "--plot", action="store_true", help="scatter-plot decision boundary (needs matplotlib)"
    )
    args = ap.parse_args()

    set_seed(args.seed)
    X_np, y_np = make_spiral(args.samples, n_classes=3, noise=args.noise, seed=args.seed)

    model = SpiralNet(hidden=args.hidden, n_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    n = X_np.shape[0]
    bs = args.batch_size if 0 < args.batch_size < n else n
    log_every = max(1, args.epochs // 10)

    for epoch in range(1, args.epochs + 1):
        if bs < n:
            perm = np.random.permutation(n)
            for i in range(0, n, bs):
                idx = perm[i : i + bs]
                xb, yb = Tensor(X_np[idx]), Tensor(y_np[idx])
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
        else:
            optimizer.zero_grad()
            loss = criterion(model(Tensor(X_np)), Tensor(y_np))
            loss.backward()
            optimizer.step()
        if epoch == 1 or epoch % log_every == 0 or epoch == args.epochs:
            logits = to_numpy(model(Tensor(X_np)))
            acc = accuracy(logits, y_np)
            print(
                f"epoch {epoch:5d}/{args.epochs}  loss={to_scalar(loss):.4f}  acc={acc * 100:5.1f}%"
            )

    logits = to_numpy(model(Tensor(X_np)))
    print(f"final acc: {accuracy(logits, y_np) * 100:.1f}%")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed; skipping --plot.")
            return
        h = 0.02
        x_min, x_max = X_np[:, 0].min() - 0.2, X_np[:, 0].max() + 0.2
        y_min, y_max = X_np[:, 1].min() - 0.2, X_np[:, 1].max() + 0.2
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
        step = 2000
        out = np.concatenate(
            [
                to_numpy(model(Tensor(grid[i : i + step]))).argmax(axis=1)
                for i in range(0, len(grid), step)
            ]
        )
        plt.contourf(xx, yy, out.reshape(xx.shape), alpha=0.3)
        plt.scatter(X_np[:, 0], X_np[:, 1], c=y_np, edgecolors="k")
        plt.title("NoTorch spiral")
        plt.show()


if __name__ == "__main__":
    main()
