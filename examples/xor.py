"""XOR demo for NoTorch.

MLP: 2 -> 8 -> 8 -> 1, Tanh hidden activations, Sigmoid output.
Loss: BCELoss. Optimizer: Adam. Full-batch training.

Run:
    python examples/xor.py
"""

import sys
from pathlib import Path

import numpy as np

# Allow `python examples/xor.py` from repo root with src layout.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from notorch import Tensor, nn, optim

try:
    import notorch
except ImportError:  # pragma: no cover
    notorch = None  # type: ignore


def set_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    if notorch is not None and hasattr(notorch, "manual_seed"):
        try:
            notorch.manual_seed(seed)
        except Exception:
            pass


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


class XORNet(nn.Module):
    def __init__(self, hidden: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden)
        self.act1 = nn.Tanh()
        self.fc2 = nn.Linear(hidden, hidden)
        self.act2 = nn.Tanh()
        self.fc3 = nn.Linear(hidden, 1)
        self.out = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.out(self.fc3(x))


def main() -> None:
    set_seed(0)

    X_np = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y_np = np.array([[0], [1], [1], [0]], dtype=np.float32)
    X = Tensor(X_np)
    y = Tensor(y_np)

    model = XORNet(hidden=8)
    criterion = nn.BinaryCrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 3000
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 500 == 0 or epoch == epochs:
            print(f"epoch {epoch:4d}/{epochs}  loss={to_scalar(loss):.4f}")

    with_probs = to_numpy(model(X)).reshape(-1)
    preds = (with_probs > 0.5).astype(int)
    targets = y_np.reshape(-1).astype(int)
    acc = (preds == targets).mean()
    print(f"probs: {with_probs.round(3).tolist()}")
    print(f"preds: {preds.tolist()}  targets: {targets.tolist()}")
    print(f"accuracy: {acc * 100:.1f}%")
    if acc < 1.0:
        print("Note: XOR not fully learned; try re-running or more epochs.")


if __name__ == "__main__":
    main()
