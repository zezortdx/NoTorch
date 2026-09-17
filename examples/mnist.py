"""MNIST demo for NoTorch (numpy + stdlib only).

Downloads the original IDX files with stdlib urllib, transparently trying
mirrors, caches them under --data-dir, parses the gzip IDX format by hand,
normalizes pixels to [0, 1], and trains an MLP 784 -> 128 -> 64 -> 10 with
CrossEntropyLoss + Adam.

NO torch / torchvision / sklearn dependency.

Run:
    python examples/mnist.py --samples 10000 --epochs 5 --batch-size 64
"""

import argparse
import gzip
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from notorch import Tensor, nn, optim

try:
    import notorch
except ImportError:  # pragma: no cover
    notorch = None  # type: ignore

MIRRORS = [
    "http://yann.lecun.com/exdb/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]

FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    if notorch is not None and hasattr(notorch, "manual_seed"):
        try:
            notorch.manual_seed(seed)
        except Exception:
            pass


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "notorch"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)


def ensure_file(name: str, data_dir: Path) -> Path:
    dest = data_dir / name
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    last_err = None
    for base in MIRRORS:
        try:
            print(f"downloading {base + name} ...")
            download(base + name, dest)
            if dest.stat().st_size > 0:
                return dest
        except Exception as e:  # try next mirror
            last_err = e
            continue
    raise RuntimeError(f"failed to download {name}: {last_err}")


def parse_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"bad magic {magic} in {path}"
        buf = f.read()
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(n, rows * cols)
    return arr.astype(np.float32) / 255.0


def parse_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, _n = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"bad magic {magic} in {path}"
        buf = f.read()
    return np.frombuffer(buf, dtype=np.uint8).astype(np.int64)


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


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.fc3(x)


def evaluate(model, X: np.ndarray, y: np.ndarray, batch: int = 512) -> float:
    correct, total = 0, 0
    for i in range(0, len(X), batch):
        logits = to_numpy(model(Tensor(X[i : i + batch]))).argmax(axis=1)
        correct += int((logits == y[i : i + batch]).sum())
        total += len(logits)
    return correct / max(total, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="NoTorch MNIST demo")
    ap.add_argument(
        "--samples", type=int, default=10000, help="train subset size (<=0 = all 60000)"
    )
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--data-dir", type=str, default="./data/mnist")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    paths = {name: ensure_file(name, data_dir) for name in FILES}

    X_train = parse_idx_images(paths["train-images-idx3-ubyte.gz"])
    y_train = parse_idx_labels(paths["train-labels-idx1-ubyte.gz"])
    X_test = parse_idx_images(paths["t10k-images-idx3-ubyte.gz"])
    y_test = parse_idx_labels(paths["t10k-labels-idx1-ubyte.gz"])
    print(f"train {X_train.shape} test {X_test.shape}")

    if args.samples and args.samples > 0:
        X_train, y_train = X_train[: args.samples], y_train[: args.samples]

    model = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    n = len(X_train)
    for epoch in range(1, args.epochs + 1):
        perm = np.random.permutation(n)
        total_loss, seen = 0.0, 0
        for i in range(0, n, args.batch_size):
            idx = perm[i : i + args.batch_size]
            xb, yb = Tensor(X_train[idx]), Tensor(y_train[idx])
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += to_scalar(loss) * len(idx)
            seen += len(idx)
        train_acc = evaluate(model, X_train, y_train)
        test_acc = evaluate(model, X_test, y_test)
        print(
            f"epoch {epoch}/{args.epochs}  loss={total_loss / seen:.4f}  "
            f"train_acc={train_acc * 100:5.1f}%  test_acc={test_acc * 100:5.1f}%"
        )


if __name__ == "__main__":
    main()
