# Examples

Small numpy-only demos for the NoTorch public API
(`Tensor`, `nn`, `optim`, `DataLoader`).

- `xor.py` — MLP 2→8→8→1 (Tanh + Sigmoid, BCELoss, Adam, full-batch).
  `python examples/xor.py`
- `spiral.py` — 3-class numpy spiral, MLP 2→64→64→3
  (CrossEntropyLoss + Adam). `python examples/spiral.py --help`
  (`--plot` needs matplotlib, optional).
- `mnist.py` — stdlib-only IDX download (lecun + googleapis mirror,
  cached in `./data/mnist`), hand-parsed gzip IDX, MLP 784→128→64→10.
  `python examples/mnist.py --help`
