# Contributing to NoTorch

Keep it small. Keep it from scratch. Keep it readable.

## Setup

```bash
pip install -e ".[dev]"
pytest
```

## Rules

1. No PyTorch / TensorFlow / JAX imports. NumPy only (+ stdlib).
2. Every op in `Tensor` needs a backward. No backward, no merge.
3. Add a test. Run `pytest` before you push.
4. Run `ruff check .` — line length 100.
5. Small PRs. One thing per PR.

## What we want

- Bug fixes in autograd / nn / optim.
- Missing activations, losses, schedulers.
- Better docs, better examples, better error messages.

## What we don't want

- Heavy dependencies.
- Magic. If a beginner can't read it, simplify it.
