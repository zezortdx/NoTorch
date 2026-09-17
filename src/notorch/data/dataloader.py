"""Data loading utilities for NoTorch (numpy-only)."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


class TensorDataset:
    """Minimal dataset holding feature/label arrays.

    Args:
        X: Features array of shape ``(n_samples, ...)``.
        y: Optional labels array of shape ``(n_samples, ...)``.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray | None = None) -> None:
        self.X = np.asarray(X)
        self.y = np.asarray(y) if y is not None else None
        if self.y is not None and len(self.X) != len(self.y):
            raise ValueError("X and y must have the same first dimension.")

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray | None]:
        """Return the ``(x, y)`` sample at ``idx`` (``y`` may be None)."""
        if self.y is None:
            return self.X[idx], None
        return self.X[idx], self.y[idx]


class DataLoader:
    """Batched data iterator over numpy arrays.

    Args:
        X: Features array of shape ``(n_samples, ...)``.
        y: Optional labels array of shape ``(n_samples, ...)``.
        batch_size: Number of samples per batch (must be >= 1).
        shuffle: Whether to shuffle sample order each epoch.
        seed: Optional seed for deterministic shuffling. Uses
            ``np.random.Generator``; a seeded loader produces a
            reproducible shuffle stream.
        drop_last: Whether to drop the final incomplete batch.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int | None = None,
        drop_last: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        self.X = np.asarray(X)
        self.y = np.asarray(y) if y is not None else None
        if self.X.shape[0] == 0:
            raise ValueError("X must contain at least one sample.")
        if self.y is not None and self.y.shape[0] != self.X.shape[0]:
            raise ValueError("X and y must have the same first dimension.")
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = seed
        self.drop_last = bool(drop_last)
        self._rng: np.random.Generator | None = (
            np.random.default_rng(seed) if seed is not None else None
        )

    def __len__(self) -> int:
        """Return the number of batches per epoch."""
        n = len(self.X)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    @property
    def batches_per_epoch(self) -> int:
        """Alias for ``len(self)``: batches yielded per epoch."""
        return len(self)

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray | None]]:
        """Yield ``(xb, yb)`` batches as ndarrays (``yb`` is None if no ``y``)."""
        n = len(self.X)
        indices = np.arange(n)
        if self.shuffle:
            if self._rng is not None:
                indices = self._rng.permutation(n)
            else:
                indices = np.random.permutation(n)
        for start in range(0, n, self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            if self.drop_last and len(batch_idx) < self.batch_size:
                break
            xb = self.X[batch_idx]
            yb = self.y[batch_idx] if self.y is not None else None
            yield xb, yb
