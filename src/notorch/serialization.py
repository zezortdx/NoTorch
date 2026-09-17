"""Model-agnostic serialization helpers (numpy-only, no pickle).

State dicts are plain ``{str: np.ndarray}`` mappings saved with
:func:`numpy.savez`. Array keys are sanitized by storing arrays under
positional ``arr_{i}`` entries plus a ``__keys__`` table of original names,
so arbitrary string keys (``"layer.weight"``, ``"a/b"``) round-trip safely.
Deserialization uses ``allow_pickle=False`` throughout.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Union

import numpy as np

PathLike = Union[str, Path]

_KEYS_ENTRY = "__keys__"


def _as_array_dict(state: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Validate and convert a state mapping to ``{str: np.ndarray}``."""
    if not isinstance(state, Mapping):
        raise TypeError("state_dict must be a mapping of str -> np.ndarray.")
    out: dict[str, np.ndarray] = {}
    for key, value in state.items():
        if not isinstance(key, str) or key == "":
            raise TypeError("state_dict keys must be non-empty strings.")
        if key == _KEYS_ENTRY:
            raise ValueError(f"state_dict key {key!r} is reserved.")
        if isinstance(value, np.ndarray):
            arr = value
        elif isinstance(value, (list, tuple, int, float, complex, bool)):
            arr = np.asarray(value)
        else:
            raise TypeError(
                f"state_dict[{key!r}] must be np.ndarray or array-like, got {type(value).__name__}."
            )
        if arr.dtype == object:
            raise TypeError(f"state_dict[{key!r}] must not have object dtype.")
        out[key] = np.ascontiguousarray(arr)
    return out


def _resolve_path(path: PathLike) -> Path:
    """Return ``path`` with parent dirs created and a ``.npz`` suffix."""
    p = Path(path)
    if p.suffix != ".npz":
        p = p.with_suffix(p.suffix + ".npz") if p.suffix else p.with_suffix(".npz")
    if p.parent != Path(""):
        p.parent.mkdir(parents=True, exist_ok=True)
    return p


def get_state_dict(module: Any) -> dict[str, np.ndarray]:
    """Return the state dict of a module.

    Args:
        module: Object exposing ``state_dict()``.

    Returns:
        Shallow copy of the module's state dict.

    Raises:
        TypeError: If the module has no ``state_dict`` method.
    """
    fn = getattr(module, "state_dict", None)
    if not callable(fn):
        raise TypeError("module must expose a state_dict() method.")
    state = fn()
    if not isinstance(state, Mapping):
        raise TypeError("module.state_dict() must return a mapping.")
    return dict(state)


def load_state_dict(module: Any, state_dict: Mapping[str, Any], strict: bool = True) -> Any:
    """Load a state dict into a module.

    Delegates to ``module.load_state_dict`` when available (passing ``strict``
    when the method accepts it); otherwise assigns attributes entry-wise.

    Args:
        module: Target module or dict.
        state_dict: Mapping of ``str -> np.ndarray``.
        strict: If True, raise on missing/unexpected keys for plain-object
            and dict targets.

    Returns:
        The module.
    """
    arrays = _as_array_dict(state_dict)
    loader = getattr(module, "load_state_dict", None)
    if callable(loader):
        try:
            loader(arrays, strict=strict)
        except TypeError:
            loader(arrays)
        return module
    if isinstance(module, dict):
        if strict:
            unexpected = set(arrays) - set(module.keys())
            missing = set(module.keys()) - set(arrays)
            if unexpected or missing:
                raise KeyError(f"unexpected={sorted(unexpected)} missing={sorted(missing)}")
        for key, arr in arrays.items():
            module[key] = arr.copy()
        return module
    if strict:
        missing = [k for k in arrays if not hasattr(module, k)]
        if missing:
            raise KeyError(f"module is missing attributes: {missing}")
    for key, arr in arrays.items():
        current = getattr(module, key, None)
        if isinstance(current, np.ndarray) and current.shape == arr.shape:
            current[...] = arr
        else:
            setattr(module, key, arr.copy())
    return module


def save(obj_or_module: Any, path: PathLike) -> str:
    """Save a state dict (or state-dict-owning module) to ``path`` via np.savez.

    Args:
        obj_or_module: Mapping of ``str -> np.ndarray`` or object exposing
            ``state_dict()``.
        path: Destination file path (``.npz`` appended if missing).

    Returns:
        The resolved file path as a string.
    """
    if isinstance(obj_or_module, Mapping):
        arrays = _as_array_dict(obj_or_module)
    else:
        arrays = _as_array_dict(get_state_dict(obj_or_module))
    target = _resolve_path(path)
    keys = np.asarray(list(arrays.keys()), dtype=str)
    payload: dict[str, np.ndarray] = {f"arr_{i}": arrays[k] for i, k in enumerate(keys)}
    payload[_KEYS_ENTRY] = keys
    np.savez(target, **payload)
    return str(target)


def load(module_or_none: Any | None, path: PathLike) -> Any:
    """Load a state dict from ``path``.

    Args:
        module_or_none: If None, return the state dict. Otherwise load it
            into this module (via :func:`load_state_dict`) and return it.
        path: Source ``.npz`` file (suffix optional).

    Returns:
        The state dict, or the module when one was given.
    """
    candidate = Path(path)
    if not candidate.exists() and candidate.suffix != ".npz":
        candidate = candidate.with_suffix(candidate.suffix + ".npz")
    with np.load(candidate, allow_pickle=False) as archive:
        if _KEYS_ENTRY not in archive:
            raise ValueError("Not a NoTorch state file: missing key table.")
        keys = [str(k) for k in archive[_KEYS_ENTRY].tolist()]
        state = {k: np.array(archive[f"arr_{i}"]) for i, k in enumerate(keys)}
    if module_or_none is None:
        return state
    return load_state_dict(module_or_none, state)
