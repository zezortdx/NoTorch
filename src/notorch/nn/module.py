"""Base neural-network module and sequential container.

Defines :class:`Module`, the base class for all NoTorch layers, activations
and losses, plus :class:`Sequential`, an ordered container of modules.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from ..tensor import Tensor
from .parameter import Parameter


def _named_parameters_of(value: Any, prefix: str) -> Iterator[tuple[str, Tensor]]:
    """Yield ``(name, parameter)`` pairs found under ``value``."""
    if isinstance(value, Module):
        yield from value.named_parameters(prefix=prefix)
    elif isinstance(value, Parameter) or (
        isinstance(value, Tensor) and bool(getattr(value, "requires_grad", False))
    ):
        yield (prefix, value)
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            yield from _named_parameters_of(item, f"{prefix}.{i}")
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _named_parameters_of(item, f"{prefix}.{key}")


def _modules_of(value: Any) -> Iterator[Module]:
    """Yield submodules found under ``value``."""
    if isinstance(value, Module):
        yield from value.modules()
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _modules_of(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _modules_of(item)


def _children_of(value: Any, prefix: str) -> Iterator[tuple[str, Module]]:
    """Yield ``(name, submodule)`` pairs found directly under ``value``."""
    if isinstance(value, Module):
        yield (prefix, value)
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            yield from _children_of(item, f"{prefix}.{i}")
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _children_of(item, f"{prefix}.{key}")


class Module:
    """Base class for all neural network modules.

    Subclasses implement :meth:`forward`; calling a module instance dispatches
    to :meth:`forward`. Trainable tensors should be stored as
    :class:`~notorch.nn.Parameter` attributes (directly or nested inside
    ``Module`` attributes, lists, tuples or dicts) so that :meth:`parameters`,
    :meth:`state_dict` and friends discover them recursively.

    Subclasses must call ``super().__init__()`` to initialise the
    :attr:`training` flag.
    """

    def __init__(self) -> None:
        self.training: bool = True

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Compute the module output. Must be overridden by subclasses."""
        raise NotImplementedError(f"{type(self).__name__} must implement forward()")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Tensor]]:
        """Yield ``(name, parameter)`` pairs, recursing into submodules.

        Dotted names reflect attribute paths (e.g. ``"layer.W"``); entries
        inside lists/tuples/dicts use their index/key (e.g. ``"layers.0.W"``).

        Args:
            prefix: Prefix prepended to every name (used for recursion).
        """
        for attr, value in self.__dict__.items():
            if attr == "training" or value is None:
                continue
            name = f"{prefix}.{attr}" if prefix else attr
            yield from _named_parameters_of(value, name)

    def parameters(self) -> list[Tensor]:
        """Return all trainable parameters, recursing into submodules."""
        return [param for _, param in self.named_parameters()]

    def named_children(self) -> Iterator[tuple[str, Module]]:
        """Yield ``(name, submodule)`` pairs for direct child modules."""
        for attr, value in self.__dict__.items():
            if attr == "training" or value is None:
                continue
            yield from _children_of(value, attr)

    def modules(self) -> Iterator[Module]:
        """Yield this module and all submodules recursively (self first)."""
        yield self
        for attr, value in self.__dict__.items():
            if attr == "training" or value is None:
                continue
            yield from _modules_of(value)

    def zero_grad(self) -> None:
        """Zero the gradients of all parameters."""
        for param in self.parameters():
            param.zero_grad()

    def train(self, mode: bool = True) -> Module:
        """Set training mode, recursing into submodules.

        Args:
            mode: ``True`` for training mode, ``False`` for evaluation mode.

        Returns:
            This module, to allow chaining.
        """
        for module in self.modules():
            module.training = mode
        return self

    def eval(self) -> Module:
        """Set evaluation mode (``training=False``), recursing into submodules."""
        return self.train(False)

    def state_dict(self) -> dict[str, np.ndarray]:
        """Return a snapshot of all parameters as ``{name: np.ndarray}``."""
        return {name: np.array(param.data, copy=True) for name, param in self.named_parameters()}

    def load_state_dict(self, state_dict: dict[str, np.ndarray], strict: bool = True) -> None:
        """Copy parameter values from ``state_dict`` into this module.

        Args:
            state_dict: Mapping of parameter names to arrays, as produced by
                :meth:`state_dict`.
            strict: If ``True``, raise on missing/unexpected keys. Shape
                mismatches always raise.

        Raises:
            RuntimeError: On missing or unexpected keys when ``strict=True``.
            ValueError: On shape mismatches.
        """
        current = dict(self.named_parameters())
        if strict:
            missing = [key for key in current if key not in state_dict]
            if missing:
                raise RuntimeError(f"load_state_dict: missing keys: {missing}")
            unexpected = [key for key in state_dict if key not in current]
            if unexpected:
                raise RuntimeError(f"load_state_dict: unexpected keys: {unexpected}")
        for name, param in current.items():
            if name not in state_dict:
                continue
            value = np.asarray(state_dict[name])
            if value.shape != tuple(param.data.shape):
                raise ValueError(
                    f"load_state_dict: shape mismatch for {name!r}: "
                    f"expected {tuple(param.data.shape)}, got {value.shape}"
                )
            np.copyto(param.data, value.astype(param.data.dtype, copy=False))

    def extra_repr(self) -> str:
        """Extra information for :meth:`__repr__`; subclasses may override."""
        return ""

    def __repr__(self) -> str:
        name = type(self).__name__
        extra = self.extra_repr()
        children = list(self.named_children())
        if not children:
            return f"{name}({extra})" if extra else f"{name}()"
        lines = [f"{name}("]
        if extra:
            lines.append(f"  {extra}")
        for key, child in children:
            lines.append(f"  ({key}): {repr(child).replace(chr(10), chr(10) + '  ')}")
        lines.append(")")
        return "\n".join(lines)


class Sequential(Module):
    """Ordered container applying modules in sequence.

    Args:
        *modules: Child modules, or a single list/tuple of modules.

    Example:
        >>> mlp = Sequential(Linear(4, 8), ReLU(), Linear(8, 1))
        >>> out = mlp(x)
    """

    def __init__(self, *modules: Any) -> None:
        super().__init__()
        if len(modules) == 1 and isinstance(modules[0], (list, tuple)):
            modules = tuple(modules[0])
        for module in modules:
            if not isinstance(module, Module):
                raise TypeError(
                    f"Sequential expects Module instances, got {type(module).__name__!r}"
                )
        self._modules: list[Module] = list(modules)

    def forward(self, x: Tensor) -> Tensor:
        """Apply each child module in order and return the final output."""
        for module in self._modules:
            x = module(x)
        return x

    def append(self, module: Module) -> Sequential:
        """Append a child module and return this container for chaining."""
        if not isinstance(module, Module):
            raise TypeError(f"Sequential expects a Module, got {type(module).__name__!r}")
        self._modules.append(module)
        return self

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Tensor]]:
        """Yield ``(name, parameter)`` pairs with positional names (``"0.W"``)."""
        for i, module in enumerate(self._modules):
            name = str(i) if not prefix else f"{prefix}.{i}"
            yield from module.named_parameters(prefix=name)

    def named_children(self) -> Iterator[tuple[str, Module]]:
        """Yield ``(index, submodule)`` pairs for direct children."""
        for i, module in enumerate(self._modules):
            yield (str(i), module)

    def __len__(self) -> int:
        return len(self._modules)

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice):
            return Sequential(self._modules[index])
        return self._modules[index]

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules)
