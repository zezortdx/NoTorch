"""NoTorch neural-network module: parameters, modules, layers, activations, losses."""

from . import init, losses
from .activations import ReLU, Sigmoid, Tanh
from .layers import Flatten, Linear
from .losses import BinaryCrossEntropyLoss, CrossEntropyLoss, MSELoss
from .module import Module, Sequential
from .parameter import Parameter

BCELoss = BinaryCrossEntropyLoss
BCEWithLogitsLoss = BinaryCrossEntropyLoss
CELoss = CrossEntropyLoss

__all__ = [
    "BCELoss",
    "BCEWithLogitsLoss",
    "BinaryCrossEntropyLoss",
    "CELoss",
    "CrossEntropyLoss",
    "Flatten",
    "Linear",
    "MSELoss",
    "Module",
    "Parameter",
    "ReLU",
    "Sequential",
    "Sigmoid",
    "Tanh",
    "init",
    "losses",
]
