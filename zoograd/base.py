from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Union

import numpy as np


@dataclass
class Parameter:
    """
    Simple wrapper of NumPy array with additional variables
    """

    value: np.ndarray
    differentiable: bool = True
    grad: np.ndarray = None

    def accumulate_grad(self, grad: np.ndarray) -> None:
        
        if not self.differentiable:
            return

        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad


class Module(ABC):

    """
    Base functional object, all layers and losses are derived from this class
    """

    Mode = Enum('Mode', ['TRAIN', 'INFERENCE'])
    
    mode = Mode.TRAIN

    @abstractmethod
    def forward(self, x: np.ndarray, *args) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, dy: np.ndarray, return_upstream_grad: bool = False) -> Union[None, np.ndarray]:
        pass

    def train(self):
        self._traverse_set_state(Module.Mode.TRAIN)

    def train(self):
        self._traverse_set_state(Module.Mode.INFERENCE)

    def parameters(self) -> List[Parameter]:
        parameters = []

        for v in vars(self).values():

            if isinstance(v, Parameter) and v.differentiable:
                parameters.append(v)

            if isinstance(v, Module):
                parameters += v.parameters()

        return parameters

    def _traverse_set_state(self, mode: Mode):
        self.mode = mode

        for v in vars(self).values():
            if isinstance(v, Module):
                v._traverse_set_state(mode)


class Loss(ABC):

    @abstractmethod
    def __call__(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        return_upstream_grad: bool = False,
        *args,
        **kwargs) -> Union[float, Tuple[float, np.ndarray]]:

        pass


class Optimizer(ABC):
    
    @abstractmethod
    def __init__(self, parameters: List[Parameter], *args, **kwargs):
        pass

    @abstractmethod
    def step(self, zero_grad: bool = True):
        pass

    def zero_grad(self):
        if self.parameters is not None:
            for p in self.parameters:
                if p.grad is not None:
                    p.grad = np.zeros_like(p.grad)


class Sequential(Module):

    def __init__(self, *args, **kwargs):
        self.modules = [m for m in args if isinstance(m, Module)]

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = x

        for m in self.modules:
            y = m.forward(y)
        
        return y
    
    def backward(self, upstream_grad: np.ndarray, return_upstream_grad: bool = False) -> Union[None, np.ndarray]:
        dx = upstream_grad

        for m in reversed(self.modules):
            dx = m.backward(dx)

        if return_upstream_grad:
            return dx

    def parameters(self) -> List[Parameter]:
        parameters = []

        for m in self.modules:
            parameters += m.parameters()

        return parameters
