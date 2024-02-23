import numpy as np

from .base import Module


class Sigmoid(Module):

    def __init__(self):
        self.s = None

    def forward(self, x):
        s = 1 / (1 + np.exp(-x))

        if self.mode == Module.Mode.TRAIN:
            self.s = s

        return s

    def backward(self, upstream_grad: np.ndarray):
        return upstream_grad * self.s * (1 - self.s)


class ReLU(Module):

    def __init__(self):
        self.s = None

    def forward(self, x):
        s = np.maximum(0, x)

        if self.mode == Module.Mode.TRAIN:
            self.s = s > 0

        return s

    def backward(self, upstream_grad: np.ndarray):
        return upstream_grad * self.s
