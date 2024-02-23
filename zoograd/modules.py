import numpy as np

from .base import Module, Parameter


class Linear(Module):

    """
    Simple linear layer `y(x) = W \cross x + b`
    """

    def __init__(self, in_dim: int, out_dim: int):

        self.s = None
        self.w = Parameter(np.random.normal(scale=np.sqrt(2 / in_dim), size=(in_dim, out_dim)))
        self.b = Parameter(np.zeros(out_dim))

    def forward(self, x):
        if self.mode == Module.Mode.TRAIN:
            self.s = x

        w, b = self.w.value, self.b.value

        return x @ w + b

    def backward(self, upstream_grad: np.ndarray, return_upstream_grad: bool = True):

        self.w.grad = self.s.T @ upstream_grad # / len(self.s)
        self.b.grad = np.sum(upstream_grad, axis=0)

        if return_upstream_grad:
            return upstream_grad @ self.w.value.T
