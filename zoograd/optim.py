from typing import List

import numpy as np

from .base import Optimizer, Parameter


class SGD(Optimizer):

    def __init__(self, parameters: List[Parameter], lr: float = 0.001):

        self.parameters = parameters
        self.lr = lr

    def step(self, zero_grad: bool = True):

        for i, p in enumerate(self.parameters):

            if not p.differentiable:
                continue

            p.value -= self.lr * p.grad

        if zero_grad:
            self.zero_grad()


class Adam(Optimizer):

    def __init__(
        self,
        parameters: List[Parameter],
        alpha: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        eps: float = 1e-8
    ):

        self.parameters = parameters
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps

        self.t = 0
        self.m = [np.zeros_like(p.value) for p in parameters]
        self.v = [np.zeros_like(p.value) for p in parameters]

    def step(self, zero_grad: bool = True):
        
        self.t += 1

        for i, p in enumerate(self.parameters):

            if not p.differentiable:
                continue
            
            self.m[i] = self.beta_1 * self.m[i] + (1 - self.beta_1) * p.grad
            self.v[i] = self.beta_2 * self.v[i] + (1 - self.beta_2) * p.grad ** 2

            m_hat = self.m[i] / (1 - self.beta_1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta_2 ** self.t)

            p.value -= self.alpha * m_hat / (v_hat ** 0.5 + self.eps)

        if zero_grad:
            self.zero_grad()
