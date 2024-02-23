from typing import Tuple, Union

import numpy as np

from .base import Loss


class MSE(Loss):

    def __call__(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        return_upstream_grad: bool = False) -> Union[float, Tuple[float, np.ndarray]]:

        delta = y_pred - y_true
        loss = np.mean(delta ** 2)
        
        if not return_upstream_grad:
            return loss
        else:
            return loss, 2 * delta / np.prod(y_pred.shape)


class MAE(Loss):

    def __call__(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        return_upstream_grad: bool = False) -> Union[float, Tuple[float, np.ndarray]]:

        delta = y_pred - y_true
        loss = np.mean(np.abs(delta))

        if not return_upstream_grad:
            return loss
        else:
            grad = np.zeros_like(delta)
            
            grad[delta > 0] = +1
            grad[delta < 0] = -1

            return loss, grad / np.prod(y_pred.shape)

            