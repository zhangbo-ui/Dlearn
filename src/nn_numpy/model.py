"""Sequential 容器：forward 正序、backward 逆序"""
import numpy as np

from .layers import Layer


class Sequential:
    def __init__(self, *layers: Layer):
        self.layers = list(layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """grad: dL/d模型输出。逆序传入每一层，返回 dL/d输入。"""
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
