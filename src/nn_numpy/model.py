"""Sequential 容器：按顺序执行各层 forward"""
import numpy as np

from .layers import Layer


class Sequential:
    def __init__(self, *layers: Layer):
        self.layers = list(layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
