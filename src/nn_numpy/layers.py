"""网络层：阶段2 只实现 forward；backward 在阶段3 补上"""
import numpy as np

from .init import he_init


class Layer:
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Linear(Layer):
    """全连接层：forward(x) = x @ W + b

    x: (N, in_features) -> 输出 (N, out_features)
    W: (in_features, out_features)   b: (out_features,)
    """

    def __init__(self, in_features: int, out_features: int, init_fn=he_init):
        self.in_features = in_features
        self.out_features = out_features
        self.W = init_fn(in_features, out_features)
        self.b = np.zeros(out_features, dtype=self.W.dtype)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.W + self.b


class ReLU(Layer):
    """逐元素 max(0, x)，无参数"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0)
