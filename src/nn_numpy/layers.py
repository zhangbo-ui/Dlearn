"""网络层：forward + backward（阶段3 完成）"""
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

    backward 时需要前向的输入 x —— 所以 forward 必须缓存它。
    """

    def __init__(self, in_features: int, out_features: int, init_fn=he_init):
        self.in_features = in_features
        self.out_features = out_features
        self.W = init_fn(in_features, out_features)
        self.b = np.zeros(out_features, dtype=self.W.dtype)
        self.dW: np.ndarray | None = None
        self.db: np.ndarray | None = None
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x                       # 缓存输入，反向传播要用
        return x @ self.W + self.b

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """grad: dL/d输出，形状 (N, out_features)

        三个公式（形状推理可当验算器）:
          dW = xᵀ @ grad          (in,N)@(N,out) -> (in,out) 与 W 同形 ✓
          db = grad 按样本维求和    (N,out) -> (out)        与 b 同形 ✓
          dx = grad @ Wᵀ           (N,out)@(out,in) -> (N,in) 与 x 同形 ✓
        """
        self.dW = self._x.T @ grad
        self.db = grad.sum(axis=0)
        return grad @ self.W.T


class ReLU(Layer):
    """逐元素 max(0, x)，无参数

    反向：前向为正的位置梯度原样通过，为负的位置梯度归零。
    """

    def __init__(self):
        self._mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = x > 0                # 缓存布尔 mask，反向要用
        return np.maximum(x, 0.0)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self._mask
