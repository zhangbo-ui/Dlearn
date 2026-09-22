"""优化器：拿梯度去更新参数（阶段4）

注意职责边界——反向传播只"生产"梯度（layer.dW/db），
优化器只"消费"梯度更新参数，二者互不越界。
"""
import numpy as np

from .layers import Linear


class SGD:
    """最朴素梯度下降：θ ← θ − lr·g"""

    def __init__(self, model, lr: float = 0.1):
        self.lr = lr
        self.layers = [l for l in model.layers if isinstance(l, Linear)]

    def step(self):
        for l in self.layers:
            if l.dW is None or l.db is None:
                continue  # 该层本步未被反向传播过，无梯度可更新
            l.W -= self.lr * l.dW
            l.b -= self.lr * l.db


class MomentumSGD(SGD):
    """带动量：v = μ·v − lr·g；θ ← θ + v

    直觉：v 是"速度"，梯度是"推力"，μ 是摩擦力。
    方向一致的梯度会不断累积速度（像下坡加速），
    方向反复横跳的梯度会互相抵消（震荡被抑制）——收敛更快更稳。
    """

    def __init__(self, model, lr: float = 0.1, momentum: float = 0.9):
        super().__init__(model, lr)
        self.momentum = momentum
        self.vW = [np.zeros_like(l.W) for l in self.layers]
        self.vb = [np.zeros_like(l.b) for l in self.layers]

    def step(self):
        for l, vW, vb in zip(self.layers, self.vW, self.vb):
            if l.dW is None or l.db is None:
                continue
            vW *= self.momentum
            vW -= self.lr * l.dW
            l.W += vW
            vb *= self.momentum
            vb -= self.lr * l.db
            l.b += vb
