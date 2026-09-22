"""Softmax + 交叉熵合并实现（数值稳定版）

合并的理由：复合后的梯度 dL/dlogits = softmax(logits) - one_hot(y)，
避免单独实现 Softmax 的 10x10 雅可比矩阵，又快又不容易错。
"""
import numpy as np


class SoftmaxCrossEntropy:
    def __init__(self):
        self.probs: np.ndarray | None = None   # 缓存 softmax 输出，backward 要用

    def forward(self, logits: np.ndarray, y: np.ndarray) -> float:
        """logits: (N, C) 任意实数；y: (N,) 0~9 整数标签（非 one-hot）

        返回标量损失（整个 batch 的平均交叉熵）。
        """
        # 1) 减 max：softmax 分子分母同乘 e^(-max) 结果不变，但 exp 永不溢出
        z = logits - logits.max(axis=1, keepdims=True)

        # 2) softmax（log 域计算更稳：先 log 再 exp，避免小概率下溢为 0 后取 log 变 -inf）
        log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True))

        # 3) 交叉熵：对每个样本取 -log(p[正确类])，再对 batch 求平均
        n = logits.shape[0]
        loss = -log_probs[np.arange(n), y].mean()

        # 缓存：backward 只需要 probs 和 y
        self.probs = np.exp(log_probs)
        self.y = y
        return float(loss)

    def backward(self) -> np.ndarray:
        """返回 dL/dlogits: (N, C)

        公式：dL/dz_i = p_i - 1[y_i == k]，
        除以 N 是因为 forward 里对 batch 取了平均。
        """
        n = self.probs.shape[0]
        grad = self.probs.copy()
        grad[np.arange(n), self.y] -= 1.0
        return grad / n

    def __call__(self, logits: np.ndarray, y: np.ndarray) -> float:
        return self.forward(logits, y)
