"""参数初始化：为什么不能全零，以及 He 初始化怎么来的"""
import numpy as np


def he_init(fan_in: int, fan_out: int, dtype=np.float32) -> np.ndarray:
    """He 初始化：W ~ N(0, sqrt(2/fan_in))，配合 ReLU 使用

    直觉：ReLU 会砍掉约一半的激活值，方差减半；
    乘 sqrt(2) 恰好把方差补回来，让信号逐层传递时不爆炸也不消失。
    """
    std = np.sqrt(2.0 / fan_in)
    return (np.random.randn(fan_in, fan_out) * std).astype(dtype)


def zeros_init(fan_in: int, fan_out: int, dtype=np.float32) -> np.ndarray:
    """全零初始化：仅用于观察"对称性灾难"，不要用于真实训练"""
    return np.zeros((fan_in, fan_out), dtype=dtype)
