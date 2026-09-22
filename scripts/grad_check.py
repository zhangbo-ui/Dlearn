"""数值梯度校验：用手写解析梯度对比数值差分梯度，相对误差 < 1e-6 才通过。

原理（导数定义的中心差分）:
    数值梯度 ≈ [f(θ+ε) - f(θ-ε)] / (2ε)
对小网络的每个参数逐一验证。用 float64 降低浮点噪声。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.nn_numpy.layers import Linear, ReLU
from src.nn_numpy.losses import SoftmaxCrossEntropy
from src.nn_numpy.model import Sequential

EPS = 1e-5
TOL = 1e-6


def make_tiny_model(seed: int) -> Sequential:
    """小网络足够验证公式，参数少跑得快（float64 降低浮点噪声）"""
    np.random.seed(seed)

    def init64(fan_in, fan_out, dtype=np.float64):
        return (np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)).astype(np.float64)

    return Sequential(
        Linear(7, 5, init_fn=init64), ReLU(),
        Linear(5, 4, init_fn=init64),
    )


def make_data(seed: int):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(6, 7)).astype(np.float64)
    y = rng.integers(0, 4, size=6)
    return x, y


def loss_of(model: Sequential, loss_fn, x, y) -> float:
    return loss_fn.forward(model.forward(x), y)


def rel_error(a: np.ndarray, b: np.ndarray) -> float:
    return np.abs(a - b).max() / (np.maximum(1e-8, np.abs(a).max() + np.abs(b).max()))


def check_layer_grads(model, x, y) -> list[tuple[str, float]]:
    results = []
    loss_fn = SoftmaxCrossEntropy()
    loss_fn.forward(model.forward(x), y)
    dlogits = loss_fn.backward()
    model.backward(dlogits)

    for li, layer in enumerate(model.layers):
        if not isinstance(layer, Linear):
            continue
        for pname, param, analytic in [
            ("W", layer.W, layer.dW),
            ("b", layer.b, layer.db),
        ]:
            numeric = np.zeros_like(param)
            it = np.nditer(param, flags=["multi_index"])
            while not it.finished:
                idx = it.multi_index
                orig = param[idx]
                param[idx] = orig + EPS
                lp = loss_of(model, loss_fn, x, y)
                param[idx] = orig - EPS
                lm = loss_of(model, loss_fn, x, y)
                param[idx] = orig
                numeric[idx] = (lp - lm) / (2 * EPS)
                it.iternext()
            err = rel_error(analytic, numeric)
            results.append((f"layers[{li}].{pname}", err))
    return results


def main():
    model = make_tiny_model(0)
    x, y = make_data(1)

    print(f"校验网络: 7->5->ReLU->4，batch=6，eps={EPS}，阈值={TOL}\n")
    results = check_layer_grads(model, x, y)

    all_ok = True
    print(f"{'参数':<18s} {'相对误差':>12s}  结果")
    for name, err in results:
        ok = err < TOL
        all_ok &= ok
        print(f"{name:<18s} {err:12.3e}  {'✓' if ok else '✗'}")

    print("\n数值梯度校验:", "全部通过 ✓" if all_ok else "存在未通过项 ✗ —— 不要进入训练！")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
