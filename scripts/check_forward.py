"""阶段2 验收：真实数据前向传播 + 全零初始化的对称性灾难演示"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_data
from src.nn_numpy.init import he_init, zeros_init
from src.nn_numpy.layers import Linear, ReLU
from src.nn_numpy.model import Sequential


def build_mlp(init_fn):
    return Sequential(
        Linear(784, 256, init_fn=init_fn), ReLU(),
        Linear(256, 128, init_fn=init_fn), ReLU(),
        Linear(128, 10, init_fn=init_fn),
    )


def main():
    data = load_data()
    x = data["x_train"][:128]
    y = data["y_train"][:128]

    print("===== 验收1：He 初始化，真实数据前向 =====")
    model = build_mlp(he_init)
    logits = model(x)
    print(f"输入  x: {x.shape}，范围 [{x.min():.3f}, {x.max():.3f}]")
    print(f"输出  logits: {logits.shape}，dtype={logits.dtype}")
    print(f"logits 均值 {logits.mean():.4f}，标准差 {logits.std():.4f}（数值健康，无爆炸/消失）")

    # 逐层跟踪形状与数值范围：理解"信号如何流过网络"
    print("\n逐层数据流（N=128）:")
    h = x
    for i, layer in enumerate(model.layers):
        h = layer.forward(h)
        print(f"  [{i}] {type(layer).__name__:6s} -> shape={h.shape}, "
              f"均值={h.mean():+.4f}, 标准差={h.std():.4f}")

    pred = logits.argmax(axis=1)
    acc = (pred == y).mean()
    print(f"\n未训练模型在 128 样本上的准确率: {acc:.1%}（应接近 10% 随机水平）")

    print("\n===== 验收2：全零初始化的对称性灾难 =====")
    zmodel = build_mlp(zeros_init)
    zlogits = zmodel(x)
    print(f"logits: {zlogits.shape}，值 = {zlogits[0]}")
    print("所有样本的 logits 完全相同（网络对输入无任何区分能力）")

    # 看隐藏层：所有神经元输出一模一样
    h1 = zmodel.layers[0].forward(x)
    print(f"第一隐藏层逐神经元标准差（跨样本）: 前5个 = {h1.std(axis=0)[:5]}")
    print("=> 每个神经元的输出和梯度永远相同，永远无法学到不同特征，这就是必须随机初始化的原因")

    ok = (
        logits.shape == (128, 10)
        and np.isfinite(logits).all()
        and abs(acc - 0.10) < 0.08
        and np.allclose(zlogits, zlogits[0])
    )
    print("\n阶段2 验收结果:", "全部通过 ✓" if ok else "存在未通过项 ✗")


if __name__ == "__main__":
    main()
