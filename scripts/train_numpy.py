"""阶段4 训练入口：完整训练循环 + 每 epoch 实时刷新训练曲线图

用法：
  python scripts/train_numpy.py                 # 默认：MomentumSGD, lr=0.1, batch=64, 30 epochs
  python scripts/train_numpy.py --optimizer sgd --lr 0.5 --epochs 10
每轮 epoch 结束把 loss/acc 曲线写到 notebooks/fig3_training_curves.png（打开后刷新即可看实时进度），
最优模型参数存到 data/best_model.npz，训练历史存到 data/history.npz。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_data
from src.nn_numpy.layers import Linear, ReLU
from src.nn_numpy.losses import SoftmaxCrossEntropy
from src.nn_numpy.model import Sequential
from src.nn_numpy.optim import SGD, MomentumSGD

ROOT = Path(__file__).resolve().parents[1]
OUT_FIG = ROOT / "notebooks" / "fig3_training_curves.png"


def build_model() -> Sequential:
    return Sequential(
        Linear(784, 256), ReLU(),
        Linear(256, 128), ReLU(),
        Linear(128, 10),
    )


def evaluate(model, x, y, batch: int = 1024) -> tuple[float, float]:
    loss_fn = SoftmaxCrossEntropy()
    total_loss, correct = 0.0, 0
    for i in range(0, len(x), batch):
        xb, yb = x[i:i + batch], y[i:i + batch]
        logits = model(xb)
        total_loss += loss_fn.forward(logits, yb) * len(xb)
        correct += (logits.argmax(axis=1) == yb).sum()
    return total_loss / len(x), correct / len(x)


def live_plot(hist: dict, args) -> None:
    epochs = range(1, len(hist["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, hist["train_loss"], "o-", ms=3, label="train")
    axes[0].plot(epochs, hist["val_loss"], "s-", ms=3, label="val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].set_title(f"Loss  (lr={args.lr}, batch={args.batch_size}, {args.optimizer})")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, hist["train_acc"], "o-", ms=3, label="train")
    axes[1].plot(epochs, hist["val_acc"], "s-", ms=3, label="val")
    axes[1].axhline(0.85, color="r", ls="--", lw=1, label="85% target")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.suptitle(f"epoch {len(epochs)}/{args.epochs}  best val acc = {max(hist['val_acc']):.2%}")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimizer", choices=["sgd", "momentum"], default="momentum")
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    data = load_data()
    model = build_model()
    loss_fn = SoftmaxCrossEntropy()
    opt = (SGD if args.optimizer == "sgd" else MomentumSGD)(model, lr=args.lr, momentum=0.9)

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc, best_params = 0.0, None
    n = len(data["x_train"])

    print(f"配置: {args.optimizer} lr={args.lr} batch={args.batch_size} epochs={args.epochs}")
    print(f"{'epoch':>5} {'train_loss':>10} {'val_loss':>9} {'train_acc':>9} {'val_acc':>8} {'time':>6}")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        perm = np.random.permutation(n)
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb, yb = data["x_train"][idx], data["y_train"][idx]
            loss_fn.forward(model(xb), yb)   # 1. 前向
            model.backward(loss_fn.backward())  # 2. 反向
            opt.step()                        # 3. 更新

        tr_loss, tr_acc = evaluate(model, data["x_train"][:10000], data["y_train"][:10000])
        va_loss, va_acc = evaluate(model, data["x_val"], data["y_val"])
        hist["train_loss"].append(tr_loss); hist["val_loss"].append(va_loss)
        hist["train_acc"].append(tr_acc); hist["val_acc"].append(va_acc)
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_params = {
                f"W{i}": l.W.copy() for i, l in enumerate(model.layers) if hasattr(l, "W")
            } | {f"b{i}": l.b.copy() for i, l in enumerate(model.layers) if hasattr(l, "b")}

        live_plot(hist, args)
        print(f"{epoch:>5} {tr_loss:>10.4f} {va_loss:>9.4f} {tr_acc:>9.2%} {va_acc:>8.2%} {time.time()-t0:>5.1f}s")

    np.savez(ROOT / "data" / "best_model.npz", **best_params)
    np.savez(ROOT / "data" / "history.npz", **{k: np.array(v) for k, v in hist.items()})
    (ROOT / "data" / "train_config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2))
    print(f"\n最优验证准确率: {best_val_acc:.2%}（验收线 85%）")
    print(f"曲线图: {OUT_FIG}\n参数与历史已保存到 data/")


if __name__ == "__main__":
    main()
