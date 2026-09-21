"""阶段1 数据探索：形状/范围/分布检查 + 每类样本可视化，图保存到 notebooks/"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import CLASS_NAMES, load_data

OUT_DIR = Path(__file__).resolve().parents[1] / "notebooks"


def plot_samples(x: np.ndarray, y: np.ndarray, per_class: int = 8) -> None:
    fig, axes = plt.subplots(10, per_class, figsize=(per_class * 1.1, 10 * 1.1))
    for label in range(10):
        samples = x[y == label][:per_class]
        for i in range(per_class):
            ax = axes[label, i]
            ax.imshow(samples[i].reshape(28, 28), cmap="gray")
            ax.axis("off")
            if i == 0:
                ax.set_title(CLASS_NAMES[label], fontsize=8, loc="left")
    fig.suptitle("Fashion-MNIST samples per class", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_class_samples.png", dpi=120)
    print(f"已保存: {OUT_DIR / 'fig1_class_samples.png'}")


def plot_distribution(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    width = 0.25
    xs = np.arange(10)
    for off, (name, y) in enumerate(
        [("train", data["y_train"]), ("val", data["y_val"]), ("test", data["y_test"])]
    ):
        counts = np.bincount(y, minlength=10)
        ax.bar(xs + off * width, counts, width, label=f"{name} (N={len(y)})")
    ax.set_xticks(xs + width)
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("count")
    ax.set_title("Class distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_class_distribution.png", dpi=120)
    print(f"已保存: {OUT_DIR / 'fig2_class_distribution.png'}")


def check_acceptance(data: dict) -> None:
    print("\n===== 阶段1 验收检查 =====")
    ok = True
    for split in ["train", "val", "test"]:
        x, y = data[f"x_{split}"], data[f"y_{split}"]
        assert_ok = x.ndim == 2 and x.shape[1] == 784 and x.dtype == np.float32
        range_ok = x.min() >= 0.0 and x.max() <= 1.0
        label_ok = y.dtype == np.int64 and y.min() >= 0 and y.max() <= 9
        ok &= assert_ok and range_ok and label_ok
        print(
            f"{split:5s}: shape={x.shape} 范围[{x.min():.1f},{x.max():.1f}] "
            f"标签[{y.min()},{y.max()}] 形状{'✓' if assert_ok else '✗'} "
            f"范围{'✓' if range_ok else '✗'} 标签{'✓' if label_ok else '✗'}"
        )
    total = len(data["y_train"]) + len(data["y_val"])
    split_ok = total == 60000 and len(data["y_val"]) == 6000
    print(f"切分检查: train+val={total}（应为60000），val={len(data['y_val'])}（应为6000）"
          f" {'✓' if split_ok else '✗'}")
    ok &= split_ok
    print("验收结果:", "全部通过 ✓" if ok else "存在未通过项 ✗")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    data = load_data()
    print("数据加载成功:")
    for k, v in data.items():
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

    print("\n各类别数量（训练集）:")
    for label, count in enumerate(np.bincount(data["y_train"], minlength=10)):
        print(f"  {label} {CLASS_NAMES[label]:12s} {count}")

    plot_samples(data["x_train"], data["y_train"])
    plot_distribution(data)
    check_acceptance(data)


if __name__ == "__main__":
    main()
