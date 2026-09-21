"""Fashion-MNIST 数据加载：IDX 二进制解析、归一化、训练/验证切分。

IDX 文件格式（大端字节序）:
  图像文件: magic(4B)=2051 | num(4B) | rows(4B) | cols(4B) | 像素字节流(uint8)
  标签文件: magic(4B)=2049 | num(4B) | 标签字节流(uint8)
"""
import gzip
import struct
from pathlib import Path

import numpy as np

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def _read_idx_images(path: Path) -> np.ndarray:
    """读 IDX 图像文件，返回 (N, rows*cols) 的 uint8 数组"""
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"不是 IDX 图像文件: {path}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
    assert data.size == n * rows * cols, f"数据长度不符: {data.size} != {n}*{rows}*{cols}"
    return data.reshape(n, rows * cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    """读 IDX 标签文件，返回 (N,) 的 uint8 数组"""
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"不是 IDX 标签文件: {path}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
    assert data.size == n, f"数据长度不符: {data.size} != {n}"
    return data


def load_data(
    data_dir: str = "data",
    val_ratio: float = 0.1,
    seed: int = 42,
    dtype=np.float32,
) -> dict:
    """加载全部数据并切分。

    返回 dict:
      x_train (54000, 784) dtype / y_train (54000,) int64
      x_val    (6000, 784)  / y_val   (6000,)
      x_test   (10000, 784) / y_test  (10000,)
    像素归一化到 [0,1]；标签为 0~9 整数（非 one-hot）。
    """
    root = Path(data_dir) / "raw"
    x_train_full = _read_idx_images(root / "train-images-idx3-ubyte.gz").astype(dtype) / 255.0
    y_train_full = _read_idx_labels(root / "train-labels-idx1-ubyte.gz").astype(np.int64)
    x_test = _read_idx_images(root / "t10k-images-idx3-ubyte.gz").astype(dtype) / 255.0
    y_test = _read_idx_labels(root / "t10k-labels-idx1-ubyte.gz").astype(np.int64)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x_train_full))
    n_val = int(len(x_train_full) * val_ratio)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    return {
        "x_train": x_train_full[tr_idx], "y_train": y_train_full[tr_idx],
        "x_val": x_train_full[val_idx], "y_val": y_train_full[val_idx],
        "x_test": x_test, "y_test": y_test,
    }


if __name__ == "__main__":
    data = load_data()
    for k, v in data.items():
        print(f"{k}: shape={v.shape}, dtype={v.dtype}, min={v.min():.3f}, max={v.max():.3f}")
