"""下载 Fashion-MNIST 原始数据文件（IDX 格式）到 data/raw/"""
import urllib.request
from pathlib import Path

FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]

MIRRORS = [
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/",
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
]


def main() -> None:
    raw_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for name in FILES:
        dest = raw_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"跳过（已存在）: {dest}")
            continue
        for base in MIRRORS:
            url = base + name
            try:
                print(f"下载 {url} ...")
                urllib.request.urlretrieve(url, dest)
                print(f"  -> {dest}（{dest.stat().st_size / 1e6:.1f} MB）")
                break
            except Exception as e:
                print(f"  失败: {e}")
        else:
            raise RuntimeError(f"所有镜像均下载失败: {name}")


if __name__ == "__main__":
    main()
