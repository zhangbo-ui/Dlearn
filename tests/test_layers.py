"""阶段2 单元测试：前向传播的形状与数值正确性"""
import numpy as np

from src.nn_numpy.init import he_init, zeros_init
from src.nn_numpy.layers import Linear, ReLU
from src.nn_numpy.model import Sequential


def test_linear_output_shape():
    lin = Linear(784, 256)
    x = np.random.randn(32, 784).astype(np.float32)
    assert lin.forward(x).shape == (32, 256)


def test_linear_computes_xW_plus_b():
    def arange_init(fan_in, fan_out, dtype=np.float32):
        return np.arange(fan_in * fan_out, dtype=dtype).reshape(fan_in, fan_out)

    lin = Linear(4, 3, init_fn=arange_init)
    x = np.arange(20, dtype=np.float32).reshape(5, 4)
    assert np.allclose(lin.forward(x), x @ lin.W + lin.b)


def test_linear_accepts_any_batch_size():
    lin = Linear(784, 10)
    for n in [1, 7, 64]:
        assert lin.forward(np.zeros((n, 784), dtype=np.float32)).shape == (n, 10)


def test_relu_zeros_negatives_only():
    x = np.array([[-3.0, -0.001, 0.0, 0.001, 5.0]], dtype=np.float32)
    out = ReLU().forward(x)
    assert np.allclose(out, [[0.0, 0.0, 0.0, 0.001, 5.0]])


def test_relu_preserves_shape():
    x = np.random.randn(8, 128).astype(np.float32)
    assert ReLU().forward(x).shape == (8, 128)


def test_he_init_shape_dtype_and_center():
    W = he_init(784, 256)
    assert W.shape == (784, 256)
    assert W.dtype == np.float32
    assert abs(W.mean()) < 0.01 * W.std()


def test_he_init_std_close_to_theory():
    fan_in = 2000
    W = he_init(fan_in, 2000)
    theory = np.sqrt(2.0 / fan_in)
    assert abs(W.std() - theory) / theory < 0.05


def test_zeros_init_is_all_zero():
    assert (zeros_init(4, 4) == 0).all()


def test_full_mlp_forward_shape():
    model = Sequential(
        Linear(784, 256), ReLU(),
        Linear(256, 128), ReLU(),
        Linear(128, 10),
    )
    x = np.random.randn(16, 784).astype(np.float32)
    assert model(x).shape == (16, 10)


def test_hidden_layers_change_shape_progressively():
    model = Sequential(Linear(784, 256), ReLU(), Linear(256, 128))
    x = np.random.randn(4, 784).astype(np.float32)
    h = model.layers[0].forward(x)
    assert h.shape == (4, 256)
    h = model.layers[1].forward(h)
    assert h.shape == (4, 256)
    assert model.layers[2].forward(h).shape == (4, 128)
