"""阶段3 单元测试：损失函数与反向传播"""
import numpy as np

from src.nn_numpy.layers import Linear, ReLU
from src.nn_numpy.losses import SoftmaxCrossEntropy
from src.nn_numpy.model import Sequential


def _stable_softmax_reference(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def test_loss_zero_for_perfect_prediction():
    """预测概率 -> 1 时损失 -> 0"""
    loss_fn = SoftmaxCrossEntropy()
    logits = np.array([[100.0, 0.0, 0.0, 0.0]])
    y = np.array([0])
    assert loss_fn.forward(logits, y) < 1e-6


def test_loss_value_against_reference():
    """和朴素 softmax+交叉熵逐数值对比"""
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(8, 4))
    y = rng.integers(0, 4, size=8)
    probs = _stable_softmax_reference(logits.astype(np.float64))
    expected = -np.log(probs[np.arange(8), y]).mean()
    got = SoftmaxCrossEntropy().forward(logits.astype(np.float64), y)
    assert abs(got - expected) < 1e-10


def test_loss_numerically_stable_with_huge_logits():
    """logits 很大也不得溢出/出 NaN"""
    loss_fn = SoftmaxCrossEntropy()
    logits = np.array([[1000.0, -1000.0, 0.0], [-2000.0, 3000.0, 0.0]])
    y = np.array([0, 1])
    loss = loss_fn.forward(logits, y)
    assert np.isfinite(loss) and loss >= 0.0


def test_softmax_rows_sum_to_one():
    loss_fn = SoftmaxCrossEntropy()
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(5, 10))
    loss_fn.forward(logits, np.zeros(5, dtype=int))
    assert np.allclose(loss_fn.probs.sum(axis=1), 1.0)


def test_gradient_of_loss_matches_softmax_minus_onehot():
    """核心公式：dL/dlogits = (softmax - onehot)/N"""
    rng = np.random.default_rng(2)
    logits = rng.normal(size=(6, 4))
    y = rng.integers(0, 4, size=6)
    loss_fn = SoftmaxCrossEntropy()
    loss_fn.forward(logits, y)
    grad = loss_fn.backward()
    onehot = np.eye(4)[y]
    assert np.allclose(grad, (softmax_ref(logits) - onehot) / 6)


def softmax_ref(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def test_relu_backward_zero_on_negatives():
    relu = ReLU()
    x = np.array([[-2.0, 3.0], [0.0, -1.0]])
    relu.forward(x)
    grad = np.full_like(x, 5.0)
    assert np.allclose(relu.backward(grad), [[0.0, 5.0], [0.0, 0.0]])


def test_linear_backward_shapes():
    lin = Linear(7, 4)
    x = np.random.randn(6, 7)
    lin.forward(x)
    grad = np.random.randn(6, 4)
    dx = lin.backward(grad)
    assert lin.dW.shape == lin.W.shape == (7, 4)
    assert lin.db.shape == lin.b.shape == (4,)
    assert dx.shape == x.shape


def test_linear_backward_formulas_exact():
    """用可控小数据逐元素核对 dW/db/dx 公式"""
    lin = Linear(3, 2, init_fn=lambda i, o, dtype=np.float64: np.array(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    lin.b = np.array([0.1, -0.1])
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    lin.forward(x)
    grad = np.array([[1.0, 0.0], [0.0, 1.0]])
    dx = lin.backward(grad)
    assert np.allclose(lin.dW, x.T @ grad)
    assert np.allclose(lin.db, grad.sum(axis=0))
    assert np.allclose(dx, grad @ lin.W.T)


def test_sequential_backward_returns_input_grad():
    model = Sequential(Linear(7, 5), ReLU(), Linear(5, 4))
    x = np.random.randn(6, 7)
    model.forward(x)
    dx = model.backward(np.random.randn(6, 4))
    assert dx.shape == (6, 7)
