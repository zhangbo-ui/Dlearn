"""阶段4 单元测试：优化器更新公式与收敛行为"""
import numpy as np

from src.nn_numpy.layers import Linear, ReLU
from src.nn_numpy.model import Sequential
from src.nn_numpy.optim import SGD, MomentumSGD


def _model():
    return Sequential(Linear(5, 3), ReLU(), Linear(3, 2))


def test_sgd_exact_update_formula():
    m = _model()
    opt = SGD(m, lr=0.1)
    lin = m.layers[0]
    W0, b0 = lin.W.copy(), lin.b.copy()
    dW, db = np.ones_like(lin.W), np.ones_like(lin.b)
    lin.dW, lin.db = dW, db
    opt.step()
    assert np.allclose(lin.W, W0 - 0.1 * dW)
    assert np.allclose(lin.b, b0 - 0.1 * db)


def test_sgd_descends_quadratic():
    """梯度恒等于参数本身（f = 0.5||θ||²），SGD 应把参数拉向 0"""
    m = _model()
    opt = SGD(m, lr=0.1)
    lin = m.layers[0]
    w0 = np.abs(lin.W).max()
    for _ in range(200):
        lin.dW, lin.db = lin.W.copy(), lin.b.copy()
        opt.step()
    # 200 步后应按 0.9^200 几何收敛到远小于初值
    assert np.abs(lin.W).max() < w0 * 1e-3


def test_sgd_skips_relu_layers():
    """ReLU 无参数，不应被更新也不应报错"""
    m = _model()
    opt = SGD(m, lr=0.1)
    assert len(opt.layers) == 2
    opt.step()


def test_momentum_exact_update_formula():
    m = _model()
    opt = MomentumSGD(m, lr=0.1, momentum=0.9)
    lin = m.layers[0]
    W0 = lin.W.copy()
    g = np.ones_like(lin.W)
    lin.dW, lin.db = g.copy(), np.ones_like(lin.b)
    opt.step()
    assert np.allclose(lin.W, W0 - 0.1 * g)  # 第一步速度 = -lr·g（初速为0）
    lin.dW = g.copy()
    opt.step()
    v2 = 0.9 * (-0.1 * g) - 0.1 * g          # v = μ·v − lr·g
    assert np.allclose(lin.W, W0 - 0.1 * g + v2)


def test_momentum_accelerates_on_constant_gradient():
    """梯度恒定时，动量版单步位移应大于纯 SGD（速度累积效应）"""
    m1, m2 = _model(), _model()
    m2.layers[0].W = m1.layers[0].W.copy()
    m2.layers[0].b = m1.layers[0].b.copy()
    o1, o2 = SGD(m1, lr=0.1), MomentumSGD(m2, lr=0.1, momentum=0.9)
    for _ in range(10):
        for m_, o_ in [(m1, o1), (m2, o2)]:
            lin = m_.layers[0]
            lin.dW, lin.db = np.ones_like(lin.W), np.ones_like(lin.b)
            o_.step()
    d1 = np.abs(m1.layers[0].W).max()
    d2 = np.abs(m2.layers[0].W).max()
    assert d2 > d1 * 1.5
