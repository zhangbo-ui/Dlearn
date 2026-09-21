# 第一个神经网络：从零搭建学习计划

> 目标：不只"跑通"一个神经网络，而是**亲手实现并真正理解**其中每一环节。
> 数据集：Fashion-MNIST（28×28 灰度图，10 类服饰，约 6 万训练样本）。
> 总时长：约 3~4 周（每天 1~2 小时），可按自身节奏伸缩。

---

## 一、学习目标（做完后你应能回答）

1. 前向传播中每一层的数据形状和数值是如何变化的？
2. 反向传播的梯度从损失函数出发是如何逐层传回每个权重的？（链式法则）
3. 损失函数为什么选交叉熵？它和 Softmax 是怎么配套的？
4. 学习率、batch size、初始化分别如何影响训练？出问题时如何诊断？
5. PyTorch 的 `nn.Linear` / `optimizer.step()` / `loss.backward()` 背后替你做了什么？

**总体验收**：用 NumPy 从零手写的 MLP 在 Fashion-MNIST 测试集达到 **≥ 85% 准确率**，且通过数值梯度校验；再用 PyTorch 复现同一结构并对照结果一致。

---

## 二、技术路线：两条实现线

| | 线路 A：NumPy 手写（核心） | 线路 B：PyTorch 对照 |
|---|---|---|
| 目的 | 理解原理 | 理解框架、为后续打基础 |
| 模型 | MLP: 784→256→128→10 | 与 A 完全相同 |
| 内容 | 手写前向、反向、SGD/Momentum、mini-batch 训练循环 | 用 `nn.Module` + `autograd` 复现 |
| 验收 | 梯度校验 + ≥85% 准确率 | 训练曲线与 A 基本一致 |

先 A 后 B。A 是理解的关键，B 让你知道框架"黑盒"里装的是什么。

---

## 三、环境准备（阶段 0，半天）

- [x] Python ≥ 3.10，建议新建虚拟环境：`python -m venv .venv && source .venv/bin/activate`
- [x] 安装依赖并写入 `requirements.txt`：`numpy`、`matplotlib`、`torch`（CPU 版即可）、`torchvision`、`pytest`
- [x] 确认 `python -c "import numpy, torch; print(numpy.__version__, torch.__version__)"` 正常
- [x] `git init`，配好 `.gitignore`（忽略 `data/`、`.venv/`、`__pycache__/`）

---

## 四、项目结构（预先规划，边做边填充）

```
Dlearn/
├── PLAN.md                 # 本文档
├── requirements.txt
├── data/                   # 数据集（gitignore）
├── src/
│   ├── nn_numpy/           # 线路 A：手写迷你框架
│   │   ├── layers.py       # Linear、ReLU
│   │   ├── losses.py       # Softmax + 交叉熵（合并数值稳定实现）
│   │   ├── optim.py        # SGD、SGD+Momentum
│   │   ├── model.py        # Sequential 容器（forward/backward）
│   │   └── init.py         # He 初始化
│   └── data.py             # 加载/归一化/划分 Fashion-MNIST
├── scripts/
│   ├── download_data.py
│   ├── grad_check.py       # 数值梯度校验
│   ├── train_numpy.py      # 线路 A 训练入口
│   └── train_torch.py      # 线路 B 训练入口
├── tests/
│   └── test_layers.py      # 各组件单元测试
├── notebooks/              # 可视化与实验分析
└── experiments/            # 实验记录（每实验一个 md + 图）
```

---

## 五、分阶段计划

### 阶段 1：数据（1~2 天）

- [ ] 下载 Fashion-MNIST（`torchvision.datasets` 或直接下载 IDX 原始文件，用 NumPy 解析更佳）
- [ ] `src/data.py`：返回形状为 `(N, 784)` 的 float 数组，归一化到 `[0,1]`；从训练集切出 10% 作验证集
- [ ] Notebook 可视化：每类抽几张图看看；打印类别分布确认均衡

**验收**：能正确取出一批数据，形状、取值范围、标签都对；图能画出来。

### 阶段 2：前向传播（2~3 天）

- [ ] `layers.py`：实现 `Linear`（存 `W`、`b`，`forward(x)` 返回 `xW+b`）和 `ReLU`
- [ ] `init.py`：He 初始化（`W ~ N(0, sqrt(2/fan_in))`）；先试试全零初始化，观察会发生什么（这是重要一课）
- [ ] `model.py`：`Sequential` 按顺序调用各层 `forward`
- [ ] 单元测试：形状正确性（`(N,784)→(N,256)→(N,128)→(N,10)`）、ReLU 只在负数处置零

**验收**：随机权重跑一次前向得到 `(N, 10)` 的 logits，`pytest tests/` 通过。

### 阶段 3：损失与反向传播（3~4 天，本项目核心）

- [ ] `losses.py`：Softmax + 交叉熵**合并实现**（先算 `logits - max(logits)` 防溢出），提供 `forward(logits, y)` 和 `backward()` 返回 `dlogits`
- [ ] 给每层补 `backward(grad)`：
  - ReLU：按 mask 传递梯度
  - Linear：`dW = xᵀ·grad`，`db = sum(grad)`，`dx = grad·Wᵀ`（自己推一遍再写，别抄）
- [ ] `Sequential.backward` 逆序调用各层，把梯度存到每层 `dW`、`db`
- [ ] `scripts/grad_check.py`：数值梯度校验——对每个参数用 `(f(θ+ε)-f(θ-ε))/(2ε)` 估计梯度，与解析梯度比较，相对误差应 `< 1e-6`

**验收**：梯度校验全部通过。**这一步没通过之前不要进入下一阶段。**

### 阶段 4：优化器与训练循环（2~3 天）

- [ ] `optim.py`：SGD、SGD+Momentum（`v = μv - lr·g; θ += v`）
- [ ] `scripts/train_numpy.py` 完整训练循环：
  1. 每 epoch 打乱数据，按 batch 切片
  2. 前向 → 算损失 → 反向 → `optimizer.step()`
  3. 每 epoch 结束在验证集上评估，记录 loss/acc 曲线
  4. 保存最优参数与训练历史
- [ ] 超参起点：`lr=0.1`（SGD）、`batch=64`、`epochs=30`、He 初始化

**验收**：训练 loss 稳定下降，验证准确率 ≥ 85%。若不收敛，按第八节清单排查。

### 阶段 5：评估与误差分析（1~2 天）

- [ ] 测试集最终评估（只做一次），输出总体准确率
- [ ] 混淆矩阵：哪些类之间互相混淆？（典型：衬衫 vs T恤）
- [ ] 挑 20 个错判样本可视化，看模型"错在哪"
- [ ] 画出：训练/验证 loss 曲线、准确率曲线；观察是否过拟合（训练远好于验证）

**验收**：一份 notebook，含结论文字（"模型在 X 类上最差，因为…"），而非只有图。

### 阶段 6：PyTorch 复现与对照（2~3 天）

- [ ] `train_torch.py`：`nn.Sequential(Linear→ReLU→Linear→ReLU→Linear)` + `CrossEntropyLoss` + `SGD(momentum=0.9)`
- [ ] 超参与 A 完全相同，对比两者训练曲线与最终准确率
- [ ] 对照实验：在 A 和 B 中取同 batch 数据，比较一次前向的 logits 与一次反向的梯度（应基本一致）
- [ ] 思考题（写进实验记录）：`loss.backward()` 对应你手写的哪段代码？`optimizer.step()` 呢？`model.eval()` / `no_grad` 是干嘛的？

**验收**：两条线路准确率差距 < 1%，对照实验数值吻合，思考题能答上。

### 阶段 7：实验与调参（3~4 天，可选但强烈推荐）

每个实验改动**一个变量**，结果记入 `experiments/`：

| 实验 | 改动 | 想观察什么 |
|---|---|---|
| 1 | lr 取 0.001 / 0.01 / 0.1 / 1.0 | 太小收敛慢、太大发散 |
| 2 | batch 取 16 / 64 / 256 | 噪声与稳定的权衡 |
| 3 | 零初始化 vs He | 对称性破坏问题 |
| 4 | 隐藏层 16 / 128 / 512 | 容量与过拟合 |
| 5 | 无 Momentum vs μ=0.9 | 收敛速度 |
| 6 | 关掉数据归一化 | 数值范围的影响 |

**验收**：一张汇总表 + 每实验一段"结论"，形成你自己的直觉。

### 阶段 8：扩展方向（做完后选一个继续）

- 加入 Dropout / L2 正则化 / 学习率衰减，看能否再涨点
- 把 A 线路扩展成支持自动微分的迷你框架（像 PyTorch 那样 `loss.backward()`）
- 用 PyTorch 搭第一个 CNN（LeNet 即可），体会卷积对图像的优势

---

## 六、里程碑总览

| 里程碑 | 判定标准 |
|---|---|
| M1 数据就绪 | 形状/范围正确，可视化正常 |
| M2 前向跑通 | logits 形状正确，pytest 通过 |
| M3 反向正确 | **数值梯度校验 < 1e-6** |
| M4 训练收敛 | 验证集 ≥ 85% |
| M5 分析完成 | 混淆矩阵 + 错例 + 曲线 + 结论 |
| M6 对照完成 | PyTorch 结果一致，思考题可答 |

---

## 七、常见误区

1. **跳过手写直接用框架**——本计划的核心价值在阶段 3，别跳。
2. **梯度校验没过就硬训**——错梯度有时也能降 loss，但学不到真东西。
3. shape 错误用 `reshape` 硬掰——先想清楚每步的维度含义（建议在注释里标注）。
4. 只看准确率不看 loss 曲线——曲线才能告诉你"病在哪"。
5. 一次改多个超参——永远单变量实验。
6. 在测试集上反复调参——那是作弊，调参只看验证集。

## 八、不收敛 / 崩溃排查清单（按顺序检查）

1. 数据：形状？范围 `[0,1]`？标签是 0~9 的整数而非 one-hot？（交叉熵两种实现要对应）
2. 损失：Softmax 有没有减 max？除法有没有加 `+1e-12`？
3. 梯度：重新跑 `grad_check.py`；确认 `backward` 顺序是逆序。
4. 学习率：loss 变 NaN → 调小 10 倍；loss 不动 → 调大 10 倍。
5. 初始化：是否忘了 He/随机初始化？
6. 更新符号：是减梯度不是加梯度（`W -= lr * dW`）。

## 九、推荐资料（按需查阅，不必先读完）

- 3Blue1Brown《神经网络》系列视频——直觉
- 《Dive into Deep Learning》(d2l.ai) 中文版 第 3~5 章——与本计划路线几乎一致，可作参考书
- CS231n 讲义：backprop / optimization 章节——梯度与优化的最佳文字讲解
- PyTorch 官方 60 分钟教程——阶段 6 前看

---

*进度追踪：完成一项就把对应 `[ ]` 勾成 `[x]`。当前阶段：阶段 1（数据加载与可视化）。*
