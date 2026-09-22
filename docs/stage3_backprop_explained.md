# 阶段 3 代码详解：损失函数与反向传播

> 配套代码：`src/nn_numpy/losses.py`、`layers.py`、`model.py`、`scripts/grad_check.py`、`tests/test_backprop.py`
> 验证结果：19/19 单元测试通过；数值梯度校验相对误差 ~1e-11（阈值 1e-6）

---

## 0. 全景图：一次完整的训练步骤中，阶段 3 负责哪一段

```
            ┌──────── 阶段2 已完成 ────────┐ ┌────────── 阶段 3（本阶段）──────────┐ ┌─ 阶段4 ─┐
   x ──> │ Linear → ReLU → Linear → ReLU → Linear │ ──> logits ──> SoftmaxCrossEntropy │ ──> loss
   y ──────────────────────────────────────────────────────────────^^──────────────────┘
                                                                    │
   dW/db ←── { Sequential.backward ←─ Linear.backward ←─ ReLU.backward ←─ loss.backward } ←─┘
            （梯度的流向：从 loss 出发，逆着前向的方向逐层回传）
```

- **前向传播**：数据 `x` 流向 `logits`，再经损失函数得到标量 `loss`
- **反向传播**：从 `loss` 出发，逆序经过每一层，算出每个参数的梯度 `dW`、`db`
- 反向传播只**生产梯度**，不**更新参数**——更新是优化器（阶段 4）的职责

---

## 1. `losses.py`：Softmax + 交叉熵合并实现

### 1.1 为什么这两个东西要绑在一起

分类网络的最后一层输出叫 **logits**：10 个任意实数，比如 `[2.1, -0.5, 0.3, ...]`。
要变成"每个类别的概率"需要 Softmax；要衡量"预测概率离正确答案多远"需要交叉熵。

如果分开实现，Softmax 的梯度是一个 10×10 雅可比矩阵（∂p_i/∂z_j 全都要算），
又慢又容易写错。但两者**复合**之后，梯度简洁得惊人：

```
dL/dz = (softmax(z) - onehot(y)) / N
```

即：**预测概率减去正确答案的 one-hot，再除以 batch 大小**。一行代码搞定。
这也是 PyTorch 的 `nn.CrossEntropyLoss` 直接吃 logits、内部自带 log-softmax 的原因。

### 1.2 `forward` 逐行拆解

```python
z = logits - logits.max(axis=1, keepdims=True)
```
**数值稳定第 1 招：减最大值。**
softmax 对整体平移不变（分子分母同乘 e^c 约掉），所以减去每行最大 logit 后：
最大值变成 0，`exp(0)=1`，永远不会 `exp(1000) → inf` 溢出。
测试 `test_loss_numerically_stable_with_huge_logits` 专门验证 logits=±3000 时也不出 NaN。

```python
log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
```
**数值稳定第 2 招：log 域计算。**
先算出 `log_softmax` 而不是先算 softmax 再取 log。
如果先算概率：小概率可能下溢成 0，`log(0) = -inf` 直接崩。
在 log 域里只有一次减法，天生安全。这正是 `torch.nn.functional.log_softmax` 的做法。

```python
loss = -log_probs[np.arange(n), y].mean()
```
**花式索引取"正确类的 log 概率"**：`np.arange(n)` 是行号 `[0,1,...,N-1]`，`y` 是每行要取的列号，
两个数组配对就取出了 `log_p[i, y_i]`——每个样本正确类的 log 概率。
取负、求平均，就是交叉熵。（除以 N 求平均这一步，backward 里要对应除回来，见下。）

```python
self.probs = np.exp(log_probs)   # 缓存
```
backward 只需要 `probs` 和 `y`，所以缓存这两个。**缓存是反向传播的原料**——没有它，反向时无米下锅。

### 1.3 `backward`：那个一行公式

```python
grad = self.probs.copy()          # 必须拷贝！直接改 self.probs 会污染缓存
grad[np.arange(n), self.y] -= 1.0 # 正确类位置减 1（softmax - onehot）
return grad / n                   # 因为 forward 对 batch 求了平均
```

梯度直观含义：**"错了多少就改多少"**。
- 若正确类概率 `p=0.7`：梯度该位置是 `0.7-1=-0.3`（负梯度→梯度上升方向会增大它）
- 若错误类概率 `p=0.5`：梯度是 `+0.5`（更新后会被压小）
- 预测越离谱，梯度越大，修正越猛

`.copy()` 是初学者最常踩的坑：不拷贝的话 `grad` 和 `self.probs` 是同一块内存，
`-= 1.0` 会把缓存也改掉，第二次调 backward 结果就错了。

---

## 2. `layers.py`：backward 与"缓存"的设计

### 2.1 一个关键认知：backward 需要前向的"证据"

数学上，`∂L/∂W` 的公式里含有前向的输入 `x`；ReLU 的梯度公式里含有"哪些位置是正数"。
所以 **forward 不只是算输出，还要为 backward 留下材料**：

| 层 | forward 缓存 | backward 用它干什么 |
|---|---|---|
| Linear | `_x`（输入） | `dW = xᵀ @ grad` 需要 x |
| ReLU | `_mask`（x>0） | 决定梯度哪些位置归零 |
| Loss | `probs`、`y` | `grad = probs - onehot(y)` |

PyTorch 里对应的就是 `ctx.save_for_backward(...)`——现在你知道它存这些是干嘛的了。

### 2.2 Linear 的三个公式与"形状验算法"

前向 `y = x @ W + b`，反向已知 `grad = dL/dy (N, out)`，要求三个梯度：

```python
self.dW = self._x.T @ grad   # (in,N)@(N,out) → (in,out)，与 W 同形 ✓
self.db = grad.sum(axis=0)   # (N,out) → (out)，与 b 同形 ✓
return grad @ self.W.T       # (N,out)@(out,in) → (N,in)，与 x 同形 ✓
```

**形状推理可以当验算器**：矩阵微积分记不清时，把所有已知量的形状摆出来，
能满足结果形状的相乘组合往往唯一（`in,N` × `N,out` 只有 `(xᵀ, grad)` 这一种搭配）。

为什么 `db` 是求和？`b` 加在了每一个样本上（广播），N 个样本各自贡献一份梯度，
按链式法则要加起来——所以沿着样本轴 `axis=0` 求和。

**为什么 dx 要传出去？** `dx` 是"损失对这层输入的梯度"，它就是**上一层的 grad**。
反向传播能一环扣一环，靠的就是每一层都把 `dx` 返回给调用者。

### 2.3 ReLU 的 backward：门卫隐喻

```python
self._mask = x > 0        # forward：记住哪些位置"门开着"
return grad * self._mask  # backward：门开的位置梯度通过，门关的归零
```

前向时负数被砍成 0（信息丢失），反向时那些位置**也不许梯度通过**——
相当于每个神经元对下游说："这次的结果不是因为我，别怪我"。
布尔 mask 与 float 相乘时自动当 0/1 用，无需类型转换。

注意边界：`x=0` 归入"关门"一侧（mask=False），这是约定，实践无影响。

---

## 3. `model.py`：Sequential 的反向就是逆序 for 循环

```python
for layer in reversed(self.layers):
    grad = layer.backward(grad)
```

前向 `A→B→C`，反向就是 `C→B→A`，每层拿到的 `grad` 恰是下游层返回的 `dx`。
**"反向传播"这个名字的字面意思就是这段代码**——没有任何更神秘的东西。

执行完之后，每个 Linear 层身上都挂着算好的 `self.dW`、`self.db`，
等着阶段 4 的优化器来取走并更新。

---

## 4. `grad_check.py`：你怎么敢说自己公式写对了？

### 4.1 原理：回到导数的定义

解析梯度（手推公式）可能是错的；**数值梯度（差分近似）不会骗人**：

```
∂L/∂θ ≈ [L(θ+ε) − L(θ−ε)] / (2ε)     （中心差分，ε=1e-5）
```

对每个参数挨个 +ε/−ε 各跑一次前向、算损失、差分，与手写的解析梯度对比。

### 4.2 几个工程细节（每个都有理由）

- **用小网络（7→5→4）**：参数少（35+20 个权重），逐参数差分要跑 2×参数次前向，网络大了会慢到没法用。公式对小的对，大的就对——数学不随规模改变
- **用 float64**：差分在 "loss 差 ~2ε×梯度" 量级上做除法，float32 的舍入噪声会淹没信号。这也是为什么校验误差能到 1e-11，远优于 1e-6 阈值
- **相对误差而非绝对误差**：`|a−b| / max(|a|,|b|)`，避免"梯度本身很小所以绝对误差也小"的假阳性
- **恢复参数原值**：`param[idx] = orig`，校验完网络要还原，不能污染

### 4.3 实测结果

```
layers[0].W   2.622e-11  ✓
layers[0].b   1.856e-11  ✓
layers[2].W   1.782e-11  ✓
layers[2].b   1.626e-11  ✓
```

四个参数张量的解析梯度与数值梯度在 1e-11 量级上吻合——
**手推的链式法则公式被实验证明了**。这就是进入阶段 4（训练）的通行证。

---

## 5. `tests/test_backprop.py`：都测了什么、为什么这么测

| 测试 | 验证的事实 |
|---|---|
| `test_loss_zero_for_perfect_prediction` | 极端置信时 loss→0（合理边界） |
| `test_loss_value_against_reference` | 与"朴素 softmax+CE"逐数值一致（合并实现没算错） |
| `test_loss_numerically_stable_with_huge_logits` | logits=±1000 有限（减 max 生效） |
| `test_softmax_rows_sum_to_one` | 输出确实是概率分布 |
| `test_gradient_of_loss_matches_softmax_minus_onehot` | **核心公式 `(p−onehot)/N` 本身** |
| `test_relu_backward_zero_on_negatives` | mask 只砍负数与零 |
| `test_linear_backward_shapes` | dW/db/dx 形状与参数/输入一致（形状验算法落地） |
| `test_linear_backward_formulas_exact` | 用手算的小矩阵逐元素核对三个公式 |
| `test_sequential_backward_returns_input_grad` | 逆序回传后能给出对输入的梯度 |

---

## 6. 对照 PyTorch：你已经手写了什么

| PyTorch 写法 | 对应本项目 |
|---|---|
| `loss.backward()` | `loss_fn.backward()` + `model.backward(dlogits)` 这两行 |
| `W.grad` | `layer.dW`、`layer.db` |
| `ctx.save_for_backward` | `self._x`、`self._mask`、`self.probs` |
| `nn.CrossEntropyLoss`（吃 logits） | `SoftmaxCrossEntropy` 的合并实现+减 max |
| autograd 的计算图 | 你手动维护的"缓存 + 逆序遍历" |

阶段 6 做 PyTorch 对照实验时，回来翻这张表。

---

## 7. 常见坑清单（写代码时实际防过的）

1. `probs.copy()` 不拷贝 → 污染缓存，第二次 backward 就错
2. forward 忘缓存 `x`/mask → backward 无材料
3. loss 求了平均但 backward 忘除 N → 梯度大 N 倍，等效学习率放大
4. 标签用 one-hot 还是整数索引，`p − y` 两种写法不通用（本项目统一整数）
5. `log(0)` → -inf：必须走 log 域
6. backward 顺序忘了 `reversed` → 梯度全错（梯度校验会立刻暴露）
