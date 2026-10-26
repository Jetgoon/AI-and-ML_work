# ResNet 中权重层梯度 $\partial L/\partial W_l$ 的推导及其与深层梯度 $\partial L/\partial x_{l+1}$ 的关联

## 0. 结论摘要

设第 $l$ 个残差单元的前向为

$$
x_{l+1} = x_l + F(x_l, W_l),
$$

其中 $F$ 为该单元的残差分支（可含多个权重层、BN 与 ReLU），$W_l$ 为其中的可学习参数，$L$ 为标量损失函数。取行向量（梯度）约定，则：

**（1）权重层梯度**

$$
\boxed{\ \frac{\partial L}{\partial W_l}
= \frac{\partial L}{\partial x_{l+1}}\cdot \frac{\partial F(x_l, W_l)}{\partial W_l}\ }
$$

**（2）输入侧（向浅层回传）的梯度**

$$
\frac{\partial L}{\partial x_l}
= \frac{\partial L}{\partial x_{l+1}}\cdot \frac{\partial x_{l+1}}{\partial x_l}
= \frac{\partial L}{\partial x_{l+1}}\left(I + \frac{\partial F(x_l, W_l)}{\partial x_l}\right)
$$

**（3）二者的关联（一句话）**：$W_l$ 的梯度以 $\partial L/\partial x_{l+1}$ 为**唯一的上游入口**，经残差分支的局部 Jacobian $\partial F/\partial W_l$ 线性映射得到；而向 $x_l$ 回传时多出一个**单位矩阵 $I$**，因为恒等短路支路不含任何参数。这个 $I$ 是 ResNet 梯度可以"无损直通"的数学根源。

---

## 1. 记号与基本设定

| 符号 | 含义 | 维度 |
| --- | --- | --- |
| $x_l$ | 第 $l$ 个残差单元的输入特征 | $\mathbb{R}^{C}$（或 $\mathbb{R}^{C\times H\times W}$） |
| $x_{l+1}$ | 该单元的输出特征 | 与 $x_l$ 同维 |
| $F(\cdot,W_l)$ | 残差分支（残差函数） | 与 $x_l$ 同维 |
| $W_l$ | 残差分支中的权重参数（可为一组张量） | — |
| $L$ | 标量损失 | $\mathbb{R}$ |
| $\partial L/\partial x_{l+1}$ | 由更深层回传的上游梯度（"深层梯度"） | 与 $x_{l+1}$ 同维 |

**假设**：$F$ 关于 $x_l$ 与 $W_l$ 均可微；$x_{l+1}$ 与 $x_l$ 维度一致，故恒等支路可直接相加（维度不匹配时见 §6.3 的投影短路讨论）。推导中把梯度写成行向量，Jacobian 右乘，与反向传播（reverse-mode AD）的实现约定一致。

**计算图上的两条依赖路径**：$x_{l+1}$ 依赖 $x_l$ 有两条路径——① 恒等支路 $x_l \to x_{l+1}$；② 残差支路 $x_l \to F \to x_{l+1}$。而 $x_{l+1}$ 依赖 $W_l$ **只有一条**路径：$W_l \to F \to x_{l+1}$。这一非对称性是全部结论的来源。

---

## 2. 先求输入侧梯度 $\partial L/\partial x_l$

由多元链式法则，对 $x_l$ 的全微分：

$$
dx_{l+1} = dx_l + \frac{\partial F}{\partial x_l}\,dx_l
= \left(I + \frac{\partial F(x_l,W_l)}{\partial x_l}\right) dx_l .
$$

两边左乘 $\partial L/\partial x_{l+1}$，得到

$$
\boxed{\ \frac{\partial L}{\partial x_l}
= \frac{\partial L}{\partial x_{l+1}}
+ \frac{\partial L}{\partial x_{l+1}}\frac{\partial F(x_l,W_l)}{\partial x_l}\ }
$$

即梯度由两项**相加**：

- **直通项** $\dfrac{\partial L}{\partial x_{l+1}}$：梯度沿恒等支路原样传回，**不经过任何权重层**；
- **残差项** $\dfrac{\partial L}{\partial x_{l+1}}\dfrac{\partial F}{\partial x_l}$：梯度经残差分支的 Jacobian 变换后传回。

若记 $J_l \triangleq \dfrac{\partial F(x_l,W_l)}{\partial x_l}\in\mathbb{R}^{C\times C}$，则简洁地写作

$$
\frac{\partial L}{\partial x_l} = \frac{\partial L}{\partial x_{l+1}}\,(I + J_l).
$$

**关键点**：$I$ 与 $W_l$ 无关，因此"1"这一项在任何训练阶段、任何参数取值下都恒存在。这正是残差连接与普通前馈层最本质的区别——普通层只有 $J_l$，一旦 $\|J_l\|<1$ 便逐层衰减。

---

## 3. 权重层梯度 $\partial L/\partial W_l$ 的表达式

$W_l$ 只出现在残差分支中，且 $x_l$ 与 $W_l$ 是相互独立的变量（$x_l$ 由前向计算得到，不显含 $W_l$），故

$$
\frac{\partial x_{l+1}}{\partial W_l}
= \frac{\partial}{\partial W_l}\bigl[x_l + F(x_l,W_l)\bigr]
= \frac{\partial F(x_l,W_l)}{\partial W_l}.
$$

代入链式法则：

$$
\boxed{\ \frac{\partial L}{\partial W_l}
= \frac{\partial L}{\partial x_{l+1}}\cdot\frac{\partial F(x_l,W_l)}{\partial W_l}\ }
$$

写作算子（伴随）形式更严谨：把 $\partial F/\partial W_l$ 视为线性算子 $\Delta W \mapsto \Delta y$，则

$$
\frac{\partial L}{\partial W_l}
= \left(\frac{\partial F(x_l,W_l)}{\partial W_l}\right)^{\!\top}
\frac{\partial L}{\partial x_{l+1}},
$$

即"上游梯度经局部 Jacobian 的转置作用回参数空间"。这与标准的反向传播公式完全一致。

**特例（标量形式）**：若 $x_l,f,W_l$ 均为标量，$x_{l+1}=x_l+f(x_l,W_l)$，则

$$
\frac{\partial L}{\partial W_l}=\frac{\partial L}{\partial x_{l+1}}\cdot\frac{\partial f}{\partial W_l},
\qquad
\frac{\partial L}{\partial x_l}=\frac{\partial L}{\partial x_{l+1}}\left(1+\frac{\partial f}{\partial x_l}\right).
$$

**易错点**：此处必须使用**输出端**的上游梯度 $\partial L/\partial x_{l+1}$，而**不是** $\partial L/\partial x_l$。因为 $W_l$ 作用于 $x_l$ 后产生 $x_{l+1}$，其梯度只能从 $x_{l+1}$ 处回传；若误写成 $\partial L/\partial x_l \cdot \partial F/\partial W_l$，等价于把直通项也乘进了参数梯度，与自动微分结果不符（附录 A 给出数值验证）。

---

## 4. 二者的关联：结构对照

把 §2 与 §3 并列：

$$
\begin{aligned}
\frac{\partial L}{\partial x_l} &= \frac{\partial L}{\partial x_{l+1}}\,(I + J_l)
&&\text{（两条支路：恒等 + 残差）}\\[4pt]
\frac{\partial L}{\partial W_l} &= \frac{\partial L}{\partial x_{l+1}}\,\frac{\partial F}{\partial W_l}
&&\text{（一条支路：仅残差）}
\end{aligned}
$$

由此可读出四层关联：

1. **共同的上游因子**。两者都以 $\partial L/\partial x_{l+1}$ 为输入，说明第 $l$ 层的参数更新信号强弱，直接由深层回传梯度的大小决定。若 $\partial L/\partial x_{l+1}\to 0$，则 $\partial L/\partial W_l \to 0$，该层即"停止学习"。ResNet 的全部设计意图，就是保证 $\partial L/\partial x_{l+1}$ 不因层数增加而衰减。

2. **是否有 $I$ 项**。$\partial L/\partial W_l$ 中**不存在** $I$，因为恒等支路没有可学习参数；对 $W_l$ 求导时 $\partial x_l/\partial W_l = 0$，直通项被求导"消掉"。因此 $W_l$ 的梯度永远要经过至少一次 Jacobian $\partial F/\partial W_l$，它本身不能免疫衰减；它能被训练，靠的是 $\partial L/\partial x_{l+1}$ 中被 $I$ 保住的直通分量。

3. **线性映射关系**。固定 $x_l$ 时，$\partial L/\partial W_l$ 是 $\partial L/\partial x_{l+1}$ 的**线性函数**（经 Jacobian 转置作用）。故"深层梯度"与"权重梯度"不是两个独立量，而是同一股梯度流在参数空间与特征空间的两个投影。

4. **梯度流的接力结构**。把 $\partial L/\partial x_{l+1}$ 用更深层展开（§5），可得

$$
\frac{\partial L}{\partial W_l}
= \underbrace{\frac{\partial L}{\partial x_L}}_{\text{深层总梯度}}
\underbrace{\prod_{i=L-1}^{l+1}\!\left(I+J_i\right)}_{\text{含恒等项的传播链}}
\underbrace{\frac{\partial F(x_l,W_l)}{\partial W_l}}_{\text{局部 Jacobian}} .
$$

乘积按 $i$ 从 $L-1$ 递减到 $l+1$ 左乘展开。当 $l=L-1$（最后一层）时传播链为空，退化为 $\partial L/\partial W_{L-1}=(\partial L/\partial x_L)(\partial F/\partial W_{L-1})$，与直觉一致。

---

## 5. 跨层展开：为什么梯度不会消失

### 5.1 递归展开

对 $l<L$ 反复使用 §2 的递推式：

$$
\frac{\partial L}{\partial x_l}
= \frac{\partial L}{\partial x_L}\prod_{i=L-1}^{l}\left(I+J_i\right),
\qquad J_i=\frac{\partial F(x_i,W_i)}{\partial x_i}.
$$

该乘积展开后共 $2^{L-l}$ 项：

$$
\frac{\partial L}{\partial x_l}
= \frac{\partial L}{\partial x_L}\Bigl[
I
+ \sum_{i=l}^{L-1} J_i
+ \sum_{l\le i<j\le L-1} J_iJ_j
+ \cdots
+ \prod_{i=l}^{L-1} J_i
\Bigr].
$$

其中：

- **第 1 项 $I$** 对应"从第 $L$ 层一路走恒等支路直达第 $l$ 层"的路径，长度为 1 条，**不经过任何权重层**；
- 第 2 类项对应"只穿过 1 个残差分支、其余走恒等支路"的路径，共 $L-l$ 条；
- 一般地，含 $k$ 个 $J$ 的项对应"穿过 $k$ 个残差分支"的路径，共 $\binom{L-l}{k}$ 条；
- **末项 $\prod J_i$** 对应"全程走残差支路"的普通网络路径。

即：梯度沿 $2^{L-l}$ 条**不同长度的路径**回传，路径长度从 1 到 $L-l$ 呈二项分布。极短路径的存在，使得深层梯度中始终保留一个等于 $\partial L/\partial x_L$ 的分量。

### 5.2 与普通（plain）网络的对比

| | 普通网络 $x_{l+1}=H(x_l,W_l)$ | ResNet $x_{l+1}=x_l+F(x_l,W_l)$ |
| --- | --- | --- |
| 单层 Jacobian | $\partial H/\partial x_l$ | $I+\partial F/\partial x_l$ |
| 跨层梯度 | $\dfrac{\partial L}{\partial x_L}\prod_i \dfrac{\partial H}{\partial x_i}$ | $\dfrac{\partial L}{\partial x_L}\prod_i\left(I+\dfrac{\partial F}{\partial x_i}\right)$ |
| 路径数 | $1$ | $2^{L-l}$ |
| 尺度行为 | 若每层谱范数 $\rho<1$，则 $\rho^{L-l}$ **指数衰减** | 若 $\|J_i\|\approx\rho\ll1$，则 $\approx(1+\rho)^{L-l}$ **不衰减（缓慢增长）** |
| 梯度消失风险 | 高 | 显著缓解 |

尺度分析可写得更明确：设 $\|J_i\|_2\le \rho$，则

$$
\left\|\prod_{i=l}^{L-1}(I+J_i)\right\|_2 \le (1+\rho)^{L-l},
$$

而普通网络的上界是 $\rho^{L-l}$。前者随深度**指数增长**（故需 BN/权重衰减等控制 $\rho$），后者随深度**指数衰减**。这正是"ResNet 能训练上千层、plain 网络 20 层就退化"的谱层面的解释。

### 5.3 严谨性补充（避免过度断言）

- "梯度不消失"是**结构性保证**而非绝对保证：$\prod(I+J_i)$ 的 $2^{L-l}$ 项之间可能发生**相消干涉**（shattered gradients 问题），且若 $\rho>1$ 仍会爆炸。ResNet 的作用是把"必然的指数衰减"变成"可控的、含恒等直通分量的传播"。
- $I$ 项保证的是：**梯度中存在一条不依赖任何权重的单位增益路径**，从而使 $\partial L/\partial x_l$ 关于 $\partial L/\partial x_L$ 的映射不会因参数变小而整体归零。

---

## 6. 推广与实现细节

### 6.1 残差分支含多个权重层

若 $F$ 由若干层复合而成，$F=F^{(K)}\circ\cdots\circ F^{(1)}$，$W_l=\{W_l^{(1)},\dots,W_l^{(K)}\}$，则对第 $k$ 层：

$$
\frac{\partial L}{\partial W_l^{(k)}}
= \frac{\partial L}{\partial x_{l+1}}
\cdot\frac{\partial F^{(K)}}{\partial z^{(K-1)}}\cdots\frac{\partial F^{(k+1)}}{\partial z^{(k)}}
\cdot\frac{\partial F^{(k)}}{\partial W_l^{(k)}},
$$

即"上游梯度 → 残差分支内部逐层回传 → 该层参数 Jacobian"。$I$ 项只在**跨残差单元**时出现，残差单元内部仍是普通的链式连乘，这一点常被忽略。

### 6.2 BN 与 ReLU 的影响

- 含 BN 时，$J_l=\partial F/\partial x_l$ 不是对角阵（BN 在 batch 维耦合），但其存在与否**不影响** $I$ 项，故直通性质保持。
- 含 ReLU 时 $J_l$ 含 0/1 掩码，可能大量元素为 0，使残差项稀疏；但直通项依然完整，这正是"ReLU 导致的死亡梯度"在 ResNet 中被缓冲的原因。
- **Pre-activation 结构**（He et al., 2016）把 BN/ReLU 移到卷积之前，使短路支路为纯恒等映射，此时 $x_L=x_l+\sum_{i=l}^{L-1}F(x_i,W_i)$ 严格成立，$\partial x_{l+1}/\partial x_l=I+J_l$ 无附加变换，梯度传播最干净。

### 6.3 投影短路（维度不匹配时）

若通道数改变，$x_{l+1}=W_s x_l+F(x_l,W_l)$，则

$$
\frac{\partial L}{\partial x_l}=\frac{\partial L}{\partial x_{l+1}}\left(W_s+J_l\right),
\qquad
\frac{\partial L}{\partial W_s}=\frac{\partial L}{\partial x_{l+1}}\,x_l^{\top}.
$$

此时直通项由 $I$ 变为可学习的 $W_s$，若 $W_s$ 退化则"梯度高速路"的增益不再恒为 1。这解释了为何原论文只在通道变化处使用 $1\times1$ 卷积投影，其余位置坚持 identity。

### 6.4 张量（卷积）情形

对特征图 $x_l\in\mathbb{R}^{C\times H\times W}$，上述公式逐元素成立，$I$ 解释为恒等映射算子，$J_l$ 为卷积算子的 Jacobian（块稀疏、Toeplitz 结构）；矩阵乘法对应张量缩并。无需改变任何结论。

### 6.5 一个可手算的最小例子

取 $F(x)=W_2\,\sigma(W_1x)$，$\sigma$ 为逐元素激活，$z=W_1x$，$D=\mathrm{diag}(\sigma'(z))$：

$$
J=\frac{\partial F}{\partial x}=W_2DW_1,\qquad
\frac{\partial L}{\partial x_l}=\frac{\partial L}{\partial x_{l+1}}\left(I+W_2DW_1\right),
$$

$$
\frac{\partial L}{\partial W_2}=\frac{\partial L}{\partial x_{l+1}}\,\sigma(z)^{\top},
\qquad
\frac{\partial L}{\partial W_1}=\frac{\partial L}{\partial x_{l+1}}\,W_2D\,x_l^{\top}.
$$

可直接看出：$W_2$ 的梯度只与"上游梯度 × 本层输入"有关（与普通网络无异），而 $x_l$ 的梯度多出 $I$ 项。

---

## 7. 结论

1. ResNet 权重层梯度为 $\dfrac{\partial L}{\partial W_l}=\dfrac{\partial L}{\partial x_{l+1}}\dfrac{\partial F(x_l,W_l)}{\partial W_l}$，它是上游梯度经残差分支局部 Jacobian 转置作用的结果，形式与普通网络一致，**差别不在参数梯度本身，而在上游梯度 $\partial L/\partial x_{l+1}$ 的质量**。
2. 输入侧梯度为 $\dfrac{\partial L}{\partial x_l}=\dfrac{\partial L}{\partial x_{l+1}}\left(I+\dfrac{\partial F}{\partial x_l}\right)$，比参数梯度多出恒等项 $I$；原因是短路支路不含参数，求导时不会被"消去"。
3. 跨层展开 $\dfrac{\partial L}{\partial x_l}=\dfrac{\partial L}{\partial x_L}\prod_{i=l}^{L-1}(I+J_i)$ 产生 $2^{L-l}$ 条长度不等的梯度路径，其中"全恒等"路径使梯度可无损跨越任意多层，从结构上缓解梯度消失。
4. 该机制同时带来上界放大（$\|I+J\|\le 1+\rho$），需借助 BN、合理初始化与权重衰减控制 $\rho$，否则存在梯度爆炸风险。

---

## 附录 A：数值验证（可选，用于佐证公式正确性）

```python
import torch, torch.nn as nn

torch.manual_seed(0)
x  = torch.randn(4, 8, requires_grad=True)
F  = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8))  # 残差分支
y  = x + F(x)                                                     # x_{l+1}
L  = (y ** 2).sum()                                               # 任取标量损失

W = list(F.parameters())
g_y, = torch.autograd.grad(L, y, retain_graph=True)               # ∂L/∂x_{l+1}
g_W  = torch.autograd.grad(L, W, retain_graph=True)               # ∂L/∂W_l（真值）
g_x  = torch.autograd.grad(L, x, retain_graph=True)[0]            # ∂L/∂x_l（真值）

# 公式(1)：∂L/∂W_l = <∂L/∂x_{l+1}, ∂F/∂W_l>
g_W_manual = torch.autograd.grad((F(x) * g_y).sum(), W, retain_graph=True)
# 公式(2)：∂L/∂x_l = ∂L/∂x_{l+1} + ∂L/∂x_{l+1}·∂F/∂x_l
g_x_manual = g_y + torch.autograd.grad((F(x) * g_y).sum(), x)[0]

print(all(torch.allclose(a, b) for a, b in zip(g_W, g_W_manual)))  # True
print(torch.allclose(g_x, g_x_manual))                             # True
```

两式均严格成立，可作为推导的实证校验。

## 附录 B：符号速查

- $J_l=\partial F(x_l,W_l)/\partial x_l$：残差分支对输入的 Jacobian
- $I$：恒等映射（单位矩阵 / 恒等算子）
- $\prod_{i=L-1}^{l}(I+J_i)=(I+J_{L-1})(I+J_{L-2})\cdots(I+J_l)$：按 $i$ 递减左乘
- "直通项"：梯度中不经过任何权重层的分量，对应展开式中的 $I$
