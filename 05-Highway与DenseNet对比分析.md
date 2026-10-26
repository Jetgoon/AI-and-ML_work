# 第 2 题：Highway Network、ResNet 与 DenseNet 的连接方式对比

## 0. 结论摘要

**a. Highway 与 ResNet 的异同**

两者都通过"短路"把浅层信息/梯度直接送到深层，都属于 $y=f(x)+g(x)$ 式的多分支结构；**差别在于短路是否被"门控"调制**：

- Highway：$y = T\odot H(x) + (1-T)\odot x$，$T=\sigma(W_Tx+b_T)$ 是**输入相关的逐元素软门控**，输出是 $H$ 与 $x$ 的**凸组合**，短路支路自带参数；
- ResNet：$y = x + F(x)$，短路是**无参数、无门控的硬恒等映射**，直通系数恒为 $1$，输出无值域约束。

一句话：**Highway 是"可学习的软路由"，ResNet 是"永远打开的硬直连"**。

**b. DenseNet 与 ResNet 的本质不同**

- ResNet 的连接算子是**逐元素相加**（element-wise addition），拓扑是"链式 + 相邻跳连"，连接数 $O(L)$，语义是**残差修正**；
- DenseNet 的连接算子是**通道维拼接**（concatenation），拓扑是"每层连接全部前驱"，连接数 $O(L^2)$，语义是**特征复用**。

代价与收益：DenseNet 用参数效率和更短的梯度路径，换来显存/带宽开销与部署不友好。

---

## 1. 问题 a：Highway Network 与 ResNet 的快捷连接

### 1.1 两种结构的形式化

**Highway Network**（Srivastava, Greff & Schmidhuber, NIPS 2015；arXiv 2015 年 7 月）：

$$
y = H(x, W_H)\odot T(x, W_T) \;+\; x\odot C(x, W_C),
\qquad
T(x)=\sigma(W_Tx+b_T),\quad C(x)=1-T(x)
$$

即

$$
y = T\odot H(x,W_H) + (1-T)\odot x .
$$

其中 $T$ 为 transform gate（变换门），$C$ 为 carry gate（携带门），$\sigma$ 为 sigmoid，$\odot$ 为逐元素乘。门控的偏置 $b_T$ 通常初始化为负值（原论文取 $-1\sim-3$），使初始 $T\approx 0$，网络一开始接近恒等映射。

**ResNet**（He et al., CVPR 2016；arXiv 2015 年 12 月）：

$$
y = x + F(x, W_F).
$$

无门控、无额外参数（恒等短路），$F$ 只需学习"残差"。

> 时间上 Highway 略早约 5 个月，二者独立提出、动机相同（让极深网络可训练），因此题目称"几乎同时"。

### 1.2 相同点

1. **动机相同**：都针对"深度增加反而退化"的问题，用短路缩短信号与梯度的传播路径。
2. **结构同族**：都是"主变换支路 + 旁路"的双分支结构，都可写成 $y=\alpha\,T(x)+(1-\alpha)\,x$ 的加权形式（ResNet 相当于 $\alpha=1$ 且把 $T$ 改成 $x+F$）。
3. **都需处理维度匹配**：$x$ 与变换输出维度不一致时，Highway 用 $1\times1$ 卷积/下采样，ResNet 用 projection shortcut（$1\times1$ 卷积）或 zero-padding。
4. **都验证了"极深可训练"**：Highway 在 MNIST 上训练到近千层，ResNet 在 ImageNet 上训练到 152 层、在 CIFAR 上到 1001 层。
5. **都成为后续设计的思想来源**：门控/短路后来演化出 SE、SKNet、gating MoE 等；恒等短路成为 ResNet 家族与几乎所有现代骨干的标配。

### 1.3 不同点（核心）

**(1) 短路是否被调制：软门控 vs 硬恒等**

Highway 的旁路系数是 $1-T(x)\in(0,1)$，**依赖输入、可学习、可关闭**：网络可以自适应地决定某一层"多变换一点"还是"多直通一点"，等价于一种**软性路由/自适应深度**。ResNet 的系数恒为 $1$，所有层都被无条件执行，网络没有"跳过某层"的能力。

**(2) 参数开销**

Highway 每个单元需要额外的门控参数 $W_T\in\mathbb{R}^{C\times C}$ 与 $b_T$，与主变换 $W_H$ 同量级，**参数量约增加一倍**；ResNet 的恒等短路**零参数**。这是 ResNet 更易扩展、更易部署的直接原因。

**(3) 输出值域：凸组合 vs 无界相加**

Highway 的输出逐元素落在 $x$ 与 $H(x)$ 之间（因 $T+(1-T)=1$，$T\in[0,1]$），是一种**插值**，天然不能"放大"信号；ResNet 的 $x+F(x)$ 无值域约束，$F$ 可以任意幅度地修正甚至反转特征。后者表达能力更强，也更适合 BN 之后的尺度变化。

**(4) 梯度结构（与第 1 题呼应）**

ResNet（第 1 题已推）：

$$
\frac{\partial y}{\partial x}=I+\frac{\partial F}{\partial x},
$$

直通项恒为 $I$，**与输入、参数无关，永不衰减**。

Highway 对 $x$ 求导（$T$ 也依赖 $x$）：

$$
\frac{\partial y}{\partial x}
=\mathrm{diag}(1-T)
+\mathrm{diag}(H-x)\frac{\partial T}{\partial x}
+\mathrm{diag}(T)\frac{\partial H}{\partial x}.
$$

其直通项是 $\mathrm{diag}(1-T)$：**取值在 $(0,1)$ 且随输入变化**。当门控倾向于"变换"（$T\to 1$）时，直通项趋于 $0$，梯度不再有保底下界；此外 $\partial T/\partial x$ 项引入 sigmoid 饱和区的额外非线性，深层连乘时可能不稳定。**ResNet 的 $I$ 是"结构性保底"，Highway 的 $1-T$ 是"学出来的、可退化的保底"**——这是二者在优化行为上最本质的差别。

**(5) 初始化与训练难度**

Highway 靠"把 $b_T$ 初始化为负值"来让网络初始接近恒等，需要谨慎的初始化与学习率 warm-up；ResNet 依靠 $F$ 的小尺度初始化（He init + BN）使初始 $F\approx 0$，天然近似恒等，且无门控的 sigmoid 饱和问题，训练更稳定。

**(6) 表达与可解释性**

Highway 的门控值可作为"该层被使用了多少"的指标，具有一定可解释性与剪枝潜力（$T\to0$ 的层可视为可跳过）；但门控与主变换耦合，实际很难干净地"关掉"一层。ResNet 没有这种选择性，结构更"笨"但更稳健。

### 1.4 对比表

| 维度 | Highway Network | ResNet |
| --- | --- | --- |
| 连接公式 | $y=T\odot H+(1-T)\odot x$ | $y=x+F(x)$ |
| 短路系数 | $1-T(x)$，输入相关、可学习 | 恒为 $1$ |
| 是否门控 | 是（sigmoid 软门控，逐元素） | 否 |
| 额外参数 | 有（$W_T,b_T$，与主变换同量级） | 无（identity） |
| 输出值域 | $x$ 与 $H$ 的凸组合，有界 | 无约束 |
| 直通梯度项 | $\mathrm{diag}(1-T)$，可变、可退化 | $I$，恒为 $1$ |
| 初始化策略 | $b_T<0$ 使初始接近恒等 | $F$ 小尺度初始化使初始接近恒等 |
| 维度不匹配 | $1\times1$ 卷积 / 下采样 | projection（$1\times1$ conv）或 zero-padding |
| 可解释性/选择性 | 门控值可解释，具自适应深度潜力 | 无选择性 |
| 代表深度 | MNIST 上近千层 | ImageNet 152 层、CIFAR 1001 层 |
| 历史地位 | 提出更早，但未成为主流 | 成为视觉骨干的事实标准 |

### 1.5 为什么最终是 ResNet 胜出

1. **更简单**：无门控参数、无额外非线性，参数量与计算量更低，超参更少。
2. **梯度更有保障**：$I$ 项结构性存在，不依赖门控学得好不好；Highway 的 $1-T$ 可能被学小。
3. **表达不受限**：凸组合排除了"放大/反转"类变换，ResNet 的加法无此限制。
4. **与 BN、大规模数据更契合**：ResNet 的结构在 ImageNet 这类大规模任务上更易 scale，工程实现（卷积、残差、BN）对硬件友好。
5. **可迁移**：恒等短路可直接嵌入检测、分割、视频等框架（FPN、Mask R-CNN 等），Highway 的门控则较少被沿用。

> 严谨表述：**不宜简单说"ResNet 是 Highway 的特例"**。ResNet 没有门控、输出也不是凸组合；只能说"Highway 在思想上更一般，ResNet 是把门控退化为恒等、把插值改为相加的简化版本"。

---

## 2. 问题 b：DenseNet 与 ResNet 的连接方式

### 2.1 DenseNet 的形式化

DenseNet（Huang et al., CVPR 2017 Best Paper；arXiv 2016）中，第 $l$ 层的输入是**前面所有层输出的通道拼接**：

$$
x_l = H_l\bigl([\,x_0;\ x_1;\ \dots;\ x_{l-1}\,]\bigr),
$$

其中 $[\cdot;\cdot]$ 表示沿通道维拼接，$H_l$ 通常为 BN-ReLU-Conv($1\times1$) → BN-ReLU-Conv($3\times3$) 的瓶颈结构。每个 $H_l$ 只产生 $k$ 个新通道（**growth rate**），故第 $l$ 层输入通道数为 $k_0+k(l-1)$，随深度**线性增长**。相邻 dense block 之间用 transition layer（BN-Conv$1\times1$ → AvgPool$2\times2$，压缩率 $\theta$）降维。

一个有 $L$ 层的 dense block 含 $L(L+1)/2$ 条连接，而 ResNet 为 $O(L)$。

### 2.2 本质不同：五个层面

**(1) 聚合算子：相加 vs 拼接**

这是最本质的区别。ResNet 做**逐元素相加** $x+F(x)$，要求两者维度相同，不同层的信息被"混在一起"，事后无法区分来源；DenseNet 做**通道拼接**，各层特征**并列保存**，来源可区分、可被后续层选择性使用。相加是"融合"，拼接是"保留"。

**(2) 拓扑：局部链式 vs 全局密集**

ResNet 的每个单元只与**前一个**单元相连（外加一条短路），连接数 $O(L)$，信息沿链传递；DenseNet 的每层与**所有前驱**相连，连接数 $O(L^2)$，信息可跨层直达。

**(3) 学习目标：残差修正 vs 新增特征**

ResNet 的 $F$ 学的是"对已有特征的修正量"，输出复用 $x$ 的全部通道；DenseNet 的 $H_l$ 只学"$k$ 个新特征"，旧特征原样保留、直接复用。前者是**深度方向的增量修正**，后者是**宽度方向的增量积累**。

**(4) 维度约束**

ResNet 因要相加，每个 stage 内通道数必须固定，通道变化只能靠 projection 或下采样；DenseNet 因拼接天然允许维度增长，用 growth rate $k$ 精细控制容量，不必"一次性"设定很宽的通道。

**(5) 梯度路径**

ResNet 的梯度靠 $I$ 逐层直通，$x_l$ 到 loss 仍需经过 $L-l$ 个单元的连乘（尽管每步增益为 1）。DenseNet 中 $x_l$ 被直接拼进其后**每一层**的输入，也直接进入最后的全局池化与分类器，因此 $x_l$ 到 loss 的路径长度可短至 $O(1)$，**不经过任何中间卷积层**。这使 DenseNet 具有类似"隐式深度监督"的效果：每一层都能直接收到损失信号。

### 2.3 优缺点

**DenseNet 的优点**

| 优点 | 机理 |
| --- | --- |
| 参数效率高 | 每层只产 $k$ 个通道，参数集中在 $1\times1$ 瓶颈；DenseNet-121 约 8M 参数，而 ResNet-50 约 25.6M，同量级精度下参数约为其 $1/3$ |
| 特征复用充分 | 浅层边缘/纹理特征可被深层直接取用，减少重复学习；拼接而非相加，不破坏特征 |
| 梯度流短 | 每层到 loss 路径短，缓解梯度消失，小数据集上也易训练 |
| 正则效应 | 密集连接近似"多子网络集成"，过拟合更少；CIFAR/STL 等中小数据集上优势明显 |
| 容量可调 | growth rate $k$ 与压缩率 $\theta$ 提供细粒度的容量/效率权衡 |

**DenseNet 的缺点**

| 缺点 | 机理 |
| --- | --- |
| 显存/内存开销大 | 反向传播需保存所有中间特征图，拼接产生大量内存拷贝；激活占用随深度近似平方增长，batch size 受限 |
| 推理不占优 | FLOPs 低但受**内存带宽**制约，实际延迟未必优于同精度 ResNet；对部署/移动端不友好 |
| 通道线性膨胀 | 后期层输入通道达 $k_0+k(L-1)$，计算集中在前部 $1\times1$ 卷积，需瓶颈与 transition 压缩 |
| 工程复杂 | 拼接/切片、通道对齐使剪枝、量化、算子融合更困难，硬件利用率不如 ResNet |
| 通用性略逊 | 检测/分割等下游任务中，ResNet/FPN 系仍是更常见、更易改造的骨干 |

**ResNet 的优缺点（作为对照）**

- 优点：结构极简、零额外参数、易优化易扩展、显存与带宽友好、部署生态成熟、可作为通用骨干。
- 缺点：相加会混合不同层信息、特征复用不显式；极深时仍存在表示瓶颈（后续由 pre-activation、Wide ResNet、ResNeXt 等改进）。

### 2.4 定量对照（量级参考）

| 模型 | 参数量 | ImageNet top-1（单 crop） | 备注 |
| --- | --- | --- | --- |
| ResNet-50 | ≈25.6M | ≈76% | 基准骨干 |
| DenseNet-121 | ≈8.0M | ≈75% | 参数约为 ResNet-50 的 1/3 |
| DenseNet-169 | ≈14.1M | ≈76% | 参数约为 ResNet-50 的 55% |
| DenseNet-201 | ≈20.0M | ≈77% | 参数更少、精度更高 |

> 数值取自各自原论文的 ImageNet 单 crop 结果，随训练配置（分辨率、epoch、增强）略有浮动，此处仅用于说明"参数效率"这一趋势。

---

## 3. 三者的统一视角

| 结构 | 连接算子 | 门控 | 额外参数 | 连接数 | 直通项 | 核心语义 |
| --- | --- | --- | --- | --- | --- | --- |
| Highway | 门控凸组合 | 有（sigmoid） | 有 | $O(L)$ | $\mathrm{diag}(1-T)$，可变 | 自适应软路由 |
| ResNet | 逐元素相加 | 无 | 无 | $O(L)$ | $I$，恒 1 | 残差修正 |
| DenseNet | 通道拼接 | 无 | 无（通道增长） | $O(L^2)$ | 切片直通，恒 1 | 特征复用 |

一条演进线索：**Highway 提出"短路 + 门控"，ResNet 把门控去掉、把短路做成恒等，换来简单与稳定；DenseNet 保留短路思想，但把"相加"换成"拼接"、把"相邻"扩展到"全前驱"，用连接换参数。**

---

## 4. 结论

1. **a 题**：Highway 与 ResNet 同为"主支路 + 短路"结构，目标一致；区别在于 Highway 的短路被输入相关的 sigmoid 门控调制、输出为凸组合、需额外参数，ResNet 的短路是无门控无参数的恒等映射、直通梯度恒为 $I$。前者灵活但昂贵且训练更难，后者简单、梯度有结构性保底，因而成为主流。
2. **b 题**：DenseNet 与 ResNet 的本质差异是**聚合算子（concat vs add）**与**拓扑（全前驱 vs 链式）**，由此决定前者是"特征复用"、后者是"残差修正"。DenseNet 以 $O(L^2)$ 连接换取参数效率与短梯度路径，代价是显存、带宽与部署复杂度；ResNet 以简单、通用、硬件友好取胜。

---

## 附录：常见误答与易错点

1. **"ResNet 是 Highway 的特例"** —— 不严谨。ResNet 无门控、输出非凸组合，二者是"简化/退化"的思想关系，不是严格的集合包含关系。稳妥说法：ResNet 相当于把 Highway 的门控固定为"恒开"并把插值改为相加。
2. **"DenseNet 是 ResNet 的改进版"** —— 不准确。DenseNet 不做残差相加，聚合算子根本不同；它是对"连接思想"的另一条路线。
3. **"DenseNet 参数更多"** —— 反了。DenseNet 参数更少（growth rate 小），但**激活/显存**更多，二者要分开说。
4. **"拼接后通道数不变"** —— 错。通道数按 $k$ 线性增长，这正是 growth rate 的含义，也是后期计算量集中的原因。
5. **忽略维度匹配** —— ResNet 的 identity 只在同维时成立，跨 stage 需 projection 或 zero-padding；Highway 同样需要 $1\times1$ 卷积处理维度变化。
6. **把"梯度不消失"绝对化** —— 恒等短路提供结构性直通项，但不排除相消干涉与爆炸；Highway 的门控直通项还可能被学小，二者都需 BN/初始化配合。

## 参考文献

1. Srivastava, R. K., Greff, K., & Schmidhuber, J. (2015). *Training Very Deep Networks.* NeurIPS 2015. arXiv:1507.06228
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR 2016. arXiv:1512.03385
3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Identity Mappings in Deep Residual Networks.* ECCV 2016. arXiv:1603.05027
4. Huang, G., Liu, Z., van der Maaten, L., & Weinberger, K. Q. (2017). *Densely Connected Convolutional Networks.* CVPR 2017. arXiv:1608.06993
