---
title: "生成模型(1): Flow Matching"
date: 2026-05-19
tldr: "从 ODE 与向量场的视角构建生成模型：通过构造连接先验与数据分布的概率路径（高斯/ConOT 路径），利用条件流匹配损失等价训练神经网络速度场，实现从标准高斯噪声到真实数据样本的确定性转换，并推导了预测速度、噪声与原始数据三种等价的参数化形式。"
math: true
---
## Generation via Sampling
生成就是从真实的数据分布中进行采样
$$ x \sim p_{data} (\cdot)$$
一个生成模型就是能够完成从真实分布中采样的模型. 
在实际中, 我们无法获得全部数据, 往往只能使模型尽可能从某个数据集的分布中采样; 模型能够贴近的数据集越大, 越全面, 模型的泛化性和通用性就越强. SOTA的图片和视频生成模型往往是在互联网数据(youtube 视频, ins 图片等)上训练得到的, 这些模型的训练目标就是使其能够采样得到符合这些大规模数据集的数据.
我们一般也会希望能够得到基于某些条件进行生成的模型. 比如给一段文字描述, 然后模型生成符合这段描述的图片. 在我们的视角下, 这个过程是从一个条件分布中采样
$$ x \sim p_{data} (\cdot | y)$$
其中 $y$ 是条件. 为了与其它概念作区分, 我们接下来称基于给定条件的生成过程为**引导下的生成** (Guided Generation).
于是接下来我们的问题就是, 如何构造并训练一个模型, 使其能够从数据集的分布中进行采样.

## Flow Model
在进行进一步讨论之前, 我们对任务作一些更规范化的处理. 我们有一个数据集, 其中的数据可以视为从数据分布中采样得到 $z_1, z_2, ..., z_N \sim p_{data}(\cdot)$. 我们的目的是得到一个模型, 其作用是从 $p_\theta$ 中采样, 要使 $p_\theta$ 尽可能接近 $p_{data}$. 此外, 我们面对的数据可以转换为 $\mathbb{R}^d$ 中的矢量(比如图片, 视频等). 接下来我们将通过构造一个常微分方程, 来得到一个生成模型.
首先我们定义一个常微分方程(ODE). 常微分方程定义在 **轨迹(Trajectory)** 上. 轨迹是一个时间的函数
$$t \to X_t, X_t \in \mathbb{R}^d$$.
一般来说, $t \in [0, 1]$
常微分方程定义了一个 **向量场 (vector field)** $u_t(x): [0, 1] \times \mathbb{R}^d \to \mathbb{R}^d$, 它满足
$$\frac{d}{d t} X_t = u_t(X_t), ~ X_0 = x_0$$
$x_0$ 被称为初始条件.
直观来说, 轨迹定义了某个粒子在不同时刻的位置, 而向量场则定义了某个时刻某个位置的粒子的速度. 那么对于初始条件为 $x$ 的轨迹, 其 $t$ 时刻在何处呢? 我们定义**流 (Flow)**
$$\psi_t(x) \in \mathbb{R}^d$$
$$\psi_0(x) = x$$
$$\partial_t \psi_t(x) = u_t(\psi_t(x))$$
对于从任何初始条件 $X_0$ 的轨迹, 其在 $t$ 时刻的值总能由流给出 $X_t = \psi_t(X_0)$. 可以说, 由速度场 $u_t(x)$ 给出的 ODE 有解 $\psi_t(x)$. 在我们的例子中, 可以简单认为在给定初始条件下, 轨迹是唯一确定的.
我们的目的是能够从目标数据分布中采样, 但是 ODE 是确定的. 我们一般通过从一个固定分布 $p_{init}$ 中采样初始条件 $X_0 \sim p_{init}$ 来获取随机, 这个初始分布一般是标准高斯分布 $N(0, I_d)$; 这样如果我们能够找到一个向量场 $u_t(x)$ 使轨迹的终点 $X_1(x) \sim p_{data}$, 那么这个向量场 $u_t(x)$ 就是一个满足我们需要的生成模型. 我们一般用一个神经网络来表示向量场, 这个神经网络的输入是 $t$ 和 $x$, 记为 $u_t^\theta(x)$ , $\theta$ 表示其参数
$$X_0 \sim p_{init}$$
$$\frac{d}{dt} X_t = u_t(X_t)$$
$$X_1 \sim p_{data}$$
如果用流  $\psi_t^\theta(x)$ ($\theta$ 意味着 $u_t^\theta(x)$ 的参数) 表示就是
$$X_0 \sim p_{init} \iff \psi_1^\theta(X_0) \sim p_{data}$$.
通过这个向量场构建的生成模型被称为 **流模型 (Flow Model)**. 其采样过程可以用 Euler 法完成
```python
h = 1 / n
X_0 ~ p_init
for i in range(n):
	X_(i + 1)h = X_ih + h * u(X_ih, i*h)
X_1 is what we want
```

## Flow Matching
我们已经讨论过用一个神经向量场 $u_t^\theta(x)$ 来构建模型, 这样生成/采样的过程就可以直接通过 Euler 法完成. 但是我们没有讨论如何训练得到这个向量场. 接下来我们将讨论目前最有效的方法, **流匹配(Flow Matching)**.
### Marginal and Conditional Probability Paths
在介绍 Flow Matching 之前, 我们要引入一个新的概念, **概率路径(Probability Path)**. 直观来说, 概率路径就是在 $p_{init}$ 和 $p_{data}$ 之间插值, 由 $p_{init}$ 逐渐过渡到 $p_{data}$,我们可以把概率路径视为概率空间上的一条轨迹. 这看上去很简单, 但其实我们根本没法做到. 所以我们需要引入 **条件概率路径(Conditional Probability Path)**.
条件概率路径是一系列的分布 $p_t(x|z)$, 满足
$$p_0(\cdot|z) \sim p_{init}$$
$$p_1(\cdot|z) = \delta_z(\cdot)$$
$\delta_z$ 是 Dirac Delta 函数, 在这里表示从这个分布中采样只能采样到 $z$. 直观上来说, 条件概率路径由 $p_{init}$ 逐渐过渡到一个来自于 $p_{data}$ 的数据点. 
基于条件概率路径, 我们可以得到对应的**边缘概率路径(Marginal Probability Path)**
$$z \sim p_{data}, x \sim p_t(\cdot|z) \Rightarrow x \sim p_t(\cdot)$$
$$p_t(x) = \int_{z \sim p_{data}} p_{data}(z) p_t(x|z) $$
这里只使用了最基本的全概率公式.
对于边缘概率路径, 我们很容易就能知道 $p_0 = p_{init}$ 以及 $p_1 = p_{data}$.
对于一对 $p_{init}$ 和 $p_{data}$ 我们可以给出很多概率路径, 但是一般情况下我们使用 **Gaussian Path**. 
$p_{init} = N(0, I_d)$ 是标准高斯分布, 对于 $z \sim p_{data}$ 单个数据点, 高斯条件概率路径定义为 
$$p_t(x|z) = N(x; \alpha_t z, \beta_t^2 I_d)$$
其中 $\alpha_t$ 和 $\beta_t$ 是 满足$\alpha_0 = \beta_1 = 0$ 以及 $\alpha_1 = \beta_0 = 1$ 的 $t \to \mathbb{R}$ 函数. 很容易证明其满足概率路径的定义.
我们可以将从对应的边缘概率路径中采样表示为
$$z \sim p_{data}, \epsilon \sim N(0, I_d) \Rightarrow x = \alpha_t z + \beta_t \epsilon \sim p_t $$
### Marginal  and Conditional Vector Fields
概率路径事实上已经给出了我们想要的, 从 $p_{init}$ 到 $p_{data}$ 的路径 $X_t$ 所应该满足的分布 $p_t$, 并给出了在特定路径 高斯路径 下的具体形式. 接下来我们将从这个路径出发, 找到我们想要的向量场 $u_t^{target} (x)$. 
同样, 我们从 **条件向量场 (Conditional Vector Fields)** 开始. 我们希望条件向量场决定轨迹的分布要符合条件概率路径
$$X_0 \sim p_{init}, \frac{d}{dt} X_t = u_t^{target}(X_t | z) \Rightarrow X_t \sim p_t(\cdot|z)$$
条件向量场所决定的轨迹会坍缩到单一数据点 $z$, 但是如果我们将这些条件向量场联合起来, 就能得到 **边缘向量场 (Marginal Vector Fields)**
$$u_t^{target}(x) = \int_{z \sim p_{data}} u_t^{target}(x|z) \frac{p_t(x|z) p(z)}{p_t(x)}$$
可以证明这个边缘向量场所决定的轨迹是符合边缘概率路径的
$$X_0 \sim p_{init}, \frac{d}{dt} X_t = u_t^{target}(X_t) \Rightarrow X_t \sim p_t$$
这个边缘向量场 $u_t^{target}(t)$ 就是我们构造流模型所需要的.
接下来让我们给出 Gaussian Path 对应的边缘向量场.
首先, 我们可以得到满足 Gaussian Path 的向量场的流
$$\psi_t^{target}(x|z) = \alpha_t z + \beta_t x$$
这是因为
$$X_0 \sim N(0, I_d) \Rightarrow \psi_t^{target}(X_0|z) = \alpha_t z + \beta_t X_0 \sim p_t(\cdot|z) $$
那么由流的定义可以得到
$$\partial_t \psi_t(x|z) = u_t(\psi_t(x|z) | z)$$
$$\Rightarrow \dot{\alpha_t} z + \dot{\beta_t} x = u_t ((\alpha_t z + \beta_t x)|z)$$
$$\Rightarrow u_t (x|z) = \dot{\alpha_t} z + \dot{\beta_t} (\frac{x - \alpha_t}{\beta_t})$$
则有
$$u_t^{target} (x|z) = (\dot{\alpha_t} - \frac{\dot{\beta_t}}{\beta_t} \alpha_t) z + \frac{\dot{\beta_t}}{{\beta_t}} x$$

### Flow Matching Loss
很自然的, 我们定义 **流匹配损失 (Flow Matching Loss)**
$$L_{FM} = \mathbb{E}_{t \sim Unif, z \sim p_{data}, x \sim p_t(\cdot|z)} (||u_t^\theta (x) - u_t^{target}(x)||^2)$$
很快就能发现, 我们没法直接优化这个损失, 因为 $u_t^{target}(x)$ 并没有显式的表达. 但是 $u_t^{target}(x | z)$ 总是能够利用上一节中最后推导 Gaussian Path 下条件向量场类似的方法得到. 基于这个观察, 我们定义 **条件流匹配损失 (Conditional Flow Matching Loss)**
$$L_{CFM} = \mathbb{E}_{t \sim Unif, z \sim p_{data}, x \sim p_t(\cdot|z)} (||u_t^\theta (x) - u_t^{target}(x|z)||^2)$$
可以证明
$$\nabla_\theta L_{FM} = \nabla_\theta L_{CFM}$$
这样一来我们就能够训练符合我们需要的 $u_t^\theta (x)$ 了.
接下来我们看一个例子. 上一节的最后我们推导了 Gaussian path 对应的向量场, 在这一节我们主要关心一个特殊的 Gaussian path, ConOT path. 此时 
$$\alpha_t = t, \beta_t = 1 - t$$
则有
$$p_t(\cdot|z) = N(t z, (1 - t)^2 I_d)$$
$$\epsilon \sim N(0, I_d) \Rightarrow x_t = t z + (1 - t) \epsilon$$
此时, 条件流匹配损失为
$$L_{CFM} = \mathbb{E}_{t \sim Unif, z \sim p_{data}, x \sim p_t(\cdot|z)} (||u_t^\theta (x) - u_t^{target}(x|z)||^2)$$
$$= \mathbb{E}_{t \sim Unif, z \sim p_{data}, x \sim p_t(\cdot|z)} (||u_t^\theta (t z + (1 - t) \epsilon) - (z - \epsilon)||^2)$$
这样一来, 我们就能给出 ConOT path 下的完整算法
```python
for batch in datasets:
	Sample z from the dataset
	Sample t from Unif[0, 1]
	Sample eps from N(0, I_d)
	x = t * z + (1 - t) * eps
	L_cfm = |u_theta(x, t) - (z - eps)| ^ 2
	theta <- grad(L_cfm)
```
这就是 Flow Matching

### Prediction beyond Velocity
直接训练一个神经向量场是最直接的想法, 但是我们起码能够预测另外两个对象, 来获得完全等价的向量场: 噪声 $\epsilon$ 和 原数据 $z$.
如果我们用一个神经网络 $\epsilon_t^\theta (x)$ 去直接拟合每一步 $x = t z + (1 - t) \epsilon$ 中的噪声 $\epsilon$, 就能通过这个 $\epsilon_t^\theta (x)$ 构造完全等价的目标向量场
$$u_t^\theta(x) = z - \epsilon_t^\theta (x)$$
在生成采样时, 只要用 $x_t$ 表示 $z$
$$z = \frac{(x_t - (1 - t) \epsilon)}{t}$$
$$u_t^\theta(x) = \frac{1}{t} (x_t - \epsilon_t^\theta(x))$$
那么我们就能够使用 Euler 法进行模拟.
如果你之前了解过 Diffusion Models (比如 DDPM), 这正是这一类模型所做的任务. 它们将生成过程建模为加噪和去噪, 将目标设定为预测噪声, 但其潜在地蕴涵了一个向量场 (有一点不同, 扩散模型其实是一个 SDE 过程, 和我们的 ODE 有一些差别, 但其思想是类似的; 有关 SDE 的内容会在介绍 Score Matching 的章节中提及).
类似的, 我们可以直接去预测目标数据点 $z_t^\theta (x)$, 同样有其和向量场的关系
$$u_t^\theta(x) = z_t^\theta (x) - \epsilon$$
即
$$u_t^\theta(x) = \frac{z_t^\theta (x) - x}{1 - t}$$
由上述推导, 我们能够看出, 使神经网络预测向量场, 噪声, 甚至是真实数据都能够得到等价的结果. 类似的, 我们也能够将之前得到条件流匹配损失改写为对应的 向量场损失, 噪声损失和真实数据损失.
对于三种神经网络输出和三种对应的损失函数, 可以参考 [JiT](https://arxiv.org/abs/2511.13720) 一文, 其中对共九种组合做了实验 (我们在讨论隐空间和模型架构的时候仍然会提及这篇文章).