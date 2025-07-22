---
title: "Latent Variable Model -- VAE"
date: 2025-06-29
draft: false
type: "post"
summary: "介绍了VAE及其改进工作"
tags: ["VAE", "无监督学习", "unsupervised learning"]
---
## 为什么需要"隐变量模型"(Latent Variable Model)

无监督学习的重要任务是学习真实数据的分布, 从而能够做到

- 生成符合真实数据分布的新数据(数据的"生成")

- 提取某一真实数据的特征(数据的"压缩")

真实数据往往是复杂并且高维的, 所以我们希望能够用一个更简单的分布(比如高斯分布)来表示真实数据. 具体来说, 隐变量模型通过建立一个从简单分布(这个简单分布是可以预先选定的, 一般会选标准正态分布)到真实分布的变换来完成无监督学习的生成任务(即生成新数据):

$$
z \to x.
$$

也就是说, 我们希望得到这样一个模型, 我们从一个简单的分布(比如高斯分布)中采样 $z \sim p_Z(z)$, 通过这个模型能够得到一个条件分布 $p_\theta (x|z)$, 我们就能得到一个符合真实数据分布的新数据.

更确切的说, 我们可以得到如下的内容

- 采样
$$
z \sim p_Z(z), ~~~ x \sim p_\theta (x|z)
$$

- 似然函数
$$
p_\theta(x) = \sum p_Z(z) p_\theta(x|z)
$$

- 训练目标(极大对数似然)
$$
max_\theta \sum_i log ~ p_\theta(x^{(i)})
$$

## 隐变量模型的训练

### 重要性采样

我们的训练目标是

$$
max_\theta \sum_i log ~ p_\theta(x^{(i)}) = max_\theta \sum_i log ~ \sum_z p_Z(z) p_\theta(x|z)
$$

一个最简单而直接想法是, 我们先从 $p_Z(z)$ 中随机进行采样, 然后计算每一个样本的似然. 如果 $z$ 只能取有限个值, 那么我们可以对每一个值计算似然, 然后进行优化; 但是有的时候 $z$ 能够去无限个值, 那么我们只能从中采样有限个数据计算似然.

可这样会有起码两个问题

- 很难从 $p_Z(z)$ 中采样

- 直接从 $p_Z(z)$ 中采样只能得到很少的信息

让我们举个例子说明一下第二个问题. 比如真实数据有一千个聚簇, 随机采样得到的 $z$ 能够和某一类的数据对应上的概率已经很低; 而要采样得到的所有的 $z$ 能够和一千类中的每一类都有对应, 则需要非常大量的采样, 这些样本中的大部分无法和任何一类的真实数据对应上.

为了解决上面的问题, 从而使我们采样得到的 $z$ 能够带来尽可能多的信息, 我们提出重要性采样的方法. 它背后的基本考量是, 我们无法有效地从 $p_Z(z)$ 中进行采样, 那么我们提出一个新的且更容易处理的分布 $q(z)$, 直接从 $q(z)$ 中进行采样.

由于

$$
\begin{align*}
E_{z \sim p_Z(z)} (f(z))
&= \int p_Z(z) f(z) dz \\
&= \int \frac{q(z)}{q(z)} p_Z(z) f(z) dz \\
&= E_{z \sim q(z)} (\frac{p_Z(z)}{q(z)} f(z))
\end{align*} 
$$

于是乎, 我们的训练目标可以改写为

$$
\sum_i log ~ \sum_z p_Z(z) p_\theta(x|z) = \sum_i log \frac{1}{K} \sum_k \frac{p_Z(z_k^{(i)})}{q(z_k^{(i)})} p_\theta(x^{(i)}|z_k^{(i)})
$$

其中, $z_k^{(i)} \sim q(z_k^{(i)})$.

不难发现, $q(z)$ 其实可以是任意的分布. 我们希望从中采样得到的 $z$ 能够和真实数据有较好的匹配度, 所以可以直接设定

$$
q(z) = p_\theta(z|x^{(i)})
$$

我们希望 $q(z)$ 尽可能符合 $p_\theta(z|x^{(i)})$, 也就是

$$
\begin{align*}
min_{q(z)} KL(q(z)||p_\theta(z|x^{(i)}))
&= min_{q(z)} ~ E_{z \sim q(z)} log (\frac{q(z)}{p_\theta(z|x^{(i)})}) \\
&= min_{q(z)} ~ E_{z \sim q(z)} log (\frac{q(z)}{p_\theta(x^{(i)}|z) p_\theta(z) / p_\theta(x)}) \\
&= min_{q(z)} ~ E_{z \sim q(z)} (log ~ q(z) - log ~ p_\theta(z) - log ~ p_\theta(x^{(i)}|z)) + C
\end{align*}
$$

这最后得到的每一项都是可以求得的.

为了优化训练和推断, 我们并不会为每一个数据点都计算一次 $q(z)$, 而是也用一个神经网络来表示 $q_\phi(z | x^{(i)})$, 这个操作被称为"摊销"(Amortize).

上述也就可以改写为

$$
min_\phi KL(q_\phi(z|x^{(i)})||p_{\theta}(z|x^{(i)}))
$$

在训练过程中, 我们通过一个`encoder`网络来指导采样, 再将采样得到的隐变量通过另一个网络`decoder`得到样本. 这样, 我们的模型最终是这样的

$$
z \xrightleftarrows[p_{\theta}(x|z)]{q_\phi(z|x)} x
$$

在下一小节中, 我们将介绍损失函数的形式, 以及为什么这个优化目标是可行的.

### ELBO/VLB

论文[Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)提出了VLB(Variational lower bound), 并证明了将其作为模型最终的训练目标的可行性.

以下我们可以从两个角度来思考和证明.

1. **VLB的推导(1)**

    第一个推导VLB的方式使用了Jensen不等式

    $$
    log ~ E (z) \geq E (log ~ z)
    $$

    首先

    $$
    \begin{align*}
    \sum_i log ~ ~ p_{\theta}(x^{(i)})
    &= \sum_i log ~ (\sum_z ~ p_Z(z) p_{\theta}(x^{(i)}|z))\\
    &= \sum_i log ~ (\sum_z ~ \frac{q(z)}{q(z)} p_Z(z) p_{\theta}(x^{(i)}|z))\\
    &\geq E_{q(z)} (log ~ p(z) - log ~ q(z) + log ~ p_{\theta}(x^{(i)}|z))
    \end{align*}
    $$

    (最后一步用到Jensen不等式)

    由上述推导, 我们知道$E_{q(z)} (log ~ p(z) - log ~ q(z) + log ~ p_{\theta}(x^{(i)}|z))$ 是似然函数的下界, 称为 VLB(也称作ELBO).

    我们仍然希望最大化似然函数, 其实只需要最大化最后这个下界

    $$
    \begin{align*}
    &max_{\theta} \sum_i log ~ p_{\theta}(x^{(i)}) \\
    &= max_{\theta} E_{q(z)} (log ~ p(z) - log ~ q(z) + log ~ p_{\theta}(x^{(i)}|z)) \\
    &= max_{\theta} ~ max_\phi E_{q_\phi(z)} (log ~ p(z) - log ~ q_\phi(z|x^{(i)}) + log ~ p_{\theta}(x^{(i)}|z))
    \end{align*}
    $$

    而其中 $max_\phi E_{q_\phi(z)} (log ~ p(z) - log ~ q_\phi(z|x^{(i)}) + log ~ p_{\theta}(x^{(i)}|z))$ 的部分实际上就是在 $min_\phi KL(q_\phi(z|x^{(i)})||p_{\theta}(z|x^{(i)}))$ (我们在上一个部分推导过了).

2. **VLB的推导(2)**

    我们可以得到

    $$
    log ~ p(x) = E_{q_x(z)} (log ~ p(z) - log ~ q_x(z) + log ~ p(x|z)) + KL(q_x(z) || p(z|x))
    $$

    即似然函数和VLB相差的正是一个KL散度(KL散度是大于等于0的). 所以VLB确实是对数似然函数的下界.

    当 $q_x(z)$ 和 $p(z|x)$ 一致时(KL散度为0), 似然函数就是VLB.

通过上述两种不同方式我们都能够得到最终的优化目标是最大化VLB, 最小化KL散度.

值得注意的是, 我们无法直接得到 $p_{\theta}(z|x^{(i)})$, 所以在实际训练的过程中, 我们往往考虑$min_\phi \sum_i KL(q_\phi(z|x^{(i)})||p(z))$(而非$min_\phi \sum_i KL(q_\phi(z|x^{(i)})||p(z|x))$), 也就是让$q(z|x)$整体接近分布$p(z)$, 而$p(z)$是已知的. 这个操作是合理的, 因为

$$
VLB = E_{z \sim q(z|x)}(log ~ p_\theta(x|z)) - KL(q_\phi(z|x)||p(z)).
$$

而由 **推导(1)**, 最大化VLB间接最小化了$min_\phi \sum_i KL(q_\phi(z|x^{(i)})||p(z|x))$, 又由 **推导(2)** 可知, 这样做也会减小VLB和真实似然函数的差距.

有关VLB的更多内容可以参考[Importance Weighted Autoencoders](https://arxiv.org/abs/1509.00519)

这样一来我们构建的损失函数就是

$$
L_{VLB} = - E_{z \sim q(z|x)}(log ~ p_\theta(x|z)) + KL(q_\phi(z|x)||p(z)).
$$

### 重参数化(Reparameterization Trick)

回顾一下VAE的模型, 隐变量是基于$z \sim p(z)$采样得到的, 但采样这个操作并不可导, 梯度没办法回传给`encoder`. 在这里有一个技巧是重参数化, 将采样过程变得可导

$$
z \sim q_\phi (z|x), ~ z = \mu_\phi (x) + \sigma_\phi (x) \cdot \epsilon, ~ \epsilon \sim N(0, 1),
$$

或者写成

$$
z \sim N(\mu_\phi(x), \sigma_\phi (x)).
$$

具体的操作是只使用`decoder`来获取均值$\mu$和方差$\sigma$, 再通过添加一个随机噪声进行采样. 这个过程是可导的.

重参数化技巧可以推广到更为一般的形式. 还是针对采样 $z \sim q_\phi(\cdot|x)$, 可以对进行重参数化

$$
z = g(\phi, \epsilon), \epsilon \sim N(0, 1).
$$

这样一来, 对于函数 $f$

$$
E_{z \sim q_\phi(\cdot|x)}(f(z)) = E_{\epsilon \sim N(0, 1)} [f(g(\phi, \epsilon))]
$$

如果函数$f$是可导的, 那么我们有

$$
\nabla_\phi E_{z \sim q_\phi(\cdot|x)}(f(z)) = E_{\epsilon \sim N(0, 1)} [\nabla_\phi f(g(\phi, \epsilon))].
$$

在我们一开始讨论的情况下(也是一般情况下), $z = \mu_\phi(\cdot) + \epsilon_\phi(\cdot) \cdot \epsilon$; 而在VAE中, $f$就是`decoder`.

### 自编码器视角

如果上面的解释比较复杂, 其实可以从自编码器(AutoEncoder)的视角得到一些更直观的理解. 自编码器的意思是, 我们有一个`encoder`和一个`decoder`, 我希望真实数据经过`encoder`之后可以被压缩成一个更低维的信息, 而这个低维信息在经过`decoder`之后能被恢复成原始的真实数据. 从直观上理解, 一个好的`encoder`应该能够从原始数据中提取出少量但关键的信息(这些信息已经具备重建原始数据的潜力, 当然是十分关键的), 而一个好的`decoder`则能够有效扩展低维信息到人类能够理解的程度.

VAE的基本架构其实也是这样的. 但是简单的AutoEncoder的结果是非常不稳定的, 而VAE的观察是, 如果我们能够让真实数据的整体在通过`decoder`之后得到的低维信息整体接近一个"比较好"的分布(比如标准正态分布), 那么结果会稳定许多. 所以可以将之前得到的$L_{VLB}$分为两个部分, 一个保证AutoEncoder有效重建原始数据的重建项$- E_{z \sim q(z|x)}(log ~ p_\theta(x|z))$, 另一个是保证结果稳定的规范项$KL(q_\phi(z|x)||p(z))$.

而从AutoEncoder出发我们也能够更好理解VAE的训练和推断过程

$$
\begin{align*}
\text{Train} ~~~~ &x \xrightarrow[q_\phi(z|x)]{encoder} z \xrightarrow[p_\theta(x|z)]{decoder} x\\
\text{Inference} ~~~~ &z \sim p(z), ~ z \xrightarrow[p_\theta(x|z)]{decoder} x.
\end{align*}
$$

## 改进工作

### 