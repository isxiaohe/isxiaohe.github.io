---
title: "Latent Variable Model -- VAE"
date: 2025-06-29
draft: false
type: "post"
summary: "介绍了VAE及其改进工作"
tags: ["VAE", "无监督学习", ]
---
## 为什么需要"隐变量模型"(Latent Variable Model)
无监督学习的重要任务是学习真实数据的分布, 从而能够做到

- 生成符合真实数据分布的新数据(数据的"生成")
- 提取某一真实数据的特征(数据的"压缩")
真实数据往往是复杂并且高维的, 所以我们希望能够用一个更简单的分布(比如高斯分布)来表示真实数据. 具体来说, 隐变量模型通过建立一个从简单分布到真实分布的变换来完成无监督学习的生成任务(即生成新数据):
    \[
        z \to x
    \]

也就是说, 我们希望得到这样一个模型, 我们从一个简单的分布(比如高斯分布)中采样 $z \sim p_Z(z)$, 通过这个模型能够得到一个条件分布 $p_\theta (x|z)$, 我们就能得到一个符合真实数据分布的新数据.

更确切的说, 我们可以得到如下的内容

- 采样
    \[
        z \sim p_Z(z), ~~~ x \sim p_\theta (x|z)
    \]
- 似然函数
    \[
        p_\theta(x) = \sum p_Z(z) p_\theta(x|z)    
    \]
- 训练目标(极大对数似然)
    \[
        max_\theta \sum_i log ~ p_\theta(x^{(i)})    
    \]

## 隐变量模型的训练
### 重要性采样
我们的训练目标是
    \[
        max_\theta \sum_i log ~ p_\theta(x^{(i)}) = max_\theta \sum_i log ~ \sum_z p_Z(z) p_\theta(x|z)
    \]
一个最简单而直接想法是, 我们先从 $p_Z(z)$ 中随机进行采样, 然后计算每一个样本的似然. 如果 $z$ 只能取有限个值, 那么我们可以对每一个值计算似然, 然后进行优化; 但是有的时候 $z$ 能够去无限个值, 那么我们只能从中采样有限个数据计算似然.

可这样会有起码两个问题
- 很难从 $p_Z(z)$ 中采样
- 直接从 $p_Z(z)$ 中采样只能得到很少的信息

让我们举个例子说明一下第二个问题. 比如真实数据有一千个聚簇, 随机采样得到的 $z$ 能够和某一类的数据对应上的概率已经很低; 而要采样得到的所有的 $z$ 能够和一千类中的每一类都有对应, 则需要非常大量的采样, 这些样本中的大部分无法和任何一类的真实数据对应上.

为了解决上面的问题, 从而使我们采样得到的 $z$ 能够带来尽可能多的信息, 我们提出重要性采样的方法. 它背后的基本考量是, 我们无法有效地从 $p_Z(z)$ 中进行采样, 那么我们提出一个新的且更容易处理的分布 $q(z)$, 直接从 $q(z)$ 中进行采样.

由于
    \[
        E_{z \sim p_Z(z)} (f(z)) = \int p_Z(z) f(z) dz 
        \\ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        = \int \frac{q(z)}{q(z)} p_Z(z) f(z) dz 
        \\ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        = E_{z \sim q(z)} (\frac{p_Z(z)}{q(z)} f(z))
    \]

于是乎, 我们的训练目标可以改写为
    \[
        \sum_i log ~ \sum_z p_Z(z) p_\theta(x|z) = \sum_i log \frac{1}{K} \sum_k \frac{p_Z(z_k^{(i)})}{q(z_k^{(i)})} p_\theta(x^{(i)}|z_k^{(i)})
    \]
其中, $z_k^{(i)} \sim q(z_k^{(i)})$.

不难发现, $q(z)$ 其实可以是任意的分布. 我们希望从中采样得到的 $z$ 能够和真实数据有较好的匹配度, 所以可以直接设定
    \[
        q(z) = p_\theta(z|x^{(i)})
    \]

我们希望 $q(z)$ 尽可能符合 $p_\theta(z|x^{(i)}$, 也就是
    \[
        min_{q(z)} KL(q(z)||p_\theta(z|x^{(i)}))
        \\ = min_{q(z)} ~ E_{z \sim q(z)} log (\frac{q(z)}{p_\theta(z|x^{(i)})})
        \\ = min_{q(z)} ~ E_{z \sim q(z)} log (\frac{q(z)}{p_\theta(x^{(i)}|z) p_\theta(z) / p_\theta(x)}) 
        \\ ~
        \\ = min_{q(z)} ~ E_{z \sim q(z)} (log ~ q(z) - log ~ p_\theta(z) - log ~ p_\theta(x^{(i)}|z)) + C
    \]
这最后得到的每一项都是可以求得的.

为了优化训练和推断, 我们并不会为每一个数据点都计算一次 $q(z)$, 而是也用一个神经网络来表示 $q_\phi(z | x^{(i)})$, 这个操作被称为"摊销"(Amortize).

上述也就可以改写为
    \[
        min_\phi KL(q_\phi(z|x^{(i)})||p_\theta(z|x^{(i)}))
    \]

这样, 我们的模型最终是这样的
    \[
        z \xrightleftarrows[p_\theta(x|z)]{q_\phi(z|x)} x
    \]

### VLB与VAE
论文[Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)提出了VLB(Variational lower bound), 并证明了将其作为模型最终的训练目标的可行性.



#### VLB的推导(1)
第一个推导VLB的方式使用了Jensen不等式
    \[
        log ~ E (z) \geq E (log ~ z)
    \]

首先
    \[
        \sum_i log ~ ~ p_\theta(x^{(i)}) \\
        = \sum_i log ~ (\sum_z ~ p_Z(z) p_\theta(x^{(i)}|z))\\
        = \sum_i log ~ (\sum_z ~ \frac{q(z)}{q(z)} p_Z(z) p_\theta(x^{(i)}|z))\\
        \geq E_{q(z)} (log ~ p(z) - log ~ q(z) + log ~ p_\theta(x^{(i)}|z))\\ ~ \\
        \text{最后一步用到Jensen不等式}
    \]
$E_{q(z)} (log ~ p(z) - log ~ q(z) + log ~ p_\theta(x^{(i)}|z))$ 是似然函数的下界, 我们称为 VLB

我们仍然希望最大似然函数, 其实只需要最大最后这个下界
    \[
        max_\theta \sum_i log ~ p_theta(x^{(i)}) \\
        = max_theta E_{q(z)} (log ~ p(z) - log ~ q(z) + log ~ p_\theta(x^{(i)}|z)) \\
        ~ \\
        = max_\theta ~ max_\phi E_{q_\phi(z)} (log ~ p(z) - log ~ q_\phi(z|x^{(i)}) + log ~ p_\theta(x^{(i)}|z)) \\
    \]
而其中 $max_\phi E_{q_\phi(z)} (log ~ p(z) - log ~ q_\phi(z|x^{(i)}) + log ~ p_\theta(x^{(i)}|z))$ 的部分实际上就是在 $min_\phi KL(q_\phi(z|x^{(i)})||p_\theta(z|x^{(i)}))$ (我们在上一个部分推导过了).

#### VLB的推导(2)
我们可以得到
    \[
        log p(x) = E_{q_x(z)} (log ~ p(z) - log ~ q_x(z) + log ~ p(x|z)) + KL(q_x(z) || p(z|x))
    \]
即似然函数和VLB相差的正是一个KL散度(KL散度是大于等于0的).

当 $q_x(z)$ 和 $p(z|x)$ 一致时(KL散度为0), 似然函数就是VLB.

通过上述两种不同方式我们都能够得到最终的优化目标是最大化VLB, 最小化KL散度.
