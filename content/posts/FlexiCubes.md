---
title: "FlexiCubes"
date: 2025-08-25
draft: false
type: "post"
summary: "介绍了FlexiCubes, 一种新的等值面提取方式"
tags: ["Graphics", "Differential Rendering", "3D Representation"]
---

在阅读论文 [InstantMesh](https://arxiv.org/abs/2404.07191) 时, 发现其使用了一个比较新的 3D 表征方法 [FlexiCubes](https://arxiv.org/abs/2308.05371), 这个方法支持从表征中提取等值面, 进而得到非常好的 mesh 等其他形式的资产, 并能够直接用于通常的可微渲染框架中.  我们将先介绍 FlexiCubes 的动机和方法, 并从可微渲染的视角(主要是 [NVDIFFREC](https://arxiv.org/abs/2111.12503)) 简单了解其应用.

## (Dual) Marching Cubes

FlexiCubes 的基础是 [Dual Marching Cubes](https://people.eecs.berkeley.edu/~jrs/meshpapers/SchaeferWarren.pdf), 而后者是对于经典算法 Marching Cubes 的改进. 我们将从这一背景引入 FlexiCubes.

Marching Cubes 是经典的等值面提取方法. 在当前语境下, 我们主要关注的是如何从 SDF 场中提取表面. SDF (Signed Distance Function) 是一个函数 $s: \mathbb{R}^3 \to \mathbb{R}$, 其含义是空间中的某个点与离该点最近的表面的距离, 如果符号为正, 则说明该点在平面所属物体的外侧, 反之则在物体的内侧. 而重建表面就是要提取 SDF 为 0 的等值面 $\{x \in R^3 | s(x) = 0\}$. 

Marching Cubes 的方法是将空间均匀划分为有限数量的小立方体, 同时记录每个立方体顶点的 sdf. 可以证明, 对于立方体而言, 顶点上的 sdf 按正负号, 一共只有 15 种情况, 分别对应了 15 中不同的 sdf 零等值面和立方体的棱的相交情况. 那么对于每一个小立方体, 我们都可以很快得到到底哪些棱和 sdf 零等值面相交, 并得到立方体内部零等值面的近似情况. 而对于连接顶点 $x_a, x_b$ 的棱, 其与等值面的交点 $u_e$ 可以由插值的方法得到:

$$
u_e = \frac{x_a s(x_b) - x_b s(x_a)}{s(x_b) - s(x_a)}
$$

Marching Cubes 的缺点是, 重建表面只能由一些分布在离散立方体的边上的 0 值点得到, 无法很好地重建锐利部分. 为解决这一问题, 一个改进方法是 Dual Marching Cubes (这个方法事实上来自于 Dual Contouring , 但在此不过多介绍).

具体来说, Dual Marching Cubes 是在 Marching Cubes 已经提取的表面基础上, 进行进一步的细化. 对于每一个由 Dual Marching Cubes 得到的基础表面, 我们从中得到一个新的点(即对偶点, dual vertex) $v_d$ :

$$
v_d = \frac{1}{|V_E|}\sum_{u_e \in V_E} u_e
$$

其中 $V_E$ 是该表面顶点的集合.

![image.png](../assets/FlexiCubes/image.png)

Dual Marching Cubes 的每个立方体共有 22 种可能的表面情况, 这比单纯的 Marching Cubes 要更多.

## FlexiCubes Method

### Flexible Dual Vertex Positioning

为了使得对偶点的提取更加灵活, 以适应多变的三维模型, FlexiCubes 引入了一些参数来改进对偶点的定位.

首先, 为每个小立方体引入了顶点权重 $\alpha \in R^8$, 每个分量对应了在进行 Marching Cubes 提取等值面时某个顶点的权重. 改进后的 Marching Cubes 遵循公式:

$$
u_e = \frac{s(x_i) \alpha_i x_j - s(x_j) \alpha_j x_i}{s(x_i) \alpha_i - s(x_j) \alpha_j}
$$

同时, 为每个小立方体引入了边权重 $\beta \to R^{12}$, 每个分量对应了 Dual Marching Cubes 提取对偶点时, 某条边上的点的权重. 改进后的对偶点提取遵循公式:

$$
v_d = \frac{1}{\sum_{u_e \in V_E} \beta_e} \sum_{u_e \in V_E} u_e \beta_e
$$

对于 $\alpha, \beta$, 使用 $tanh( \cdot ) + 1$ 作为激活函数, 来保证权重都在一定的范围之内, 这样就能保证得到的所有顶点都在对应的小立体的凸包之内.

每一个立方体都有自己的一组 $\alpha, \beta$ (一共 $4+8=12$ 个参数), 所有的立方体都不共享(即便两个立方体共用顶点或者边, 相应的权重也不共享). 

![image1.png](../assets/FlexiCubes/image1.png)

### Mesh Extraction

等值面提取之后可以转换为多种形式的资产, 我们主要关心 mesh (triangle mesh). 与 Duel Marching Cubes类似, 由 FlexiCubes 得到的等值面是一些不共面的四边形, 需要一些合适的方法将这四边形划分为三角形.

为了得到更灵活的三维表征, FlexiCubes 没有使用基础的三角剖分, 而是为每一个小立方体引入一个一个参数 $\gamma$, 这个参数会被继承给这个小立方体产生的所有对偶点. 这样对于一个四边形, 我们可以在其中找到划分点 $\overline{v_d}$ :

$$
\overline{v_d} = \frac{\gamma_{c_1} \gamma_{c_3} (v_{c_1} + v_{c_3}) / 2 + \gamma_{c_2} \gamma_{c_4} (v_{c_2} + v_{c_4}) / 2}{\gamma_{c_1} \gamma_{c_3} + \gamma_{c_2} \gamma_{c_4}}
$$

这个划分点实际上就是对角线中点的加权和, 基于划分点可以将四边形划分为四个三角形. 

FlexiCubes 支持动态调整局部的分辨率, 为了方式发生拓扑上的错误, 当增加局部地区分辨率时, 应该保证细分出的孩子八叉树中, 每个小立方体顶点的 SDF 都来自于其双亲节点的插值.

### Regularizers

原论文提供了一些正则损失来规范 FlexiCubes.

第一个损失惩罚了对偶点和对偶点所在的基本平面(由改进 Marching Cubes 得到的)的顶点均值之间的距离:

$$
L_{dev} := \sum_{v \in V} MAD(\{|v - u_e|_2 : u_e \in N_v\})
$$

其中 $N_v$ 表示表示 $v$ 所在的基本平面, 而 $MAD(Y) = \frac{1}{|Y|}\sum_{y \in Y}|y - \overline{y}|$, 即一阶均差.

第二个损失惩罚所有小立方体内相邻顶点的 SDF 符号变化(参考了 [NVDIFFREC](https://arxiv.org/abs/2111.12503)). 定义 $\mathcal{E_g}$ 是所有的被边连接的立方体顶点对, 则有:

$$
L_{sign} := \sum_{(a, b) \in \mathcal{E_g}} \text{CrossEntropy} (\text{sigmoid}(s_a), \text{sign}(s_B))
$$