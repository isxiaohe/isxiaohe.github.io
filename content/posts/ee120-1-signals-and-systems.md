---
title: "信号系统(1) 信号与系统介绍"
date: 2025-06-03
draft: false
type: "post"
summary: "基于ee120. 第一部分, 介绍信号和系统, 以及离散的LTI系统及其性质"
tags: ["信号与系统", "数学", ]
---

## 信号
**信号**是一元或者多元函数, 一些典型例子包括: 声音信号(时间的函数), 数字图像信号(行列号的函数). 在本课程中我们更多考查一元函数, 且一般会认为这个变量是时间.

我们一般认为定义在实数域 $\mathbb{R}$ 上的信号是连续信号, 记作 $x(t)$ ; 定义在整数上的信号是离散信号, 记作 $x[n]$. 在大多数情况下, 基于上下文很容易推断我们在讨论连续还是离散的, 所以可能会出现用 $x(n)$ 表示离散信号的情况. 这是可以容忍的符号滥用(起码我可以忍受).

在很多情况下我们会先讨论离散信号的一些性质, 然后再将这些性质符合直觉地推广到连续信号(虽然并非总是如此), 按照这个逻辑, 我们可以串联起信号与系统这门课程中的大部分主题.

让我们先来讨论两个常见的离散信号: **Unit Impulse** 和 **Unit Step**
\[
    \text{Unit Impulse } \delta[n] = 
    \begin{cases}
        1 & \text{, } n = 0,\\
        0 & \text{otherwise}.
    \end{cases}    
\]

\[
    \text{Unit Step } u(n) = 
    \begin{cases}
        1 & \text{, } n \geq 0,\\
        0 & \text{otherwise}.
    \end{cases}    
\]

![](../assets/ee120-signal-system/1-unit-impulse.png)  
![](../assets/ee120-signal-system/1-unit-step.png)

有关这两个信号有一些很有意义的结论. 

首先, 对于任意的整数 $n$, 我们有 $u[n] = \Sigma_{k=-\infty}^n \delta[k] = \Sigma_{l=0}^\infty \delta[n-l]$.

事实上, 我们可以断言: **任意离散信号 $x[n]$, 是 $\delta[n]$ 的线性加和, 且 $x[n] = \Sigma_{k=-\infty}^\infty x[k]\delta[n-k]$**. 这个结论比较显然, 因为在右式中只有当 $k = n$ 时, $\delta[n-k]$ 才非零.

## 系统
**系统**被定义为由信号到信号的映射. 我们一般写作 $x \to H \to y$, 信号 $x$ 经系统 $H$ 变作了 $y$. $x, y$ 都可能是连续或者离散的. 

系统是极其多样的, 我们只要保证每一个输入信号都有唯一一个输出信号对应, 那这个信号变化就是一个系统.

