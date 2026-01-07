---
title: "G4Splat: Geometry-Guided Gaussian Splatting with Generative Prior"
date: 2025-10-14
authors: "Junfeng Ni, Yixin Chen, **Zhifei Yang**, Yu Liu, Ruijie Lu, Song-Chun Zhu, Siyuan Huang+"
venue: "arxiv"
paper: "https://arxiv.org/abs/2510.12099"
code: "https://github.com/DaLi-Jack/G4Splat"
image: "/images/g4splat.png"
selected: True
abstract: "Despite recent advances in leveraging generative prior from pre-trained diffusion models for 3D scene reconstruction, existing methods still face two critical limitations. First, due to the lack of reliable geometric supervision, they struggle to produce high-quality reconstructions even in observed regions, let alone in unobserved areas. Second, they lack effective mechanisms to mitigate multiview inconsistencies in the generated images, leading to severe shape–appearance ambiguities and degraded scene geometry. In this paper, we identify accurate geometry as the fundamental prerequisite for effectively exploiting generative models to enhance 3D scene reconstruction. We first propose to leverage the prevalence of planar structures to derive accurate metric-scale depth maps, providing reliable supervision in both observed and unobserved regions. Furthermore, we incorporate this geometry guidance throughout the generative pipeline to improve visibility mask estimation, guide novel view selection, and enhance multi-view consistency when inpainting with video diffusion models, resulting in accurate and consistent scene completion. Extensive experiments on Replica, ScanNet++, and DeepBlending show that our method consistently outperforms existing baselines in both geometry and appearance reconstruction, particularly for unobserved regions. Moreover, our method naturally supports single-view inputs and unposed videos, with strong generalizability in both indoor and outdoor scenarios with practical real-world applicability."
---