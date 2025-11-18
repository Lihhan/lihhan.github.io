---
layout: post
title: "Diffusion Model 教程 - Score Perspective: 分数视角"
subtitle: "从SDE到DreamFusion的完整推导"
date: 2025-02-14
author: Lihan
tags: [Diffusion, Deep Learning, 教程, LaTeX]
---

<p>The <em>score function</em> of a density is defined as its log‑likelihood gradient, <span class="math inline">\(s(x)=\nabla_x\log p(x)\)</span>, pointing in the direction of steepest increase in probability. In Langevin dynamics and diffusion models, this score acts as the drift term that “pulls’’ noisy samples back toward high‑density regions, as schematically illustrated below.</p><figure><img src="pics/DDPM_score.png" alt="Intuition: score‑based drift guides noisy particles toward the data manifold." /><figcaption aria-hidden="true">Intuition: score‑based drift guides noisy particles toward the data manifold.</figcaption></figure><p>Accordingly, the reverse process trains the network to approximate the score of each noisy image <span class="math inline">\(x_t\)</span>, effectively teaching it how to guide particles from <span class="math inline">\(p(x_{t+1})\)</span> back to <span class="math inline">\(p(x_t)\)</span>.</p>

