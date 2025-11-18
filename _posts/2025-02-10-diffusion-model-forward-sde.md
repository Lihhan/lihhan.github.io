---
layout: post
title: "Diffusion Model 教程 - Forward SDE: 从离散到连续"
subtitle: "从SDE到DreamFusion的完整推导"
date: 2025-02-10
author: Lihan
tags: [Diffusion, Deep Learning, 教程, LaTeX]
---

<p>During the noise‑adding stage of a <strong>DDPM</strong>, every time‑step applies the following discrete Markov chain:</p><p><span class="math display">\[x_i \;=\; \sqrt{1-\beta_i}\,x_{i-1}
           \;+\; \sqrt{\beta_i}\,\varepsilon_{i-1},
  \qquad i = 1,\dots, N .\]</span></p><p>To turn this into a continuous process, we let the discrete interval <span class="math inline">\(\Delta t\)</span> shrink to zero-equivalently, we consider the limit <span class="math inline">\(N \to \infty\)</span> of the Markov chain.</p><p>Before taking the limit, introduce an auxiliary noise scale <span class="math inline">\(\{\tilde{\beta}_i = N\,\beta_i\}_{i=1}^{N}\)</span> and rewrite</p><p><span class="math display">\[x_i \;=\;
  \sqrt{1-\dfrac{\tilde{\beta}_i}{N}}\,x_{i-1}
  \;+\;
  \sqrt{\dfrac{\tilde{\beta}_i}{N}}\,\varepsilon_{i-1},
  \qquad i = 1,\dots,N .\]</span></p><p>As <span class="math inline">\(N\to\infty\)</span>, the sequence <span class="math inline">\(\{\tilde{\beta}_i\}_{i=1}^{N}\)</span> becomes a continuous schedule <span class="math inline">\(\beta(t)\)</span> on <span class="math inline">\(t\in[0,1]\)</span>. Set <span class="math display">\[\Delta t = \frac{1}{N}.\]</span></p><p>At each <span class="math inline">\(\Delta t\)</span>, the continuous functions <span class="math inline">\(\beta(t),\,x(t),\,\varepsilon(t)\)</span> coincide with their discrete counterparts:</p><p><span class="math display">\[\beta\!\bigl(\tfrac{i}{N}\bigr)=\tilde{\beta}_i,\qquad
  x\!\bigl(\tfrac{i}{N}\bigr)=x_i,\qquad
  \varepsilon\!\bigl(\tfrac{i}{N}\bigr)=\varepsilon_i .\]</span></p><p>For <span class="math inline">\(t\in\{0,\,\tfrac1N,\dots,\,\tfrac{N-1}{N}\}\)</span> and <span class="math inline">\(t+\Delta t = t+\tfrac1N\)</span>, we rewrite the update with continuous notation:</p><p><span class="math display">\[\begin{aligned}
  x(t+\Delta t)
  &amp;= \sqrt{1-\beta(t+\Delta t)\,\Delta t}\;x(t)
     + \sqrt{\beta(t+\Delta t)\,\Delta t}\;\varepsilon(t) \\
  &amp;\approx x(t) - \tfrac12 \beta(t+\Delta t)\,\Delta t\,x(t)
     + \sqrt{\beta(t+\Delta t)\,\Delta t}\;\varepsilon(t) \\
  &amp;\approx x(t) - \tfrac12 \beta(t)\,\Delta t\,x(t)
     + \sqrt{\beta(t)\,\Delta t}\;\varepsilon(t).\end{aligned}\]</span></p><p>(The second line holds when <span class="math inline">\(\Delta t \ll 1\)</span>.) Thus</p><p><span class="math display">\[x(t+\Delta t) - x(t) \;\approx\;
  -\tfrac12 \beta(t)\,\Delta t\,x(t)
  + \sqrt{\beta(t)\,\Delta t}\;\varepsilon(t).\]</span></p><p>Letting <span class="math inline">\(\Delta t \to 0\)</span> gives the stochastic differential equation (SDE)</p><p><span class="math display">\[\begin{aligned}
  \boxed{\,
    \mathrm{d}x = -\tfrac12 \beta(t)\,x\,\mathrm{d}t
      + \sqrt{\beta(t)}\,\mathrm{d}w\,
  }.\end{aligned}\]</span></p>

