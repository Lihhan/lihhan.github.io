---
layout: post
title: "Diffusion Model 教程 - Reverse Discrete: 离散化采样"
subtitle: "Diffusion入门: 从SDE到DreamFusion的完整推导"
date: 2025-02-13
author: Lihan
tags: [Diffusion, Deep Learning, 教程, LaTeX]
---

<p>Sections 1 and 2 showed that a trained DDPM can be interpreted as a stochastic process that models the <em>reverse</em> (denoising) dynamics. Here we discretize those continuous dynamics and make explicit how to sample <span class="math inline">\(x_{t-1}\)</span> from a given <span class="math inline">\(x_t\)</span>.</p><p>By Bayes’ theorem <span class="math display">\[p(x_{t-1}\mid x_t)
  \;=\;
  \frac{p(x_t\mid x_{t-1})\,p(x_{t-1})}{p(x_t)} .\]</span> Because the marginals <span class="math inline">\(p(x_{t-1})\)</span> and <span class="math inline">\(p(x_t)\)</span> are unknown in closed form, this expression is not directly usable (the continuous‑time derivation in Section 2 sidestepped them by taking <span class="math inline">\(\Delta t\!\to\!0\)</span>).</p><p>Instead, condition on the (unknown) clean image <span class="math inline">\(x_0\)</span>: <span class="math display">\[p(x_{t-1}\mid x_t,x_0)
  \;=\;
  \frac{p(x_t\mid x_{t-1},x_0)\,p(x_{t-1}\mid x_0)}
       {p(x_t\mid x_0)} .\]</span> Now every factor is known analytically, yielding <span class="math display">\[p(x_{t-1}\mid x_t,x_0)
  \;=\;
  \mathcal{N}\!\Bigl(
      x_{t-1}\!;
      \,
      \frac{\alpha_t\bar\beta_{t-1}^{\,2}}{\bar\beta_t^{\,2}}\,
      x_t
      +\frac{\bar\alpha_{t-1}\beta_t^{2}}{\bar\beta_t^{\,2}}\,
      x_0,\;
      \frac{\bar\beta_{t-1}^{\,2}}{\bar\beta_t^{\,2}}\,I
  \Bigr).\]</span></p><p>Because <span class="math inline">\(x_0\)</span> is unavailable during inference, we replace it by a learned predictor <span class="math inline">\(\hat{\mu}(x_t)\)</span>. If <span class="math inline">\(\hat{\mu}(x_t)\)</span> is trained with the loss <span class="math inline">\(\lVert x_0-\hat{\mu}(x_t)\rVert^2\)</span>, then <span class="math display">\[p(x_{t-1}\mid x_t)
  \;\approx\;
  p\!\bigl(x_{t-1}\mid x_t,x_0=\hat{\mu}(x_t)\bigr)
  =\mathcal{N}\!\Bigl(
      x_{t-1}\!;\,
      \tfrac{\alpha_t\bar\beta_{t-1}^{\,2}}{\bar\beta_t^{\,2}}\,x_t
      + \tfrac{\bar\alpha_{t-1}\beta_t^{2}}{\bar\beta_t^{\,2}}\,
        \hat{\mu}(x_t),\;
      \tfrac{\bar\beta_{t-1}^{\,2}}{\bar\beta_t^{\,2}}\,I
    \Bigr).\]</span></p><p>Because <span class="math inline">\(x_t=\bar\alpha_t x_0+\bar\beta_t\varepsilon\)</span>, one may solve for <span class="math inline">\(x_0\)</span> as <span class="math inline">\(x_0=\tfrac1{\bar\alpha_t}(x_t-\bar\beta_t\varepsilon)\)</span>. This motivates <span class="math display">\[\hat{\mu}(x_t)=
  \frac{1}{\bar\alpha_t}\Bigl(
      x_t-\bar\beta_t\,\varepsilon_\theta(x_t,t)
  \Bigr),\]</span> where <span class="math inline">\(\varepsilon_\theta\)</span> is the usual noise‑predicting UNet.</p><p>Training <span class="math inline">\(\varepsilon_\theta\)</span> with <span class="math display">\[\lVert x_0-\hat{\mu}(x_t)\rVert^{2}
  =\frac{\bar\beta_t^{\,2}}{\bar\alpha_t^{\,2}}
    \Bigl\lVert
      \varepsilon-\varepsilon_\theta(
        \bar\alpha_t x_0+\bar\beta_t\varepsilon,\,t)
    \Bigr\rVert^{2}\]</span> reduces (up to a scalar) to the simplified DDPM loss.</p><p>Substituting <span class="math inline">\(\hat{\mu}(x_t)\)</span> back yields <span class="math display">\[p(x_{t-1}\mid x_t)
  \;\approx\;
  \mathcal N\!\Bigl(
     x_{t-1}\!;
     \,
     \frac1{\alpha_t}\Bigl(
       x_t-\tfrac{\beta_t^{2}}{\bar\beta_t}
             \varepsilon_\theta(x_t,t)
     \Bigr),\;
     \frac{\bar\beta_{t-1}^{\,2}\beta_t^{2}}{\bar\beta_t^{\,2}}\,I
  \Bigr),\]</span> so that a single sampling step is <span class="math display">\[x_{t-1}
  =\frac1{\alpha_t}\Bigl(
      x_t-\tfrac{\beta_t^{2}}{\bar\beta_t}\,
      \varepsilon_\theta(x_t,t)
    \Bigr)
   + \frac{\bar\beta_{t-1}\beta_t}{\bar\beta_t}\;\varepsilon,
   \qquad \varepsilon\sim\mathcal N(0,I).\]</span></p>

