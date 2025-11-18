---
layout: post
title: "Diffusion Model 教程 - Reverse SDE: 去噪过程"
subtitle: "从SDE到DreamFusion的完整推导"
date: 2025-02-11
author: Lihan
tags: [Diffusion, Deep Learning, 教程, LaTeX]
---

<p>Using probabilistic language, Eq. (6) can be generalized to the stochastic differential equation (SDE) <span class="math display">\[\mathrm{d}x \;=\; f_t(x)\,\mathrm{d}t
                 \;+\; g_t\,\mathrm{d}w ,\]</span> which likewise describes the forward noise‑adding Markov chain. Whether the chain is <em>linear</em> depends on whether <span class="math inline">\(f_t(x)\)</span> is linear in <span class="math inline">\(x\)</span>. The corresponding one‑step conditional density is <span class="math display">\[p\!\bigl(x_{t+\Delta t}\mid x_t\bigr)
  \;=\;
  \mathcal{N}\!\Bigl(
      x_{t+\Delta t};
      \,x_t + f_t(x_t)\,\Delta t,\;
      g_t^{2}\,\Delta t\,I
  \Bigr)
  \;\propto\;
  \exp\!\Bigl(
    -\frac{\lVert x_{t+\Delta t}-x_t-f_t(x_t)\,\Delta t\rVert^{2}}
          {2\,g_t^{2}\,\Delta t}
  \Bigr).\]</span> (The constant normalization factor is omitted.)</p><p>Following the DDPM philosophy—“learning to build by watching demolition’’— we ultimately wish to find the reverse density <span class="math inline">\(p(x_t\mid x_{t+\Delta t})\)</span>. Applying Bayes’ rule, <span class="math display">\[\begin{aligned}
  p\!\bigl(x_t\mid x_{t+\Delta t}\bigr)
  &amp;= \frac{p(x_{t+\Delta t}\mid x_t)\,p(x_t)}
          {p(x_{t+\Delta t})} \nonumber\\
  &amp;\propto
    \exp\!\Bigl(
      -\frac{\lVert x_{t+\Delta t}-x_t-f_t(x_t)\,\Delta t\rVert^{2}}
            {2\,g_t^{2}\,\Delta t}
      + \log p(x_t)-\log p(x_{t+\Delta t})
    \Bigr).\end{aligned}\]</span></p><p>Because <span class="math inline">\(\Delta t\)</span> is small, <span class="math inline">\(p(x_{t+\Delta t}\mid x_t)\)</span> is appreciable only when <span class="math inline">\(x_{t+\Delta t}\)</span> is close to <span class="math inline">\(x_t\)</span>; the same is true for the reverse density. Hence we expand <span class="math display">\[\log p(x_{t+\Delta t})
  \;\approx\;
  \log p(x_t)
  + (x_{t+\Delta t}-x_t)^{\!\top}\!\nabla_{x_t}\log p(x_t)
  + \Delta t\,\partial_t\log p(x_t).\]</span></p><p>Substituting and collecting terms, <span class="math display">\[p\!\bigl(x_t\mid x_{t+\Delta t}\bigr)\;\propto\;
  \exp\!\Bigl(
    -\frac
      {\bigl\lVert
          x_{t+\Delta t}-x_t-
          \bigl[f_t(x_t)-g_t^{2}\nabla_{x_t}\log p(x_t)\bigr]\Delta t
       \bigr\rVert^{2}}
      {2\,g_t^{2}\,\Delta t}
    + \mathcal{O}(\Delta t)
  \Bigr).\]</span></p><p>Letting <span class="math inline">\(\Delta t \to 0\)</span> gives <span class="math display">\[p\!\bigl(x_t\mid x_{t+\Delta t}\bigr)
  \;\propto\;
  \exp\!\Bigl(
    -\frac
      {\bigl\lVert
          x_{t+\Delta t}-x_t-
          \bigl[
            f_{t+\Delta t}(x_{t+\Delta t})
            -g_{t+\Delta t}^{2}\nabla_{x_{t+\Delta t}}\log p(x_{t+\Delta t})
          \bigr]\Delta t
       \bigr\rVert^{2}}
      {2\,g_{t+\Delta t}^{2}\,\Delta t}
  \Bigr).\]</span></p><p>Hence <span class="math inline">\(p(x_t\mid x_{t+\Delta t})\)</span> is approximately Gaussian with mean <span class="math display">\[x_{t+\Delta t}
  -\Bigl[
     f_{t+\Delta t}(x_{t+\Delta t})
     - g_{t+\Delta t}^{2}\nabla_{x_{t+\Delta t}}\log p(x_{t+\Delta t})
   \Bigr]\Delta t,\]</span> and covariance <span class="math inline">\(g_{t+\Delta t}^{2}\,\Delta t\,I\)</span>. Taking <span class="math inline">\(\Delta t\to0\)</span> recovers the reverse‑time SDE <span class="math display">\[\mathrm{d}x
  \;=\;
  \bigl[f_t(x)-g_t^{2}\nabla_x\log p(x)\bigr]\,\mathrm{d}t
  + g_t\,\mathrm{d}w .\]</span></p><p>For the linear forward Markov chain in Eq. (1), <span class="math inline">\(f_t(x)=-\tfrac12\beta(t)x\)</span> and <span class="math inline">\(g_t=\sqrt{\beta(t)}\)</span>, yielding <span class="math display">\[\boxed{\;
    \mathrm{d}x
    =\bigl[-\tfrac12\beta(t)x-\beta(t)\nabla_x\log p(x)\bigr]\,\mathrm{d}t
    + \sqrt{\beta(t)}\,\mathrm{d}w \; }.\]</span></p>

