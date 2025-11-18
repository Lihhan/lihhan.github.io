---
layout: post
title: "Diffusion Model 教程 - Neural Network Score: 分数函数估计"
subtitle: "Diffusion入门: 从SDE到DreamFusion的完整推导"
date: 2025-02-12
author: Lihan
tags: [Diffusion, Deep Learning, 教程, LaTeX]
---

<p>Given the reverse‑time SDE in Eq. (13), the noise schedule <span class="math inline">\(\beta(t)\)</span> is known at every step; to complete the <em>generation</em> (or “building‑up”) process we need only the score <span class="math inline">\(\nabla_x\log p(x)\)</span>. Here <span class="math inline">\(p(x)\)</span> denotes the marginal distribution of <span class="math inline">\(x\)</span> at time <span class="math inline">\(t\)</span>. For a <em>linear</em> forward Markov process this marginal can be written in closed form, but doing so requires an average over <em>all</em> training samples <span class="math inline">\(x_0\)</span>, which is computationally expensive and does not generalize well. We therefore train a neural network to <em>directly</em> approximate <span class="math inline">\(\nabla_x\log p(x)\)</span>.</p><p>The diffusion process specifies the forward transition density <span class="math inline">\(p(x_{t+\Delta t}\mid x_t)\)</span>. Integrating these infinitesimal transitions in sequence gives <span class="math display">\[p(x_t\mid x_0) \;=\;
  \lim_{\Delta t\to0}
  \!\!\int\!\!\cdots\!\!\int
  p(x_t\mid x_{t-\Delta t})\,
  p(x_{t-\Delta t}\mid x_{t-2\Delta t})\,
  \cdots
  p(x_{\Delta t}\mid x_0)\,
  dx_{t-\Delta t}\!\cdots dx_{\Delta t}.\]</span></p><p>When the forward chain is linear, the expression above admits a closed‑form solution (not guaranteed if the chain is nonlinear). Consequently the marginal at time <span class="math inline">\(t\)</span> is <span class="math display">\[p(x_t)\;=\;
  \int p(x_t\mid x_0)\,p(x_0)\,dx_0
  \;=\;
  \mathbb{E}_{x_0}\!\bigl[p(x_t\mid x_0)\bigr],\]</span> and <span class="math display">\[\nabla_{x_t}\log p(x_t)
  \;=\;
  \frac{
    \mathbb{E}_{x_0}\!\bigl[
      p(x_t\mid x_0)\,
      \nabla_{x_t}\log p(x_t\mid x_0)
    \bigr]}
       {\mathbb{E}_{x_0}\!\bigl[p(x_t\mid x_0)\bigr]}.\]</span></p><div class="lemma"><p><span id="lem:weighted_mse" label="lem:weighted_mse">[lem:weighted_mse]</span> Let <span class="math inline">\(x\in\mathbb{R}^d\)</span> be a random vector with <span class="math inline">\(\mathbb{E}\|x\|^2&lt;
  \infty\)</span>, and let <span class="math inline">\(w\ge0\)</span> be a non‑negative random weight on the same space with <span class="math inline">\(\mathbb{E}[w]&gt;0\)</span>. For any fixed vector <span class="math inline">\(y\in\mathbb{R}^d\)</span>, define the weighted mean‑squared error <span class="math display">\[J(y)\;=\;\mathbb{E}\bigl[w\,\|y-x\|^2\bigr].\]</span> Then <span class="math inline">\(J(y)\)</span> is strictly convex in <span class="math inline">\(y\)</span> and attains its unique minimizer at <span class="math display">\[y^{\star}\;=\;\frac{\mathbb{E}[w\,x]}{\mathbb{E}[w]}.\]</span> When <span class="math inline">\(w\equiv1\)</span>, this reduces to the classical result <span class="math inline">\(y^{\star}=\mathbb{E}[x]\)</span>.</p></div><p>Choose <span class="math display">\[w = p(x_t\mid x_0),\quad
  x = \nabla_{x_t}\log p(x_t\mid x_0),\quad
  y = s_\theta(x_t,t),\]</span> and define the loss <span class="math display">\[\mathcal{L}(s_\theta)
  \;=\;
  \mathbb{E}_{x_0}\!\bigl[
    p(x_t\mid x_0)\,
    \bigl\|s_\theta(x_t,t) -
           \nabla_{x_t}\log p(x_t\mid x_0)\bigr\|^{2}
  \bigr].\]</span> By the lemma, this loss is minimized at <span class="math display">\[s_\theta^{\star}(x_t,t)
  \;=\;
  \frac{\mathbb{E}_{x_0}\!\bigl[
      p(x_t\mid x_0)\,
      \nabla_{x_t}\log p(x_t\mid x_0)\bigr]}
       {\mathbb{E}_{x_0}[p(x_t\mid x_0)]}
  \;=\;\nabla_{x_t}\log p(x_t).\]</span></p><p>Since the denominator merely rescales the loss, we drop it for simplicity. Expanding the outer expectation over <span class="math inline">\(x_0\)</span> yields <span class="math display">\[\mathcal{L}(s_\theta)
  \;=\;
  \int
    \mathbb{E}_{x_0\mid x_t}\!
      \bigl[
        \|s_\theta(x_t,t) -
          \nabla_{x_t}\log p(x_t\mid x_0)\|^{2}
      \bigr]
    p(x_t)\,dx_t
  \;=\;
  \mathbb{E}_{x_0,\,x_t\sim p(x_t\mid x_0)\,\tilde p(x_0)}
  \!\Bigl[
     \|\,s_\theta(x_t,t)-\nabla_{x_t}\log p(x_t\mid x_0)\|^{2}
  \Bigr].\]</span></p><h4 id="forwardprocess-reparameterization.">Forward‑process reparameterization.</h4><p>From the DDPM forward dynamics one can write <span class="math inline">\(x_t\)</span> in closed form: <span class="math display">\[\begin{aligned}
x_t &amp;= \alpha_t x_{t-1} + \beta_t\varepsilon_t
     = \cdots
     = (\alpha_t\!\cdots\!\alpha_1)\,x_0
       + \textstyle\sum_{k=1}^{t}\!
         \bigl(\alpha_t\!\cdots\!\alpha_{k+1}\bigr)\beta_k\varepsilon_k.
\end{aligned}\]</span> Because <span class="math inline">\((\alpha_t\cdots\alpha_1)^2
  + (\alpha_t\cdots\alpha_2)^2\beta_1^2
  + \cdots + \beta_t^{2}=1 ,\)</span> we may write <span class="math display">\[x_t = \tilde\alpha_t x_0 + \tilde\beta_t\varepsilon,
  \qquad \tilde\alpha_t^2+\tilde\beta_t^2=1,\]</span> with <span class="math inline">\(p(x_t\mid x_0)=\mathcal N\!\bigl(x_t;\,\tilde\alpha_t x_0,
                   \tilde\beta_t^{\,2}I\bigr).\)</span></p><p>Hence <span class="math display">\[\nabla_{x_t}\log p(x_t\mid x_0)
  = -\frac{1}{\tilde\beta_t^{\,2}}\bigl(x_t-\tilde\alpha_t x_0\bigr)
  = -\frac{1}{\tilde\beta_t}\,\varepsilon .\]</span></p><p>Let <span class="math inline">\(s_\theta(x_t,t)=-\varepsilon_\theta(x_t,t)/\bar\beta_t\)</span>; dropping the constant factor <span class="math inline">\(1/\bar\beta_t^{2}\)</span> gives the final training loss <span class="math display">\[\boxed{%
\mathcal{L}(\varepsilon_\theta)=
\mathbb{E}_{x_0\sim\tilde p(x_0),\,\varepsilon\sim\mathcal N(0,I)}
          \Bigl[
            \bigl\|
              \varepsilon_\theta\!\bigl(
                \bar\alpha_t x_0+\bar\beta_t\varepsilon,\;t\bigr)
              -\varepsilon
            \bigr\|^{2}
          \Bigr] }.\]</span></p><p>Minimizing this loss forces the network <span class="math inline">\(\varepsilon_\theta\)</span> to approximate the true score <span class="math inline">\(\nabla_{x_t}\log p(x_t)\)</span> at <em>every</em> time step <span class="math inline">\(t\)</span>.</p>

