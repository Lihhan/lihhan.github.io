---
layout: post
title: "Diffusion Model 教程 - DDIM: 确定性采样"
subtitle: "从SDE到DreamFusion的完整推导"
date: 2025-02-16
author: Lihan
tags: [Diffusion, Deep Learning, 教程, LaTeX]
---

<p>Why model diffusion with the <em>solution</em> of an SDE—i.e. a <em>stochastic</em> process? Because an SDE captures the key feature of DDPM denoising: given the previous state, the next state is <em>not</em> deterministic; only its <em>distribution</em> is specified. Suppose all images possessing a certain attribute follow a latent distribution <span class="math inline">\(q\)</span>. We want a model that, when fed a noisy image <span class="math inline">\(x_t\)</span> at noise level <span class="math inline">\(t\)</span>, predicts <em>the distribution of</em> <span class="math inline">\(x_{t-1}\)</span>, not a single value. By sampling from that distribution at each step we eventually obtain a random draw from <span class="math inline">\(q\)</span>. If a large noise level <span class="math inline">\(T\)</span> converts any image to roughly standard normal, then by <span class="math inline">\(T\)</span> steps of prediction and sampling we can transform pure Gaussian noise into a valid sample from <span class="math inline">\(q\)</span>.</p><p>Once the forward relation <span class="math inline">\(x_t\!\leftrightarrow x_0\)</span> is available in closed form, the step‑by‑step noising is unnecessary; the time parameter <span class="math inline">\(t\)</span> merely controls the noise intensity. This observation leads to DDIM: because the <em>result</em> does not depend on <span class="math inline">\(p(x_t\mid x_{t-1})\)</span>, we can drop the “build‑and‑demolish’’ construction entirely.</p><p>In principle, even without an explicit <span class="math inline">\(p(x_t\mid x_{t-1})\)</span>, the conditional <span class="math inline">\(p(x_{t-1}\mid x_t,x_0)\)</span> is solvable; indeed, the solution set is larger and easier to characterize. All that is required is the <em>marginal‑consistency</em> condition <span class="math display">\[\int p(x_{t-1}\mid x_t,x_0)\,p(x_t\mid x_0)\,dx_t
  \;=\;
  p(x_{t-1}\mid x_0).\]</span> With undetermined coefficients we can solve this directly. More generally, assume <span class="math display">\[p(x_{t-1}\mid x_t,x_0)
  \;=\;
  \mathcal N\!\bigl(
     x_{t-1};
     \,\kappa_t x_t + \lambda_t x_0,\,
     \sigma_t^{2}I
  \bigr),\]</span> with <span class="math inline">\(\kappa_t,\lambda_t,\sigma_t\)</span> to be determined. Using <span class="math inline">\(p(x_{t-1}\mid x_0)\)</span> and <span class="math inline">\(p(x_t\mid x_0)\)</span> one obtains a <em>family</em> of solutions parameterized by the free variance <span class="math inline">\(\sigma_t\)</span>. Training is unaffected (the saved model is unchanged), but generation now has a tunable parameter <span class="math inline">\(\sigma_t\)</span>—the key novelty introduced by DDIM.</p><p>In the building‑demolishing metaphor, we know what the fully demolished building looks like <span class="math inline">\(\bigl(p(x_t\mid x_0)\)</span> and <span class="math inline">\(p(x_{t-1}\mid x_0)\bigr),\)</span> but not how each individual plank is removed <span class="math inline">\(\bigl(p(x_t\mid x_{t-1})\bigr)\)</span>. If <span class="math inline">\(x_t\)</span> lets us estimate <span class="math inline">\(x_0\)</span>, then teaching the model, given <span class="math inline">\((x_0,x_t)\)</span>, to recover the intermediate state <em>is</em> to learn every reverse step <span class="math inline">\(p(x_{t-1}\mid x_t)\)</span>—in other words, to “rebuild’’ the house one floor at a time without ever specifying the original demolition plan.</p>

