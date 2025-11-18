---
layout: post
title: "Diffusion Model 教程 - Prolific Dreamer: 高质量3D生成"
subtitle: "从SDE到DreamFusion的完整推导"
date: 2025-02-19
author: Lihan
tags: [Diffusion, Deep Learning, 教程, LaTeX]
---

<p>The previous section (DreamFusion) trains a <em>single</em> NeRF parameter set <span class="math inline">\(\theta\)</span> via diffusion and the SDS loss, but such a point estimate can suffer from <em>low diversity</em>. Inspired by 2‑D diffusion, where one first learns a <em>distribution</em> over images and then samples from it, we ask: can we likewise learn a <em>distribution</em> <span class="math inline">\(\mu\)</span> over 3‑D parameters <span class="math inline">\(\theta\)</span> and then sample a diverse <span class="math inline">\(\theta\)</span> from <span class="math inline">\(\mu\)</span>? This is the core idea behind <strong>ProlificDreamer</strong>.</p><h2 id="principle-overview">Principle Overview</h2><h3 id="variational-score-distillation-vsd">Variational Score Distillation (VSD)</h3><p>For a text prompt <span class="math inline">\(y\)</span> there exists a distribution of all 3‑D scenes consistent with that prompt. Denote the distribution of NeRF parameters by <span class="math inline">\(\mu(\theta\mid y)\)</span>. Rendering with viewpoint <span class="math inline">\(c\)</span> yields <span class="math inline">\(x_0=g(\theta,c)\)</span>. Let <span class="math inline">\(q^{\mu}_{0}(x_0\mid c,y)\)</span> be the image distribution obtained by sampling <span class="math inline">\(\theta\!\sim\!\mu(\cdot\mid y)\)</span> and rendering at <span class="math inline">\(c\)</span>. Let <span class="math inline">\(p_{0}(x_0\mid y)\)</span> be the distribution produced by a frozen text‑to‑image diffusion model conditioned on <span class="math inline">\(y\)</span>.</p><p>We wish to minimize their KL divergence: <span class="math display">\[\min_{\mu}
  D_{\mathrm{KL}}\!\bigl(
     q^{\mu}_{0}(x_{0}\mid y)\,\|\,p_{0}(x_{0}\mid y)
  \bigr),\]</span> a <em>classical variational‐inference</em> objective.</p><p>To ensure closeness <em>at every noise level</em>, we instead minimize the weighted KL along the entire noising trajectory: <span class="math display">\[\mu^{\!*}
  =\arg\min_{\mu}
     \mathbb E_{t,c}\!
     \Bigl[
       \tfrac{\sigma_t}{\alpha_t}\,w(t)\,
       D_{\mathrm{KL}}\!\bigl(
          q^{\mu}_{t}(x_{t}\mid c,y)\,\|\,p_{t}(x_{t}\mid y)
       \bigr)
     \Bigr].\]</span></p><h3 id="updating-mu-via-particlebased-variational-inference">Updating <span class="math inline">\(\mu\)</span> via Particle‑Based Variational Inference</h3><p>Treat <span class="math inline">\(n\)</span> parameter sets <span class="math inline">\(\{\theta_i\}_{i=1}^{n}\)</span> as <em>particles</em> representing <span class="math inline">\(\mu\)</span> (<span class="math inline">\(n\!=\!4\)</span> in the original paper). Gradient descent with step size <span class="math inline">\(\eta\!\to\!0\)</span> leads to an ODE: <span class="math display">\[\dot\theta_\tau
  =-\nabla_\theta L(\theta_\tau),\]</span> known here as the <em>Wasserstein gradient flow of VSD</em>. Starting from <span class="math inline">\(\theta_0\!\sim\!\mu_0(\theta\mid y)\)</span> and integrating the ODE yields the optimal distribution <span class="math inline">\(\mu_\tau\)</span> as <span class="math inline">\(\tau\!\to\!\infty\)</span>.</p><p>Define <span class="math display">\[\frac{d\theta_\tau}{d\tau}
=
-\mathbb{E}_{t,\varepsilon,c}\!\left[
  \omega(t)\Bigl(
    -\sigma_t\nabla_{\mathbf x_t}\log p_t(\mathbf x_t\mid y)
    +\sigma_t\nabla_{\mathbf x_t}\log q^{\mu_\tau}_t(\mathbf x_t\mid c,y)
  \Bigr)
  \frac{\partial g(\theta_\tau,c)}{\partial\theta_\tau}
\right].\]</span></p><p>Real‑image scores use the frozen diffusion model <span class="math inline">\(\epsilon_{\mathrm{pretrain}}\)</span>; rendered‑image scores use a learnable network <span class="math inline">\(\epsilon_\phi\)</span> (a lightweight UNet or LoRA‑tuned clone of the pretrained network). Alternating updates yield <span class="math display">\[\boxed{
  \nabla_{\theta}\mathcal L_{\mathrm{VSD}}(\theta)
  =\mathbb E_{t,\varepsilon,c}\!
    \Bigl[
      \omega(t)\bigl(
        \epsilon_{\mathrm{pretrain}}(\mathbf x_t,t,y)
        -\epsilon_{\phi}(\mathbf x_t,t,c,y)
      \bigr)
      \frac{\partial g(\theta,c)}{\partial\theta}
    \Bigr].
  }\]</span> SDS is the special case where <span class="math inline">\(\mu\)</span> is a Dirac delta.</p><h2 id="practical-workflow-pseudocode">Practical Workflow (Pseudo‑Code)</h2><p><strong>Inputs:</strong> number of particles <span class="math inline">\(n\)</span>, prompt <span class="math inline">\(y\)</span>, frozen score <span class="math inline">\(\epsilon_{\mathrm{pretrain}}\)</span>, learning rates <span class="math inline">\(\eta_1,\eta_2\)</span>, renderer <span class="math inline">\(g(\theta,c)\)</span>, noise schedule <span class="math inline">\(\{\alpha_t,\sigma_t,\omega(t)\}\)</span>.</p><p><strong>Initialization:</strong> <span class="math inline">\(\{\theta^{(i)}_0\}_{i=1}^n\sim\mathrm{InitPrior},
\quad \phi_0\leftarrow\mathrm{InitWeights}.\)</span></p><p><strong>Main Loop:</strong> for <span class="math inline">\(k=0,\dots\)</span> until convergence</p><ol><li><p>Sample particle index and camera: <span class="math inline">\(i_k\!\sim\!\mathrm{Uniform}\{1,\dots,n\},~
          c_k\!\sim\!p(c).\)</span></p></li><li><p>Render clean image: <span class="math inline">\(x^{(k)}_0=g(\theta^{(i_k)}_k,c_k).\)</span></p></li><li><p>Noise it: <span class="math inline">\(t_k\!\sim\!\mathrm U(0,1),~
          \varepsilon_k\!\sim\!\mathcal N(0,I),~
          x^{(k)}_{t_k}=\alpha_{t_k}x^{(k)}_0+\sigma_{t_k}\varepsilon_k.\)</span></p></li><li><p>Compute real and rendered scores: <span class="math inline">\(\hat\varepsilon^{\mathrm{real}}_k
           =\epsilon_{\mathrm{pretrain}}(x^{(k)}_{t_k},t_k,y),~
          \hat\varepsilon^{\mathrm{rend}}_k
           =\epsilon_{\phi_k}(x^{(k)}_{t_k},t_k,c_k,y).\)</span></p></li><li><p>Update geometry: <span class="math display">\[\theta^{(i_k)}_{k+1}
          =\theta^{(i_k)}_k
           -\eta_1\,
             \omega(t_k)
             \bigl(\hat\varepsilon^{\mathrm{real}}_k
                  -\hat\varepsilon^{\mathrm{rend}}_k\bigr)
             \frac{\partial g(\theta^{(i_k)}_k,c_k)}{\partial\theta^{(i_k)}}.\]</span></p></li><li><p>Update <span class="math inline">\(\phi\)</span>: minimize <span class="math inline">\(\|\epsilon_{\phi_k}(x^{(k)}_{t_k},t_k,c_k,y)-\varepsilon_k\|_2^2\)</span> with step <span class="math inline">\(\eta_2\)</span>.</p></li></ol><p>Return the final particle set <span class="math inline">\(\{\theta^{(i)}_{\mathrm{final}}\}_{i=1}^n\)</span> and score network <span class="math inline">\(\phi_{\mathrm{final}}\)</span>.</p>

