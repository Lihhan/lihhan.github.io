---
layout: post
title: "Diffusion Model 教程 - DreamFusion: 3D生成"
subtitle: "从SDE到DreamFusion的完整推导"
date: 2025-02-18
author: Lihan
tags: [Diffusion, Deep Learning, 教程, LaTeX]
---

<p>Diffusion models have achieved impressive text‑to‑<em>image</em> generation, yet text‑to‑<em>3‑D</em> remains difficult because large text–shape pairs are scarce. <em>DreamFusion: Text‑to‑3D Using 2‑D Diffusion</em> remedies this by using a <em>pre‑trained text‑to‑image</em> diffusion model to supervise a 3‑D generator.</p><h2 id="conceptual-overview">Conceptual Overview</h2><p>Let <span class="math inline">\(\theta\)</span> denote the parameters of a volume renderer <span class="math inline">\(g(\cdot)\)</span> (e.g. a Gaussian NeRF). Render <span class="math inline">\(x=g(\theta)\)</span>, feed it into a frozen U‑Net, and define the diffusion loss <span class="math display">\[\mathcal L_{\text{Diff}}(\varphi,x=g(\theta))
  =\mathbb E_{t,\varepsilon}\!
    \bigl[
      \|\,
        \hat\varepsilon_{\varphi}(\alpha_t g(\theta)+\sigma_t\varepsilon;\,y,t)
        -\varepsilon
      \|^{2}
    \bigr],\]</span> where <span class="math inline">\(\hat\varepsilon_\varphi\)</span> is the frozen U‑Net predictor. If <span class="math inline">\(g(\theta)\)</span> produces photorealistic pixels, the U‑Net predicts noise well and the loss is low; otherwise it is high.</p><p>With <span class="math inline">\(z_t=\alpha_t x+\sigma_t\varepsilon\)</span> and <span class="math inline">\(x=g(\theta)\)</span>, <span class="math display">\[\nabla_\theta\mathcal L_{\text{Diff}}
  =\mathbb E_{t,\varepsilon}\!
     \bigl[
       2(\hat\varepsilon_\varphi(z_t)-\varepsilon)
       \,\partial_{z_t}\hat\varepsilon_\varphi(z_t)
       \,\partial_x z_t
       \,\partial_\theta x
     \bigr].\]</span> Here <span class="math inline">\(\partial_{z_t}\hat\varepsilon_\varphi\)</span> (the U‑Net Jacobian) is enormous; propagating it would require a full backward pass <em>per</em> pixel sample and yields unstable gradients at high noise levels.</p><p>Therefore DreamFusion <em>drops</em> the Jacobian, retaining only the residual <span class="math inline">\(\hat\varepsilon_\varphi(z_t)-\varepsilon\)</span>. The gradient becomes <span class="math display">\[\nabla_\theta\mathcal L_{\text{SDS}}
  =\mathbb E_{t,\varepsilon}\!
     \bigl[
       w(t)\bigl(\hat\varepsilon_\varphi(z_t;y,t)-\varepsilon\bigr)
       \,\partial_\theta x
     \bigr],\]</span> the celebrated <strong>score‑distillation sampling (SDS)</strong> gradient.</p><p>Crucially, SDS is <strong>not</strong> a heuristic; it performs <em>probability‑density distillation</em> in parameter space: updating <span class="math inline">\(\theta\)</span> with the SDS gradient equals minimizing a <em>weighted KL divergence</em> between (1) the noisy distribution of the rendered image and (2) the true diffusion distribution at the same noise level <span class="math inline">\(t\)</span>.</p><h2 id="density-distillation-in-parameter-space">Density Distillation in Parameter Space</h2><h4 id="kl-divergence.">KL divergence.</h4><p>For densities <span class="math inline">\(p(z)\)</span> and <span class="math inline">\(q(z)\)</span>, <span class="math display">\[\mathrm{KL}(p\|q)=\int p(z)\log\frac{p(z)}{q(z)}\,dz
  =\mathbb E_{z\sim p}[\log p(z)-\log q(z)].\]</span> If <span class="math inline">\(q\)</span> encodes <span class="math inline">\(p\)</span>, the extra code length over the optimal <span class="math inline">\(H(p)\)</span> is exactly <span class="math inline">\(\mathrm{KL}(p\|q)\)</span>.</p><h4 id="equivalence-to-sds.">Equivalence to SDS.</h4><p>Let <span class="math inline">\(q(z_t\mid x)=\mathcal N(z_t;\alpha_t x,\sigma_t^2I)\)</span> with <span class="math inline">\(x=g(\theta)\)</span>, and <span class="math inline">\(p_\phi(z_t\mid y)\)</span> be the true diffusion distribution at level <span class="math inline">\(t\)</span>. One finds <span class="math display">\[\nabla_\theta\mathrm{KL}\bigl(q\|p_\phi\bigr)
=\mathbb E_{t,\varepsilon}\bigl[
    w(t)\bigl(\epsilon_\phi(z_t)-\varepsilon\bigr)
    \,\partial_\theta x
  \bigr],\]</span> identical to <span class="math inline">\(\nabla_\theta\mathcal L_{\text{SDS}}\)</span>.</p><h2 id="practical-pipeline-stable-dreamfusion">Practical Pipeline (Stable DreamFusion)</h2><p>Each step:</p><p><strong>1. Rendering.</strong> Render a low‑resolution image <span class="math inline">\(x=g(\theta)\)</span> (<span class="math inline">\([B,3,H,W]\)</span>), then upsample.</p><p><strong>2. Forward noise.</strong> Encode the high‑res image with the VAE of Stable Diffusion; sample <span class="math inline">\(t\sim\mathrm{Uniform}(1,\dots,T)\)</span> and add noise.</p><p><strong>3. Denoising.</strong> Run the noisy latent through the frozen U‑Net to obtain <span class="math inline">\(\hat\varepsilon\)</span>.</p><p><strong>4. Back‑prop SDS.</strong> Because the U‑Net is frozen, stop its gradient and compute</p><div class="sourceCode" id="cb1" data-language="Python"><pre class="sourceCode python"><code class="sourceCode python"><span id="cb1-1"><a href="#cb1-1" aria-hidden="true" tabindex="-1"></a>w       <span class="op">=</span> (<span class="dv">1</span> <span class="op">-</span> <span class="va">self</span>.alphas[t])                <span class="co"># weight w(t)</span></span>
<span id="cb1-2"><a href="#cb1-2" aria-hidden="true" tabindex="-1"></a>grad    <span class="op">=</span> w[:, <span class="va">None</span>, <span class="va">None</span>, <span class="va">None</span>] <span class="op">*</span> (noise_pred <span class="op">-</span> noise)</span>
<span id="cb1-3"><a href="#cb1-3" aria-hidden="true" tabindex="-1"></a>target  <span class="op">=</span> (latents <span class="op">-</span> grad).detach()           <span class="co"># stop gradient</span></span>
<span id="cb1-4"><a href="#cb1-4" aria-hidden="true" tabindex="-1"></a>loss_sds <span class="op">=</span> <span class="fl">0.5</span> <span class="op">*</span> F.mse_loss(</span>
<span id="cb1-5"><a href="#cb1-5" aria-hidden="true" tabindex="-1"></a>             latents, target, reduction<span class="op">=</span><span class="st">&quot;sum&quot;</span>) <span class="op">/</span> batch_size</span></code></pre></div><p><strong>5. Update <span class="math inline">\(\theta\)</span>.</strong> Back‑propagate <code>loss_sds</code> through the NeRF MLP only; iterate until convergence.</p>

