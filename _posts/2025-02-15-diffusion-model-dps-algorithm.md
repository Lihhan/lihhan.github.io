---
layout: post
title: "Diffusion Model 教程 - DPS Algorithm: 数据后验采样"
subtitle: "从SDE到DreamFusion的完整推导"
date: 2025-02-15
author: Lihan
tags: [Diffusion, Deep Learning, 教程, LaTeX]
---

<h2 id="algorithmic-rationale">Algorithmic Rationale</h2><p>Armed with the preceding background, we now examine the <strong>DPS (Denoising Posterior Sampling)</strong> algorithm in a practical setting.</p><p>In an <em>unconditional</em> diffusion model the only guidance is the prior score <span class="math inline">\(\nabla_x\log p(x)\)</span>; the learned “force field’’ merely pulls noisy samples toward the natural‑image manifold. In real applications—say, image denoising—we need the final output to be both <em>realistic</em> (prior fidelity) and <em>consistent</em> with a noisy observation <span class="math inline">\(y\)</span> (data fidelity). Hence we upgrade the marginal <span class="math inline">\(p(x)\)</span> at every step to a conditional <span class="math inline">\(p(x\mid y)\)</span>. In the reverse SDE, the network must now predict <span class="math inline">\(\nabla_x\log p(x\mid y)\)</span> instead of <span class="math inline">\(\nabla_x\log p(x)\)</span>.</p><p>Bayes’ rule gives <span class="math display">\[\nabla_x\log p(x\mid y)
  = \nabla_x\log p(x) + \nabla_x\log p(y\mid x).\]</span> The prior term can be pretrained as usual, but <span class="math inline">\(\nabla_x\log p(y\mid x)\)</span> lacks a closed form because the likelihood is time‑dependent and we only know the relation between <span class="math inline">\(y\)</span> and the clean image <span class="math inline">\(x_0\)</span>.</p><p>Assume the observation model <span class="math inline">\(y=A(x_0)+n\)</span>, with <span class="math inline">\(A\)</span> an imaging operator and <span class="math inline">\(n\)</span> additive noise. Because the likelihood <span class="math inline">\(p(y\mid x_t)\)</span> has no analytic form, we relate it to the known <span class="math inline">\(p(y\mid x_0)\)</span> via conditional independence: <span class="math display">\[\begin{aligned}
p(y\mid x_t) &amp;=
  \int p(y\mid x_0,x_t)\,p(x_0\mid x_t)\,dx_0
  =\int p(y\mid x_0)\,p(x_0\mid x_t)\,dx_0
  =\mathbb E_{x_0\sim p(x_0\mid x_t)}[p(y\mid x_0)] .
\end{aligned}\]</span></p><p>Applying Jensen’s inequality, <span class="math display">\[p(y\mid x_t)\approx
  p\!\bigl(y\mid\hat x_0\bigr),\qquad
  \hat x_0:=\mathbb E[x_0\mid x_t].\]</span></p><p>Because the forward process satisfies <span class="math inline">\(x_t=\sqrt{\bar\alpha}\,x_0+\sqrt{1-\bar\alpha}\,z\)</span>, one finds <span class="math display">\[\hat x_0(x_t)=
  \frac{1}{\sqrt{\bar\alpha}}
  \Bigl(
    x_t + (1-\bar\alpha)\,\nabla_{x_t}\log p_t(x_t)
  \Bigr).\]</span></p><p>Assuming Gaussian observation noise <span class="math display">\[p(y\mid x_0)
  \propto
  \exp\!\Bigl(-\tfrac1{2\sigma^2}\|y-A(x_0)\|_2^2\Bigr),\]</span></p><p>we obtain <span class="math display">\[\nabla_{x_t}\log p_t(y\mid x_t)
  \approx
  -\frac{1}{\sigma^2}\,\nabla_{x_t}
  \|y-A\!\bigl(\hat x_0(x_t)\bigr)\|_2^2.\]</span></p><p>Hence the <em>posterior</em> score is approximated by <span class="math display">\[\nabla_{x_t}\log p_t(x_t\mid y)
  \approx
  s_\theta^*(x_t,t)
  -\rho\,\nabla_{x_t}\|y-A(\hat x_0)\|_2^2,
  \qquad \rho:=\frac1{\sigma^2}.\]</span></p><h2 id="practical-implementation">Practical Implementation</h2><ol><li><p><strong>Initialization.</strong> Start the reverse chain with pure noise <span class="math inline">\(x_N\sim\mathcal N(0,I)\)</span>.</p></li><li><p><strong>Backward iteration.</strong> For <span class="math inline">\(i=N{-}1,\dots,0\)</span>:</p><ol><li><p>Compute the prior score with the pretrained model <span class="math inline">\(\hat s=s_\theta(x_i,i)\approx\nabla_{x_i}\log p_t(x_i)\)</span>.</p></li><li><p><strong>Tweedie update</strong> (estimate a clean image) <span class="math display">\[\hat x_0
              =\frac{1}{\sqrt{\bar\alpha_i}}
                \bigl(x_i+(1-\bar\alpha_i)\hat s\bigr).\]</span></p></li><li><p><strong>Prior‑only DDPM step</strong> <span class="math display">\[x&#39;_{i-1}=
                \sqrt{\frac{\bar\alpha_{i-1}(1-\alpha_i)}{1-\bar\alpha_i}}\,x_i
                +\sqrt{\frac{\bar\alpha_{i-1}\beta_i}{1-\bar\alpha_i}}\,
                 \hat x_0
                +\tilde\sigma_i\,z,\;
              z\sim\mathcal N(0,I).\]</span></p></li><li><p><strong>Likelihood correction</strong> (Gaussian case) <span class="math display">\[x_{i-1}=x&#39;_{i-1}
                       -\zeta_i\,A^\top\bigl(A(\hat x_0)-y\bigr),\]</span> where <span class="math inline">\(\zeta_i\)</span> trades off prior versus data fidelity.</p></li></ol></li><li><p><strong>Output.</strong> After the loop, return the reconstruction <span class="math inline">\(\hat x_0\)</span>—now consistent with both the prior (realism) and the observation <span class="math inline">\(y\)</span> (data fidelity).</p></li></ol>

