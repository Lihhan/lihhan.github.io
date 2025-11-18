---
layout: post
title: "Diffusion Model 教程 - DDPM Practical: 实用技巧"
subtitle: "从SDE到DreamFusion的完整推导"
date: 2025-02-17
author: Lihan
tags: [Diffusion, Deep Learning, 教程, LaTeX]
---

<h2 id="training-procedure">Training Procedure</h2><ul><li><p>Sample an image <span class="math inline">\(x_{0}\)</span> from the training set (<span class="math inline">\(x_{0}\sim q(x_{0})\)</span>);</p></li><li><p>Draw a random time step <span class="math inline">\(t\sim\mathrm{Uniform}(1,\ldots,T)\)</span>;</p></li><li><p>Sample Gaussian noise <span class="math inline">\(\varepsilon\sim\mathcal N(0,\mathbf I)\)</span>;</p></li><li><p>Compute the loss <span class="math display">\[\mathit{loss}
          \;=\;
          \Bigl\lVert
            \varepsilon
            -
            \varepsilon_{\theta}\!\bigl(
              \sqrt{\bar\alpha_{t}}\,x_{0}
              +\sqrt{1-\bar\alpha_{t}}\,\varepsilon,\;t
            \bigr)
          \Bigr\rVert^{2},\]</span> where <span class="math inline">\(\varepsilon_{\theta}\)</span> is the UNet‑based denoising network;</p></li><li><p>Back‑propagate the loss and update the model parameters; repeat until convergence.</p></li></ul><h2 id="inference-sampling-procedure">Inference (Sampling) Procedure</h2><ul><li><p>Draw an initial latent <span class="math inline">\(x_T\sim\mathcal N(0,\mathbf I)\)</span>;</p></li><li><p>For <span class="math inline">\(t=T,\,T-1,\ldots,1\)</span>:</p><ul><li><p>Sample fresh noise <span class="math inline">\(\varepsilon\sim\mathcal N(0,\mathbf I)\)</span>;</p></li><li><p>Use Eq. (24) to obtain <span class="math inline">\(x_{t-1}\)</span> from <span class="math inline">\(x_t\)</span>.</p></li></ul></li><li><p>The final output is <span class="math inline">\(x_0&#39;\)</span>.</p></li></ul><h2 id="extension-to-text-to-image-generation">Extension to Text-to-Image Generation</h2><p>To turn a DDPM into a text‑to‑image diffusion model, replace each self‑attention block in the UNet with a <em>cross‑attention</em> block, feeding the encoded text prompt <span class="math inline">\(y\)</span> as <strong>K</strong> and <strong>V</strong> while retaining the latent features as <strong>Q</strong>. The training loss becomes <span class="math display">\[\mathit{loss}
  \;=\;
  \Bigl\lVert
    \varepsilon
    -
    \varepsilon_{\theta}\!\bigl(
      \sqrt{\bar\alpha_{t}}\,x_{0}
      +\sqrt{1-\bar\alpha_{t}}\,\varepsilon,\;t,\,y
    \bigr)
  \Bigr\rVert^{2}.\]</span></p>

