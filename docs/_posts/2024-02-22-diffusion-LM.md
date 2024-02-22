---
title: "[Review] Diffusion-LM Improves Controllable Text Generation"

categories:
    - paper_review
tags:
    - [NLP, Generative AI]

date: 2024-02-22
last_modified_at: 2024-02-22
layout: post
---

this is my review of the paper “Diffusion-LM Improves Controllable Text Generation”. 

## 1. Introduction

Large Language Models are now being widely used in real-world applications. However, for the reliable use of these models, the output of the models should be “controllable”: the generated text should match the intended requirements - topics, syntactic structure, etc. Straightforward method to meet this goal will be fine-tuning with supervised learning, with data of the form (control, text). However, this method comes with multiple updates of the LLM parameters(which is expansive), and accounting for every possible set of control parameters is impossible. 

To solve this issue, the authors of the paper suggests a new LM, diffusion-LM, a new language model based on continuous diffusion. The model starts from gaussian noise, which the model iteratively denoises to a word representation.

## 2. Problem Statement and Background

In this section, we briefly review diffusion process on continuous domain.

Diffusion model is a Markov chain $x_T, ... \ x_0$ with each $x_i$ is in $\reals ^ d$. $x_T$ is gaussian, and the diffusion model denoises the latent vector to approximate samples from the target data distribution. 

During the forward process $x_{t-1} \rightarrow x_{t}$, a gaussian noise is added to the latent variable, and the process can be parametrized as 

$q(\bm{x}_t|\bm{x_{t-1}}) = \mathcal{N}(\bm{x}_t; \sqrt{1-\beta_t}\bm{x_{t-1}}, \beta_tI)$

where the hyperparameter $\beta_t$ is the amount of noise added during each processes. A diffusion model aims to backtrack this forward processes and tries to denoise the gaussian $\bm{x_T}$ into a meaningful latent variable $\bm{x_0}$. 

The transition $x_{t} \rightarrow x_{t-1}$ can be parametrized as below:

<div align="center">
$p_\theta(\bm{x_t}|\bm{x_{t-1}})=\mathcal{N}(\bm{x_t}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$
</div>

where $\mu_\theta, \Sigma_\theta$ are the learnable parameters of a diffusion model.

The diffusion model is trained to maximize the marginal likelihood of the data $\mathbb{E}_{\bm{x_0} \sim p_{data}}[\log{p_\theta(\bm{x_0})}]$, and the objective function is given as

<div align="center">
$\mathcal{L}_{vlb}(\bm{x_0})=\mathbb{E}_{q(\bm{x_{1:T}|\bm{x_0}})} [\log{\frac {q(\bm{x_T}| \bm{x_0})} {p_\theta(\bm{x_T})}} + \sum\limits_{t=2}^T\log{\frac {q(\bm{x_t-1}| \bm{x_0}, \bm{x_t})} {p_\theta(\bm{x_{t-1} | \bm{x_t}})}} - \log{p_\theta(\bm{x_0}|\bm{x_1})}]$.
</div>

However, this objective incurs unstable learning process and requires many optimization tricks. For this reason, simpler objective is purposed as below.

<div align="center">
$\mathcal{L}_{simple}=\sum\limits_{t=1}^T \mathbb{E}_{q(\bm{x_t} | \bm{x_0})} || \mu_\theta(\bm{x_t, t}) - \hat{\mu}(\bm{x_t}, \bm{x_0}|| ^2$
</div>

Diffusion LM follows this simplified objective.


## 3. Diffusion-LM

Application of diffusion models to text requires several modifications. First, the model should learn a way to map discrete text into a continuous space. Second, the model needs a rounding method to project vectors in continuous space back to text embedding. This section explains the approaches taken to address these issues. 

### 3.1. End-to-End Training

The embedding function, $EMB(w_i)$, maps a word $w_i$ to a vector in $\mathbb{R}^d$. The embedding of a sentence $\bm{w}$ with n words is defined as $[EMB(w_1), EMB(w_2), ...\ , EMB(w_n)]$, where $w_i$ is a i-th word of the sentence. 

In figure 2, there is an extra transition step between text $\bm{w}$ and latent variable $\bm{x_0}$, denoted as ‘rounding’ and ‘embedding’. The embedding process is parametrized by $q_\theta(\bm{x_0} | \bm{w}) = \mathcal{N}(EMB(\bm{w}), \sigma_0I)$, where the rounding step is paramtrized by $p_\theta(\bm{w}|\bm{x_0})=\prod\limits_{i=i}^n p_\theta(w_i | x_i)$ with $p_\theta (w_i|x_i)$  as softmax distribution.

With this additional changes, the training objectives introduced earlier now becomes

<div align="center">
$\mathcal{L}^{e2e}_{vlb}(\bm{w})=\mathbb{E}_{q_\theta(\bm{x_0|\bm{w}})}[\mathcal{L}_{vlb}(\bm{x_0})+\log{q_\phi(\bm{x_0|\bm{w} }) - \log{p_\theta(\bm{w} | \bm{x_0})}}]$,
</div>

<div align="center">
$\mathcal{L}^{e2e}_{simple}(\bm{w})=\mathbb{E}_{q_\theta(\bm{x_{0:T}|\bm{w}})}[\mathcal{L}_{simple}(\bm{x_0})+ || EMB(\bm{w} ) - \mu_\theta(\bm{x_1, 1}) ||^2 - \log{p_\theta(\bm{w} | \bm{x_0})}]$.
</div>

Additional terms in variational lower bound accounts for the KL divergence between the probability distribution of embedding and rounding process. 

In the paper, the researchers experimented with fixed word embeddings and learnable word embeddings. Learnable word embeddings performed better. 

### 3.2. Reducing Rounding Errors

Rounding processes was defined by $\argmax p_\theta(\bm{w}|\bm{x_0})=\prod\limits_{i=i}^n p_\theta(w_i | x_i)$. However, the paper says that the model failed to generate $\bm{x_0}$ that commits to a single word. 

The paper argues that the original end-to-end training objective puts insufficient emphasis on $\bm{x_0}$ to commit to a single word. The paper suggests additional training objective, $\mathcal{L}^{e2e}_{\bm{x_0}-simple}(\bm{x_0}) = \sum_{t=1}^T\mathbb{E}_{\bm{x_t}}|| f_\theta(\bm{x_t}, t) - \bm{x_0}||^2$, where model $f_\theta(\bm{x_t}, t)$ predicts $\bm{x_0}$ directly. Paper reports that model trained with this objective quickly learns that $\bm{x_0}$ should be centered at a word embedding. 

Same intuition can be applied during the decoding scheme. Here, ‘Clamping’ trick is introduced, where $Clamp(f_\theta(\bm{x_t}, t))$ is the nearest word embedding from the predicted vector. 

Now, instead of the original sampling process $\bm{x_{t-1}} = \sqrt{\bar{\alpha}} \cdot f_\theta(\bm{x_t}, t) + \sqrt{1-\bar{\alpha}}\epsilon$ (Note that $f_\theta(\bm{x_t}, t)$ predicts $\bm{x_0}$), we have $\bm{x_{t-1}} = \sqrt{\bar{\alpha}} \cdot Clamp(f_\theta(\bm{x_t}, t)) + \sqrt{1-\bar{\alpha}}\epsilon$. The clamping trick forces the predicted vector to commit to a word in the intermediate steps, improving prediction and reducing rounding errors.