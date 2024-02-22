---
title: "[Review] Diffusion-LM Improves Controllable Text Generation"

categories:
    - paper_review
tags:
    - [NLP, Generative AI]

date: 2024-02-22
last_modified_at: 2024-02-22
layout: single
use_math: true
---

this is my review of the paper “Diffusion-LM Improves Controllable Text Generation”. 

## 1. Introduction

Large Language Models are now being widely used in real-world applications. However, for the reliable use of these models, the output of the models should be “controllable”: the generated text should match the intended requirements - topics, syntactic structure, etc. Straightforward method to meet this goal will be fine-tuning with supervised learning, with data of the form (control, text). However, this method comes with multiple updates of the LLM parameters(which is expansive), and accounting for every possible set of control parameters is impossible. 

To solve this issue, the authors of the paper suggests a new LM, diffusion-LM, a new language model based on continuous diffusion. The model starts from gaussian noise, which the model iteratively denoises to a word representation.

## 2. Problem Statement and Background

In this section, we briefly review diffusion process on continuous domain.

Diffusion model is a Markov chain $x_T, ... \ x_0$ with each $x_i$ is in $\mathbb{R} ^ d$. $x_T$ is gaussian, and the diffusion model denoises the latent vector to approximate samples from the target data distribution. 

During the forward process $x_{t-1} \rightarrow x_{t}$, a gaussian noise is added to the latent variable, and the process can be parametrized as $q(\boldsymbol{x_t}|\boldsymbol{x_{t-1}}) = \mathcal{N}(\boldsymbol{x_t}; \sqrt{1-\beta_t}\boldsymbol{x_{t-1}}, \beta_tI)$
where the hyperparameter $\beta_t$ is the amount of noise added during each processes. A diffusion model aims to backtrack this forward processes and tries to denoise the gaussian $\boldsymbol{x_T}$ into a meaningful latent variable $\boldsymbol{x_0}$. 

The transition $x_{t} \rightarrow x_{t-1}$ can be parametrized as 

$
p_\theta(\boldsymbol{x_{t}}|\boldsymbol{x_{t-1}})=\mathcal{N}(\boldsymbol{x_{t}}; \mu_\theta(x_{t}, t), \Sigma_\theta(x_{t}, t)),
$ 

where $\mu_\theta, \Sigma_\theta$ are the learnable parameters of a diffusion model.

The diffusion model is trained to maximize the marginal likelihood of the data 

![likelihood image](/assets/images/log_likelihood.png "log_likelihood")

and the objective function is given as vlb loss. However, this objective incurs unstable learning process and requires many optimization tricks. For this reason, simpler objective is purposed as below.

![training loss image](/assets/images/vlb_simple_loss.png "training loss")

Diffusion LM follows this simplified objective.


## 3. Diffusion-LM

Application of diffusion models to text requires several modifications. First, the model should learn a way to map discrete text into a continuous space. Second, the model needs a rounding method to project vectors in continuous space back to text embedding. This section explains the approaches taken to address these issues. 

### 3.1. End-to-End Training

The embedding function, $EMB(w_i)$, maps a word $w_i$ to a vector in $\mathbb{R}^d$. The embedding of a sentence $\boldsymbol{w}$ with n words is defined as $[EMB(w_1), EMB(w_2), ...\ , EMB(w_n)]$, where $w_i$ is a i-th word of the sentence. 

In figure 2, there is an extra transition step between text $\boldsymbol{w}$ and latent variable $\boldsymbol{x_0}$, denoted as ‘rounding’ and ‘embedding’. 
The embedding process is parametrized by $q_\theta(\boldsymbol{x_0} | \boldsymbol{w}) = \mathcal{N}(EMB(\boldsymbol{w}), \sigma_0I)$, 
where the rounding step is paramtrized by $p_\theta(\boldsymbol{w}|\boldsymbol{x_0})=\prod\limits_{i=i}^n p_\theta(w_i | x_i)$ with $p_\theta (w_i|x_i)$  as softmax distribution.

With this additional changes, the training objectives introduced earlier now becomes

![e2e loss image](/assets/images/e2e_loss.png "e2e loss")

Additional terms in variational lower bound accounts for the KL divergence between the probability distribution of embedding and rounding process. 

In the paper, the researchers experimented with fixed word embeddings and learnable word embeddings. Learnable word embeddings performed better. 

### 3.2. Reducing Rounding Errors

Rounding processes was defined by $argmax \ p_{\theta}(\boldsymbol{w}|\boldsymbol{x_0})=\prod\limits_{i=i}^n p_{\theta}(w_{i} | x_{i})$. 
However, the paper says that the model failed to generate $\boldsymbol{x_0}$ that commits to a single word. 

The paper argues that the original end-to-end training objective puts insufficient emphasis on $\boldsymbol{x_0}$ to commit to a single word. 
It suggests additional training objective, 

![new loss image](/assets/images/new_loss.png "new loss")

where model $f_\theta(\boldsymbol{x_t}, t)$ predicts $\boldsymbol{x_0}$ directly. 
Paper reports that model trained with this objective quickly learns that $\boldsymbol{x_0}$ should be centered at a word embedding. 

Same intuition can be applied during the decoding scheme. Here, ‘Clamping’ trick is introduced, where $Clamp(f_\theta(\boldsymbol{x_t}, t))$ is the nearest word embedding from the predicted vector. 
Now, instead of the original sampling process $\boldsymbol{x_{t-1}} = \sqrt{\bar{\alpha}} \cdot f_\theta(\boldsymbol{x_t}, t) + \sqrt{1-\bar{\alpha}}\epsilon$ (Note that $f_\theta(\boldsymbol{x_t}, t)$ predicts $\boldsymbol{x_0}$), we have $\boldsymbol{x_{t-1}} = \sqrt{\bar{\alpha}} \cdot Clamp(f_\theta(\boldsymbol{x_t}, t)) + \sqrt{1-\bar{\alpha}}\epsilon$. The clamping trick forces the predicted vector to commit to a word in the intermediate steps, improving prediction and reducing rounding errors.