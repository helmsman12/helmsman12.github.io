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

In this post, we briefly introduce the standard diffusion process.