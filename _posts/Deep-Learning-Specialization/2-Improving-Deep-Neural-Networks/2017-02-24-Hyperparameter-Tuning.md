---
layout: post
title: Hyperparameter Tuning
tags: Deep-Learning
mathjax: true
categories: Deep-Learning
excerpt:
    <p><Important things to keep in mind when tuning hyperparameters/p>
---

The relative importance of the various hyperparameters (according to Andrew Ng) are as follows:
1. Learning Rate
2. Momentum beta.
3. Mini-batch size.
4. No. of hidden units.
5. No. of layers.
6. Learning rate decay.
7. Regularization lambda.
8. Activation functions.
9. Adam beta1 & beta2.

Hyperparameters tuning in practice: Pandas vs. Caviar
- Intuitions about hyperparameter settings from one application area may or may not transfer to a different one.
- If you don't have much computational resources you can use the "babysitting model":
    - Day 0 you might initialize your parameter as random and then start training.
    - Then you watch your learning curve gradually decrease over the day.
    - And each day you nudge your parameters a little during training.
    - Called panda approach.
- If you have enough computational resources, you can run some models in parallel and at the end of the day(s) you check the results.
    - Called Caviar approach.
