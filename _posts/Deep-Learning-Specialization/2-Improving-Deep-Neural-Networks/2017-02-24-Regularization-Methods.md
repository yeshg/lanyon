---
layout: post
title: Regularization Methods
tags: Deep-Learning
mathjax: true
excerpt:
    <p>Regularization is a way to enhance neural networks used to reduce variance, and by extension overfitting. In a nutshell, regularization penalizes the loss function of a neural network to artificially discourage overfitting to the training dataset, such as learning the background noise. Thus, regularization helps models that have poor prediciton and generalization power. </p>
---

Regularization is a way to enhance neural networks used to reduce variance, and by extension overfitting. In a nutshell, regularization penalizes the loss function of a neural network to artificially discourage overfitting to the training dataset, such as learning the background noise. Thus, regularization helps models that have poor prediciton and generalization power.

### Standard Regularization
There are two ways to regularize: using L1 and L2 matrix Norms.

L1 matrix Norm:
- `||W|| = Sum(|W[i,j]|) # Sum of all Ws with abs`\

L2 matrix Norm sometimes its called Frobenius norm:
- `||W||2 = Sum(|W[i,j]|^2) #Sum of all Ws squared`
- Also can be calculated using`||W||2 = W.T * W`

Regularization For logistic regression: Penalizing the loss function.
- The normal cost function that we want to minimize is: `J(w,b) = (1/m) * Sum(L(y(i),y'(i)))`
- The L2 Regularization version: `J(w,b) = (1/m) * Sum(L(y'(i),y'(i))) + (Lmda/2m) * ||W||2`
- The L1 Regularization version: `J(w,b) = (1/m) * Sum(L(y'(i),y'(i))) + (Lmda/2m) * (||W||)`
- The L1 Regularization version makes a lot of w values become zeros, which makes the model size is small.
- L2 Regularization is being used much often.
- `Lmda` here is the Regularization parameter (Hyperparameter)

Regularization For NN:

- The normal cost function that we want to minimize is: `J(W1,b1...,WL,bL) = (1/m) * Sum(L(y'(i),y'(i)))`
- The L2 Regularization version: `J(w,b) = (1/m) * Sum(L(y'(i),y'(i))) + (Lmda/2m) * Sum((||W[l]||) ^2)`
- We stack the matrix as one vector `(mn,1)` and then we apply `sqrt(w1^2+w2^2.....)`
- To do back propagation (old way):
    - `w[l] = w[l] - learningRate * dw[l]`
- The new way:
    - `dw[l] = (Back prob) + (Lmda/m)*w[l]`
- So:
    - `w[l] = w[l] - (Lmda/m)*w[l] - learningRate * dw[l]`
    - `w[l] = (1 - (learninRate*Lmda)/m) w[l] - learninRate*dw[l]`
- In practice this penalizes large weights and effectively limits the freedom in your model.
- The new term `(1 - (learninRate*Lmda)/m) w[l]` causes the weight to decay in proportion to its size.

### Dropout Regularization
Dropout is a really effective, and quite simple way to emphasize generalizations instead of overfitting. While training the neural network, dropout is implemented by randomly keeping neurons active based on some probability hyperparameter.

![](Images/dropout.jpeg)

The image above is from the original paper that proposed dropout, and it illustrates the basic idea. By turning off some neurons during training, the dropout only updates a sample of the full neural network, and the activations for every neuron in the network are forced to be more robust as training goes on. Overfitting is very hard when any number of neurons could be switched off during training.

### Other Regularization Methods
#### Data Augmentation
A transformation can be applied to the data, to get more data. For example, in computer vision applications, a rotation or flip could be used to get more data out of the original dataset. In OCR, distorting digits is a common practice.
Although getting larger datasets this way isn't as good as having actual independent data, it can still be used as a regularization technique.

#### Early stopping
In early stopping, we plot the training and dev sets cost logs together, and identify the point at which the training data graph and dev data graph are best:

![](Images/early_stopping.png)

This technique isn't recommended, because it changes the ultimate goal from optimzing weights to finding point of best performance on dev and train sets... but its advantage is that it doesn't require yet another hyperparameter to tune like lambda in standard regularization.

#### Model Ensembles
Model ensembles contain several independent models trained on the same data. At test time, the models' results are averaged and you can get an extra 2% of performance.
