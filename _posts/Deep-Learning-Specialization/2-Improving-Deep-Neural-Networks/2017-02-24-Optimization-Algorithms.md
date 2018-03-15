---
layout: post
title: Optimization Algorithms
tags: Deep-Learning
mathjax: true
categories: Deep-Learning
excerpt: >-
  <p>Optimization algorithms are the mathematical techniques to minimize the
  cost function as quickly and efficiently as possible. There have been several
  optimization algorithms proposed in the history of deep learning
  literature.</p>
published: true
---

Optimization algorithms are the mathematical techniques to minimize the cost function as quickly and efficiently as possible. There have been several optimization algorithms proposed in the history of deep learning literature.

### Mini-batch Gradient descent
Training a neural network with a large amount of data is slow (with classical gradient descent), so can split the data into mini-batches. Here, the old and slow form of graident descent is known as *Batch Gradient Descent* (run on the whole dataset) and the faster one is known as *Mini-batch Gradient Descent* (run on the mini-batches).

Mini-batch algorithm pseudo code:
```
for t = 1:No_of_batches                                     #This is called on epoch
	AL, caches = forward_prop(X{t}, Y{t})
	Cost = compute_cost(AL, Y{t})
	grads = backward_prop(AL, caches)
	UpdateParameters(grads)
```
Although there is a for loop here, the code actually inside the epoch should be vectorized.

### Understanding Mini-batch Gradient descent
The mini-batch gradient descent algorithm's cost vs iterations graph won't have a smooth downward sloping curve like that of batch gradient descent. Instead, the cost will contain several local ups and downs, but a global downward curve:

![](miniBatch.png)

Choosing mini-batch size:
- If (mini batch size = m) ==> Batch gradient descent
    - If (mini batch size = 1) ==> Stochastic gradient descent
    - Might be faster than standard in big data > 10^7
    - If (mini batch size = between 1 and m) ==> Mini Batch gradient descent
- In Stochastic gradient descent is so noisy regarding cost minimization and won't reach the minimum cost. Also you lose vectorization advantage.
- In mini batch gradient descent is so noisy regarding cost minimization and won't reach the minimum cost. But you have the vectorization advantage and you can look at the costs when the code is running to see if its right. To help with the noisy cost minimization you should reduce the learning rate.
- Guidelines for using mini batch:
    - It has to be a power of 2 to take advantage of vectorization: 64, 128, 256, 512, 1024....
    - Make sure mini-batch fits in CPU/GPU
- Mini batch size is a Hyperparameter.

### Exponentially Weighted Averages
Exponentially weighted averages are a fundamental concept for understading the more advanced, faster optimization algorithms.

If we have data like the temperature of day through the year it could be like this:
```
t(1) = 40
t(2) = 49
t(3) = 45
..
t(180) = 60
..
```
This data is small in winter and big in summer. If we plot this data we will find it some noisy.
Now lets compute the Exponentially weighted averages:

```
V0 = 0
V1 = 0.9 * V0 + 0.1 * t(1) = 4		# 0.9 and 0.1 are hyperparameters
V2 = 0.9 * V1 + 0.1 * t(2) = 8.5
V3 = 0.9 * V2 + 0.1 * t(3) = 12.15
...
```

Let's plot the result:
![](expAvg1.png)

This image refers to the general equation $V(t) = \beta V_{t-1} + (1-\beta )\theta_{t}$ where beta is a hyperparameter.

### Momentum Update
Using a physics based approach, we can skip an intermediate step in vanilla gradient descent.

In the momentum update, the loss is interpreted as the height of a hilly terrain. With this height, the current position of the weights has a certain potential energe $U = mgh$ (where g is the gravitational constant). The random initialization of weights will be equivalent to starting at some position with zero initial velocity.

Then, the downward force experienced by a particle at this location is $F = - \nabla U$

```
# Momentum update
v = mu * v - learning_rate * dx # integrate velocity
x += v # integrate position
```

### Nesterov Momentum Update
![](Images/nesterov.jpeg)