---
layout: post
title: Optimization Algorithms
tags: Deep-Learning
mathjax: true
categories: Deep-Learning
excerpt:
    <p>Optimization algorithms are the mathematical techniques to minimize the cost function as quickly and efficiently as possible. There have been several optimization algorithms proposed in the history of deep learning literature.</p>
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

#### Understanding Mini-batch Gradient descent
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

The exponentially weighted average pseudo code is very simple:
{% highlight js %}
V = 0
Repeat
{
	Get ceta(t)
	V = beta * V + (1-beta) * ceta(t)
}
{% endhighlight %}

### Gradient Descent with Momentum

#### Physics perspective
The momentum algorithm uses a physical interpretation of the gradient as an elevation map, and the current parameters as a position among this map's hills and valleys. From the initial paramters, where the velocity of a particle at the position of the initial weights is 0, the force on the particle is simply the negative gradient of the loss function.

Unlike the sgd update, where the gradient directly integrates the position, th emomentum update has the gradient influence the velocity, which in turn affects the position:

```
# Momentum update
v = mu * v - learning_rate * dx # integrate velocity
x += v # integrate position
```

Note that the hyperparameter `mu` from the above pseudo code is commonly called *momentum*, when it really should be called the coefficient of friciton. The hyperparameter has a dampening effect on the velocity. It's very common to *anneal* the momentum hyperparameter over the duration of training to bring the particle to a gradual stop at the bottom of the gradient map.

#### Momentum in terms of Exponentially Weighted Averages
The momentum algorithm is the same as calculating the exponentially weighted averages for gradients and then updating weights with new values:
```
VdW = 0, Vdb = 0
on iteration t:
	# The mini batch can be the whole batch its ok
	compute dw, db on current mini batch

	VdW = (beta * VdW) + (1 - beta)dW
	Vdb = (beta * Vdb) + (1 - beta)db
	W = W - learning_rate * VdW
	b = B - learning_rate * Vdb
```
Note that `beta = 0.9` is a common value in practice.

### RMSprop

RMSprop stands for root mean square prop.

This algorithm speeds up gradient descent by slowing down movements on the vertical axis of the cost function and by speeding up movement in the horizontal direction of the cost function:
![](RMSprop.png)

Pseudo code:
```
SdW = 0, Sdb = 0
on iteration t:
	# The mini batch can be the whole batch its ok
	compute dw, db on current mini batch

	SdW = (beta * SdW) + (1 - beta)dW^2
	Sdb = (beta * Sdb) + (1 - beta)db^2
	W = W - learning_rate * dW/sqrt(SdW)
	b = B - learning_rate * db/sqrt(Sdb)
```

### Adam Optimization Algorithm
Stands for Adaptive Momentum Estimation. Along with RMSprop, Adam is one of the most popular optimization algorithms. Adam is a combination of RMSprop and momentum:
```
VdW = 0, VdW = 0
SdW = 0, Sdb = 0
on iteration t:
	# The mini batch can be the whole batch its ok
	compute dw, db on current mini batch

	VdW = (beta1 * dW) + (1 - beta1)dW                    #Momentum
	Vdb = (beta1 * db) + (1 - beta1)db					#Momentum

	SdW = (beta2 * dW) + (1 - beta2)dW^2					#RMSprop
	Sdb = (beta2 * db) + (1 - beta2)db^2					#RMSprop

	VdW = VdW/ (1 - beta^t)			#Fixing bias
	Vdb = Vdb/ (1 - beta^t)			#Fixing bias

	SdW = SdW/ (1 - beta^t) 		#Fixing bias
	Sdb = Sdb/ (1 - beta^t)			#Fixing bias

	W = W - learning_rate * VdW/(sqrt(SdW) + epsilon)
	b = B - learning_rate * Vdb/(sqrt(Sdb) + epsilon)
```

Hyperparameters:
- Learning rate
- `Beta1` - momentum parameter, `0.9` recommended
- `Beta2` - RMSprop parameter, `0.999` recommended
- `epsilon` - `10^-8` recommended

### Learning rate decay
- Slowly reduce learning rate.
- In mini batch algorithm, we said that the minimization of the cost won't reach optimum point. But by making the learning rate decays with iterations it will reach it as the steps beside the optimum is small.
- One technique equations islearning_rate = (1 / (1 + decay_rate * epoch_num)) * learning_rate_0
    - epoch_num is over all data (not a single mini batch).
- Other learning rate decay methods (Continuous):
    - learning_rate = (0.95 ^ epoch_num) * learning_rate_0
    - learning_rate = (k / sqrt(epoch_num)) * learning_rate_0
- Some people is making changing the learning rate manually.
- For Andrew Ng, learning rate decay has less priority
