---
layout: post
title: The Basics of Neural Networks
tags: Deep-Learning
mathjax: true
excerpt:
    <p>Neural Networks are the computational models used in machine learning and deep learning. Before building a neural network, we will see binary classification, logistic regression, cost functions, and gradient descent. These are the basic elements that make neural networks possible and effective. </p>
---

Neural Networks are the computational models used in machine learning and deep learning. Before building a neural network, we will see binary classification, logistic regression, cost functions, and gradient descent. These are the basic elements that make neural networks possible and effective.

### Binary classification
Binary classification is a classification problem where the goal is to output 0 or 1 (no or yes) for some condition. As an example, let's say that our problem is classifying an image as either containing a cat, or not containing a cat. To create such a classifier, we can use logistic regression, but first let's define some notation:
- $M$ is the number of training vectors
- $N{x}$ is the size of the input vector
- $N{y}$ is the size of the output vector
- $X(1)$ is the first input vector
- $Y(1)$ is the first output vector
- $X = [x(1), x(2).. x(M)]$
- $Y = (y(1), y(2).. y(M))$

### Logistic Regression
The logistic regression algorithm is used in binary classification.

In an algebra or statistics class, the equation we're used to for logistic regression is

$$
y = wx = b
$$

In machine learning, the x term is usually a vector X. So this equation is represented as

$$
y = W^{T}X+b
$$

The output y needs to be between 0 and 1 so we can get the output of the classification, so we put this equation through the sigmoid function and squish its values between 0 and 1.

$$
y = sigmoid(W^{T}X+b)
$$

### Logistic Regression Cost Function
The function we use for finding the error (that will be used when optimizing) in the logistic regression operation.

$$
L(y',y) = - (y*log(y') + (1-y)*log(1-y'))
$$

To explain this, let's see what would make y equal to 1 and 0:
- if $y = 1$, then $L(y',1) = -log(y')$ so we want to maximize $y'$, and after passing through sigmoid its largest value is $1$
- if $y = 0$, then $L(y',0) = -log(1-y')$ so we want to maximize $1-y'$, and minimize $y'$

The cost function is:

$$
J(w,b) = (1/m) * \sum L(y'[i],y[i])
$$

Note that the **loss** function is the error of a single training example wheras the **cost** function is the error for the entire training set.

### Gradient Descent
Our goal is to minimize the cost function. In calculus, this is analogous to finding the minimum point of the convex cost function.

First we set, or 'initialize', w and b to zero or random values. Then we find the slope dw by taking the derivative of w. The derivative gives us the direction in which to improve our parameters. We want to travel down the slope of w to find the local minimum:

$$w = w - \alpha  * d(J(w,b) / dw)$$
$$b= b- \alpha  * d(J(w,b) / db$$

Pseudo code for logistic regression:

{% highlight js %}
J = 0; dw1 = 0; dw2 =0; db = 0;
w1 = 0; w2 = 0; b=0;
for i = 1 to m
  Forward pass:
  z(i) = W1*x1(i) + W2*x2(i) + b
  a(i) = Sigmoid(z(i))
  J += (Y(i)*log(a(i)) + (1-Y(i))*log(1-a(i)))

Backward pass:
dz(i) = a(i) - Y(i)
dw1 += dz(i) * x1(i)
dw2 += dz(i) * x2(i)
db  += dz(i)
J /= m
dw1/= m
dw2/= m
db/= m

Gradient descent:
w1 = w1 - alpa * dw1
w2 = w2 - alpa * dw2
b = b - alpa * db
{% endhighlight %}

This code has to run for some iterations to get to the minimum, so really there is another for loop.

For loops in python take a long time because they are not optimized at all. NumPy is a python library that allows us to eliminate for loops in matrix manipulation tasks through [vectorization]({% post_url /Deep-Learning-Specialization/1-Neural-Networks-and-Deep-Learning/2017-02-23-Vectorization-NumPy %}).
