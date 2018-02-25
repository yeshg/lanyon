---
layout: post
title: Shallow Neural Networks
tags: Deep-Learning
mathjax: true
excerpt:
    <p>Shallow neural networks have one hidden layer. Another way to look at them is a stack of logistic regression objects.</p>
---

Shallow neural networks have one hidden layer. Another way to look at them is a stack of logistic regression objects.


### Neural Networks Overview
In logistic regression we had 

```
X1  \  
X2   ==>  z = XW + B ==> a = Sigmoid(z) ==> l(a,Y)
X3  /
```

In 1-layer neural networks we have
```
X1  \  
X2   =>  z1 = XW1 + B1 => a1 = Sigmoid(a1) => z2 = a1W2 + B2 => a2 = Sigmoid(z2) => l(a2,Y)
X3  /
```
`x` is the input vector `(X1, X2, X3)`, and `Y` is the `(1,1)` output

### Neural Network Representation
- We will define the neural networks that has one hidden layer.
- NN contains of input layers, hidden layers, output layers.
- Hidden layer means we cant see that layers in the training set.
- `a0 = x` (the input layer)
- `a1` will represent the activation of the hidden neurons.
- `a2` will represent the output layer.
- We are talking about 2 layers NN. The input layer isn't counted.

### Computing a Neural Network's Output
Let's continue with 3 inputs $x_1, x_2, x_3$, and lets compute the output of a shallow neural network with 4 hidden neurons.
![](Images/computeShallowNN.png)

#### Now let's write some code:
Pseudo code for forward propogation for the 2 layer NN:
{% highlight js %}
for i = 1 to m
	z[1, i] = W1*x[i] + b1        # shape of z[1, i] is (noOfHiddenNeurons,1)
	a[1, i] = sigmoid(z[1, i])    # shape of a[1, i] is (noOfHiddenNeurons,1)
	z[2, i] = W2*a[1, i] + b2     # shape of z[2, i] is (1,1)
	a[2, i] = sigmoid(z[2, i])    # shape of a[2, i] is (1,1)
{% endhighlight %}

Lets say we have `X` on shape `(Nx ,m)`. So the new pseudo code

{% highlight js %}
Z1 = W1X + b1       # shape of Z1 (noOfHiddenNeurons,m)
A1 = sigmoid(Z1)    # shape of A1 (noOfHiddenNeurons,m)
Z2 = W2A1 + b2	    # shape of Z2 is (1,m)
A2 = sigmoid(Z2)    # shape of A2 is (1,m)
{% endhighlight %}

If you notice always m is the number of columns.

In the last example we can call `X`, `A0` for instance:

{% highlight js %}
Z1 = W1A0 + b1      # shape of Z1 (noOfHiddenNeurons,m)
A1 = sigmoid(Z1)    # shape of A1 (noOfHiddenNeurons,m)
Z2 = W2A1 + b2	    # shape of Z2 is (1,m)
A2 = sigmoid(Z2)    # shape of A2 is (1,m)
{% endhighlight %}

### Activation Functions

tbd, add pics from cs231n

