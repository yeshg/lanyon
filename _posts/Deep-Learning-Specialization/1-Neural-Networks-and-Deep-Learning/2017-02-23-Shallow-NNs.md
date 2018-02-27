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

Activation functions are essential for neural networks to excel in complex tasks. More specifically, non-linear activation functions allow the neural network to act more like a polynomial than linear logistic regression.

#### sigmoid

So far, we've been using the sigmoid activation function to squish the outputs of our computations between 0 and 1, but this is actually outdated. The sigmoid activation function shown below leads to three problems:
1. Saturated neurons "kill" the gradients: the horizontal left and right portions of the sigmoid causes the gradient to become very small, and as a result updates slow down.
2. Sigmoid outputs are not zero-centered: we want zero-centered activations because they allow the gradients on the weight matrix to be both positive and negative, rather than all positive or negative.
3. the exp() computation is a bit expensive to execute.

$$
\sigma (x) =  \frac{1}{1+ e^{-x} } 
$$

![](Images/sigmoid.jpeg)

While the coursera class talks about the vanishing gradient issue, Stanfords CS231n goes into more details on the other problems with the sigmoid and other activation functions.

#### tanh

The tanh activation function shown below solves some of the problems with the sigmoid - the activaions are now zero-centered, and there isn't any computationally expensive exp() operation. Still, the tanh function has very horizontal end-behavior, and will kill the gradients when the neurons are saturated.

$$
f(x) = tanh(x)
$$

![](Images/tanh.jpeg)

#### ReLU

The most popular activation function in modern deep learning is the Rectified Linear Unit (ReLU):
$$
f(x) = max(0,x)
$$
This activation function does not saturate (in the positive region), is very computationally efficient, and has been shown to lead to faster convergence than sigmoid and tanh (one research paper reported it to be 6 times faster).
The only downside of this function is that it's no longer zero-centered, which can be annoying to deal with.

![](Images/relu.jpeg)

Looking at the graph above, we see that the negative portion of the relu will always output 0. While this is fine for most cases, it's possible that a large gradient flowing through a ReLU neuron could cause the weights to update in such a way that the neuron will never activate on any datapoint again. The gradient flowing through that point would always be 0, because the ReLU will keep on outputting the same value (0). Usually this is avoidable with a properly set learning rate.

To visualize the dying relu problem, one of the slides from CS231n shows how a dead ReLU unit that has learned a large negative bias term for its weights takes no role in discriminating between inputs. It's operating in a decision plane outside all of the possible input data. This is what makes a dead ReLU unit irreversible (while saturated sigmoid and tanh units still can come back because there is always a small gradient).

![](Images/deadReLU.png)

#### Leaky ReLU

Although less popular than ReLU, the Leaky ReLU adds a slight positive slope to the negative region of the ReLU that prevents the dead ReLU problem. In addition, it enjoys the other benefits of ReLU in computational efficiency, speedy convergence, and resitance to neuron saturation.

$$
f(x) = max(0.01x,x)
$$

![](Images/leakyRelu.png)

#### Other Activation Functions

There are a lot of other interesting activation functions, and literature keeps getting updated with experiments on their performance and on new activation functions. One I'd like to explore a little more is maxout, which is really just a generalization of the ReLU and Leaky ReLU.

In a maxout neuron, rather than computing the output with $f(w^ix+b)$, where $f$ is the activation function, the following functional form is used:

$$
\max(w_1^ix+b_1, w_2^ix + b_2)
$$

At a closer glance, we see that ReLU and Leaky ReLU are a special case of this more generalized form. Foe example, in ReLU we have $w_1,b_1=0$. This gives maxout the same benefits of both ReLu and Leaky ReLU, but it doubles the number of parameters for each neuron, leading to a high number of parameters to learn.

#### TLDR of all the activation functions

General consensus in the deep learning community is to use ReLu with carefully set learning rates and weight initializations. If the dead ReLU problem comes up then options like Leaky ReLu or Maxout should be tried. Tanh works fine, but would probably be slower than the others, and avoid using sigmoid.

### Derivatives of the activation functions

#### Sigmoid:
{% highlight js %}
g(z) = 1 / (1 + np.exp(-z))
g'(z) = (1 / (1 + np.exp(-z))) * (1 - (1 / (1 + np.exp(-z))))
g'(z) = g(z) * (1 - g(z))
{% endhighlight %}

#### Tanh:
{% highlight js %}
g(z)  = (e^z - e^-z) / (e^z + e^-z)
g'(z) = 1 - np.tanh(z)^2 = 1 - g(z)^2
{% endhighlight %}

#### ReLU:
{% highlight js %}
g(z)  = np.maximum(0,z)
g'(z) = { 0  if z<0
		  1  if z>=0  }
{% endhighlight %}

#### Leaky ReLU:
{% highlight js %}
g(z)  = np.maximum(0.01 * z, z)
g'(z) = { 0.01  if z<0
				  1     if z>=0   }
{% endhighlight %}

### Gradient Descent for Neural Networks
{% highlight js %}
Repeat:
		Compute predictions (y'[i], i = 0,...m)
		Get derivatives: dW1, db1, dW2, db2
		Update: W1 = W1 - LearningRate * dW1
				b1 = b1 - LearningRate * db1
				W2 = W2 - LearningRate * dW2
				b2 = b2 - LearningRate * db2
{% endhighlight %}

#### Forward Propagation:
{% highlight js %}
Z1 = W1A0 + b1    # A0 is X
A1 = g1(Z1)
Z2 = W2A1 + b2
A2 = Sigmoid(Z2)      # Sigmoid because the output is between 0 and 1
{% endhighlight %}

#### Backward Propagation:
{% highlight js %}
dZ2 = A2 - Y      # derivative of cost function we used * derivative of the sigmoid function
dW2 = (dZ2 * A1.T) / m
db2 = Sum(dZ2) / m
dZ1 = (W2.T * dZ2) * g'1(Z1)  # element wise product (*)
dW2 = (dZ1 * A0.T) / m   # A0 = X
db2 = Sum(dZ1) / m
{% endhighlight %}

### Random Initialization
In logistic regression we didn't need to randomly initialize. But in NNs we have to initialize randomly because if all weights start out as the same then all of the hidden units will be updated the same.
{% highlight js %}
W1 = np.random.randn((2,2)) * 0.01   #0.01 to make it small enough so it doesn't saturate sigmoid
b1 = np.zeros((2,1))   # its ok to have b as zero, it won't get us to the symmetry problem.
{% endhighlight %}