---
layout: post
title: Vectorization in NumPy
tags: Deep-Learning
mathjax: true
excerpt:
    <p>Vectorization in Python's NumPy library allows us to eliminate for loops and speed up the computation speed of large matrix manipulation processes.</p>
---

Vectorization in Python's NumPy library allows us to eliminate for loops and speed up the computation speed of large matrix manipulation processes.

Vectorization is really important because deep learning works best with large datasets. If we dealed with these datasets using Python for loops, we would be waiting for a long time. NumPy is a python library that has mostly vectorized functions.

Vectorization will speed up computations on CPU and GPU, but it's a lot faster on GPU.

### Vectorizing Logistic Regression
input: matrix $X$ of dimensions $[N_x, m]$
output: matrix $Y$ of dimensions $[N_y, m]$

We will compute $Y = [z_1, z_2, ... z_m] = W' * X + [b,b,...b]$. In Python, with NumPy, we write this as:

{% highlight js %}
Z = np.dot(W.T,X) + b
{% endhighlight %}

Now, the shape of $Z$ is $(1,m)$. Our next step is to pass this into the sigmoid activation function:

{% highlight js %}
A = 1 / 1 + np.exp(-Z)
{% endhighlight %}

The resultant matrix $A$, has shape $(1, m)$. Now we need to vectorize the logistic regression's gradient output.

{% highlight js %}
dz = A - Y              # Shape of dz is (1, m)
dw = np.dot(X,dz.T)/m   # Shape of dw is (N_x, 1)
db = dz.sum()/m         # Shape of db is (1,1)
{% endhighlight %}

### Other notes on Python and NumPy
In NumPy, `obj.sum(axis = 0)` sums the columns while `obj.sum(axis = 1)` sums the rows.
In NumPy, `obj.reshape(1,4)` changes the shape of the matrix by broadcasting the values.

Broadcasting works when matrix dimensions don't match. When this happens, NumPy will automatically make the shapes work for the operation by broadcasting the values.

The derivative of a sigmoid:

{% highlight js %}
s = sigmoid(x)
ds = s * (1 - s)       # derivative  using calculus
{% endhighlight %}

So, to summarize the major steps in building a neural network, we have:
- Define the model structure (number of inputs and outputs)
- Initialize the parameters
- Loop for some iterations
	- calculate the current loss (forward propogation)
	- calculate the current gradient (backward propogation)
	- update parameters (gradient descent)