---
layout: post
title: Normalization, Initializing Weights, and Gradient Checking
tags: Deep-Learning
mathjax: true
categories: Deep-Learning
excerpt:
    <p>This post will discuss some miscellaneous practices in training neural networks, such as normalizing input data, and will also discuss the vanishing/exploding gradient problem and how to solve it with proper weight initialization and gradient checking.</p>
---

This post will discuss some miscellaneous practices in training neural networks, such as normalizing input data, and will also discuss the vanishing/exploding gradient problem and how to solve it with proper weight initialization and gradient checking.

### Normalizing Inputs
Normalizing inputs is a standard practice in training a neural network. The reason it's done is to make the shape of the loss function more consistent, and by extension making optimizing easier. If inputs aren't normalized, the loss function will be deep and its shape will be inconsistent. Thus, normalizing inputs saves a lot of time in the training phase.

Normalization can be done by subtracting the mean from each input and normalizing the variance:
```
Mean = (1/m) * sum(x(i))
X = X - Mean

variance = (1/m) * sum(x(i)^2)
X/= variance
```

Note that although the benefits of this step are seen in training time, the normalization step should happen for all of the datasets (so dev and test in addition to the train dataset).

### Vanishing and Exploding Gradients
When the derivatives of the loss function become very small or very big, we see vanishing or exploding gradients.

To understand this, let's take an example neural network with layers `L` and completely linear activation functions, each bias `b` equal to 0.

Then:
`Y' = W[L]W[L-1].....W[2]W[1]X`
Then, if we have 2 layers, in each layer, we have two assumptions:
`Y' = (W[L][1.5  0]^(L-1)) X = 1.5^L 	# which will be so large
          [0  1.5]`
`Y' = (W[L][0.5  0]^(L-1)) X = 0.5^L 	# which will be so small
          [0  0.5]`

So, if W > 1 the weights will explode, and if W < 1, the weights will vanish

So as the layers of the neural network increase, there is also this increased risk of exploding or vanishing gradients. Still, researchers have trained incredibly deep networks (such as Microsoft's 152 layer ResNet), so there are several ways to deal with the exploding/vanishing gradient problem.

### Weight Initialization for Deep Networks
- A partial solution to the Vanishing / Exploding gradients in NN is better or more careful choice of the random initialization of weights.
- In a single neuron (Perceptron model): `Z = w1X1 + w2X2 + ...+wnXn`
    - So if Nx is large we want W's to be smaller to not explode the cost.
- So it turns out that we need the variance which equals 1/Nx to be the range of W's
- So lets say when we initialize W's we initialize like this (For Tanh its better to use this):
```
np.random.rand(shape)*np.sqrt(1/n[l-1])               #n[l-1] In the multiple layers.
```
- Setting this to 2/n[l-1] especially for RELU is better:
```
np.random.rand(shape)*np.sqrt(2/n[l-1])               #n[l-1] In the multiple layers.
```
- This is the best way to solve Vanishing / Exploding gradients (RELU + Weight Initialization with variance)
- The initialization in this video is called "He Initialization / Xavier Initialization"

### Gradient Check
There are other ways to solve the exploding and vanishing gradient problems (such as residual blocks), which are more advanced and will come later in the series of notes. In the mean time, another important way to diagnose gradient computation problems that could lead to exploding/vanishing gradient is the gradient check.

Instead of calculating the gradient 'analytically' by solving for the derivative $f'(x)$ of the selected operation $f(x)$, we can check the gradient calculation with a numerical gradient by going back to the definition of the derivative:

$$
f'(x) =  \lim_{h \rightarrow 0}  \frac{f(x+h)-f(x)}{h}
$$

In actual code, this check would work as follows:
1. First take `W[1],b[1]...W[L]b[L]` and reshape into one big vector `Ceta`
2. The cost function will be `L(Ceta)`
3. Then take `dW[1],db[1]......dW[L]db[L]` into one big vector `d_ceta`

Algorithm:
```
eps = 10^-7   #Small number
for i in len(Ceta):
	d_ceta_calc[i] = (J(ceta1,..,ceta[i] + eps) -  J(ceta1,..,ceta[i] - eps)) / 2*eps
```

Finally, check this formula: `(||d_ceta_calc - d_ceta||) / (||d_ceta_calc||+||d_ceta||)`
- The `||` is the Euclidean distance:
$$
||a,b|| =  \sqrt{ \sum_{i=0}^n (b_{i}^2 - a_{i}^2)^2 }
$$

#### Some more notes on gradient checking

Use the gradient check for debugging, so once in a while just to check that calculations are working out.
- this is because calculating the numerical gradient every time, although very accurate, is very computationally expensive
- Remember if you use the normal Regularization to add the value of the additional check to your equation  `(lamda/2m)sum(W[l])`
- Gradient checking doesn't work with dropout.
- Run gradient checking at random initialization and train the network for a while maybe there's a bug that are not on the first iteration.
