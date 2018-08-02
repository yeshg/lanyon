---
layout: post
title: Neural Style Transfer
tags: Deep-Learning
mathjax: true
categories: Deep-Learning
excerpt: <p> How to use gram matrices and convolutional neural networks in an interesting application of machine learning in artwork called Neural Style Transfer. </p>
---
### Neural Style Transfer

#### What is neural style transfer?

- Neural style transfer is one of the application of Conv nets.
- Neural style transfer takes a content image `C` and a style image `S` and generates the content image `G` with the style of style image.
- ![](Images/37.png)
- In order to implement this you need to look at the features extracted by the Conv net at the shallower and deeper layers.
- It uses a previously trained convolutional network like VGG, and builds on top of that. The idea of using a network trained on a different task and applying it to a new task is called transfer learning.

#### What are deep ConvNets learning?

- Visualizing what a deep network is learning:
  - Given this AlexNet like Conv net:
    - ![](Images/38.png)
  - Pick a unit in layer l. Find the nine image patches that maximize the unit's activation. 
    - Notice that a hidden unit in layer one will see relatively small portion of NN, so if you plotted it it will match a small image in the shallower layers while it will get larger image in deeper layers.
  - Repeat for other units and layers.
  - It turns out that layer 1 are learning the low level representations like colors and edges.
- You will find out that each layer are learning more complex representations.
  - ![](Images/39.png)
- The first layer was created using the weights of the first layer. Other images are generated using the receptive field in the image that triggered the neuron to be max.
- [[Zeiler and Fergus., 2013, Visualizing and understanding convolutional networks]](https://arxiv.org/abs/1311.2901)
- A good explanation on how to get **receptive field** given a layer:
  - ![](Images/receptiveField.png)
  - From [A guide to receptive field arithmetic for Convolutional Neural Networks](https://medium.com/@nikasa1889/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807)

#### Cost Function

- We will define a cost function for the generated image that measures how good it is.
- Give a content image C, a style image S, and a generated image G:
  - `J(G) = alpha * J(C,G) + beta * J(S,G)`
  - `J(C, G)` measures how similar is the generated image to the Content image.
  - `J(S, G)` measures how similar is the generated image to the Style image.
  - alpha and beta are relative weighting to the similarity and these are hyperparameters.
- Find the generated image G:
  1. Initiate G randomly
     - For example G: 100 X 100 X 3
  2. Use gradient descent to minimize `J(G)`
     - `G = G - dG`  We compute the gradient image and use gradient decent to minimize the cost function.
- The iterations might be as following image:
  - To Generate this:
    - ![](Images/40.png)
  - You will go through this:
    - ![](Images/41.png)

#### Content Cost Function

- In the previous section we showed that we need a cost function for the content image and the style image to measure how similar is them to each other.
- Say you use hidden layer `l` to compute content cost. 
  - If we choose `l` to be small (like layer 1), we will force the network to get similar output to the original content image.
  - In practice `l` is not too shallow and not too deep but in the middle.
- Use pre-trained ConvNet. (E.g., VGG network)
- Let `a(c)[l]` and `a(G)[l]` be the activation of layer `l` on the images.
- If `a(c)[l]` and `a(G)[l]` are similar then they will have the same content
  - `J(C, G) at a layer l = 1/2 || a(c)[l] - a(G)[l] ||^2`

#### Style Cost Function

- Meaning of the ***style*** of an image:
  - Say you are using layer l's activation to measure ***style***.
  - Define style as correlation between **activations** across **channels**. 
    - That means given an activation like this:
      - ![](Images/42.png)
    - How correlate is the orange channel with the yellow channel?
    - Correlated means if a value appeared in a specific channel a specific value will appear too (Depends on each other).
    - Uncorrelated means if a value appeared in a specific channel doesn't mean that another value will appear (Not depend on each other)
  - The correlation tells you how a components might occur or not occur together in the same image.
- The correlation of style image channels should appear in the generated image channels.
- Style matrix (Gram matrix):
  - Let `a(l)[i, j, k]` be the activation at l with `(i=H, j=W, k=C)`
  - Also `G(l)(s)` is matrix of shape `nc(l) x nc(l)`
    - We call this matrix style matrix or Gram matrix.
    - In this matrix each cell will tell us how correlated is a channel to another channel.
  - To populate the matrix we use these equations to compute style matrix of the style image and the generated image.
    - ![](Images/43.png)
    - As it appears its the sum of the multiplication of each member in the matrix.
- To compute gram matrix efficiently:
  - Reshape activation from H X W X C to HW X C
  - Name the reshaped activation F.
  - `G[l] = F * F.T`
- Finally the cost function will be as following:
  - `J(S, G) at layer l = (1/ 2 * H * W * C) || G(l)(s) - G(l)(G) ||`
- And if you have used it from some layers
  - `J(S, G) = Sum (lamda[l]*J(S, G)[l], for all layers)`
- Steps to be made if you want to create a tensorflow model for neural style transfer:
  1. Create an Interactive Session.
  2. Load the content image.
  3. Load the style image
  4. Randomly initialize the image to be generated
  5. Load the VGG16 model
  6. Build the TensorFlow graph:
     - Run the content image through the VGG16 model and compute the content cost
     - Run the style image through the VGG16 model and compute the style cost
     - Compute the total cost
     - Define the optimizer and the learning rate
  7. Initialize the TensorFlow graph and run it for a large number of iterations, updating the generated image at every step.

#### 1D and 3D Generalizations

- So far we have used the Conv nets for images which are 2D.
- Conv nets can work with 1D and 3D data as well.
- An example of 1D convolution:
  - Input shape (14, 1)
  - Applying 16 filters with F = 5 , S = 1
  - Output shape will be 10 X 16
  - Applying 32 filters with F = 5, S = 1
  - Output shape will be 6 X 32
- The general equation `(N - F)/S + 1` can be applied here but here it gives a vector rather than a 2D matrix.
- 1D data comes from a lot of resources such as waves, sounds, heartbeat signals. 
- In most of the applications that uses 1D data we use Recurrent Neural Network RNN.
- 3D data also are available in some applications like CT scan:
  - ![](Images/44.png)
- Example of 3D convolution:
  - Input shape (14, 14,14, 1)
  - Applying 16 filters with F = 5 , S = 1
  - Output shape (10, 10, 10, 16)
  - Applying 32 filters with F = 5, S = 1
  - Output shape will be (6, 6, 6, 32)
