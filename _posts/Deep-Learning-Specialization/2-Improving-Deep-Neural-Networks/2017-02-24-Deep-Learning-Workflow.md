---
layout: post
title: Deep Learning Workflow
tags: Deep-Learning
mathjax: true
categories: Deep-Learning
excerpt:
    <p>A major part of deep learning is hyperparameter tuning. It's impossible to get all of the hyperparameters on the first run of a model, so in real deep learning projects, there is an iterative loop or workflow before a final system is deployed.</p>
---

A major part of deep learning is hyperparameter tuning. It's impossible to get all of the hyperparameters on the first run of a model, so in real deep learning projects, there is an iterative loop or workflow before a final system is deployed:
```
Idea --> Code --> Experiment
```
Before finding the hyperparameters of the final system, a deep learning researcher/programmer must go through the loop several times.
To help the workflow, the labeled datasets are split into three parts:
- training set (much larger than the others)
- dev set
- testing set

Basic idea is to train a model on the training set such that the dev set has very high accuracy. Then, once model is ready it is evaluated on the testing set.

General trends on splitting the datasets:
- If the size of the dataset is 100 to 10,000,00 --> 60/20/20 split
- If size is above that --> 99+/1-/1- (so for example, 99.5/0.25/0.25)

Main takeaway is that the training set is where models get really good.

One thing that's important in splitting the data is to make sure the test, dex, and training datasets come from the same distribution/source.

It's ok to have a dev set without a training set. Unfortunately, this is commonly referred to as a train/test split, which is inaccurate because it's really a train/dev split.

### Bias vs. Variance
Identifying Bias and Variance is critical to knowing how to improve a model.
- If your model is underfitting (logistic regression of non linear data) it has a "high bias"
- If your model is overfitting then it has a "high variance"
- Your model will be alright if you balance the Bias / Variance
- see the image below for a visualization of this:

![](Images/biasAndVariance.png)

Many times, we don't have a nice 2D plot to visualize our model's performance. We can still tell the bias and variance from the error percentages on the train, dev, and test datasets:
- High variance (overfitting) for example:
    - Training error: 1%
    - Dev error: 11%
- high Bias (underfitting) for example:
    - Training error: 15%
    - Dev error: 14%
- high Bias (underfitting) and High variance (overfitting) for example:
    - Training error: 15%
    - Test error: 30%
- Best:
    - Training error: 0.5%
    - Test error: 1%
- These Assumptions came from that human has 0% error. If the problem isn't like that you'll need another approach.

### Basic Recipie for Machine Learning
- If the model has high bias:
  - Try to make your NN bigger (Size of Hidden units, Number of layers)
  - Try a different model that are suitable for your data.
  - Try to run it longer.
  - Different optimization algorithm.
- If the model has high variance:
  - More data.
  - Try regularization.
  - Try a different model that are suitable for your data.
- You should try the previous two points until you have a low bias / low variance.
- In the older days before deep learning there was a "Bias / variance trade off". But because now you have more options on solving the bias of variance problem its really helpful to use deep learning.
- Training a bigger neural network never hurt.

As previosuly mentioned, these posts are based off the Deep Learning Specialization on Coursera, the notes graciously provided by [mbadry1](https://github.com/mbadry1/DeepLearning.ai-Summary)
