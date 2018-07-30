---
layout: post
title: Machine Learning in Practice
tags: Deep-Learning
categories: [Deep-Learning]
excerpt:
    <p>The <a href="https://www.coursera.org/specializations/deep-learning">Deep Learning Specialization</a> on Coursera is one of the most popular online resources for learning machine learning. After following Andrej Karpathy’s CS231n (another popular course for learning machine learning) and self-studying deep learning from a variety of research papers and websites, I think that Andrew Ng’s Coursera specialization is the most comprehensive and structured way to become a deep learning expert without going to a university. The purpose of the following posts is to summarize the key points of the specialization as an extended guide/cheatsheet for deep learning.</p>
---

### Overview
Most of the work in deep learning lies in tuning and choosing the correct model based on the use cases of that model. Often times, the development and testing of these machine learning models requries large teams, and employing the right principles can save lots of precious work time.

There are four major goals to structuring machine learning projects:
1. Understand how to diagnose errors in a machine learning system
2. Be able to identify the most promising steps to take for reducing error
3. Understand the effects of changing hperparameters, and how a machine learning system compares to human-level performance
4. Know how to apply end-to-end learning, transfer learning, and multi-task learning.

- Different ways to improve the accuracy of the deep learning system:
  - Collect more data.
  - Collect more diverse training set.
  - Train algorithm longer with gradient descent.
  - Try different optimization algorithm.
  - Try bigger/smaller network.
  - Try dropout.
  - Add L2 regularization.
  - Change network architecture (activation functions, # of hidden units, etc.)

### Orthogonalization vs Single Number Evaluation Metric
- Some DL developers know exactly what to change through process called orthogonalization, but this process is very slow
  - Basic idea is to have knob-like controls that do a specific task and don't affect anything else
  - Example orthogonalization workflow:
      1. Fit training set well on cost function, trying to get near human level performance. If not, then try increasing the size of the network or changing the optimization algorithm
      2. Fit dev set well on cost function, and if not then try tweaking regularization and the training set
      3. Next, work down to test set, and if that doesn't work out try increasing size of dev set
      4. Finally, apply system in real world, and if it doesn't work well change dev set or cost function

Rather than going through this, a better practice is using a single number to evaluate the entire project. An important concept in this "single number evaluation metric" based tuning is understanding the difference  between precision and recall.
- The precision metric is the percentage of correctly predicted outcomes over all outcomes with a positive prediction
- Recall is the percentage of correctly predicted outcomes over all outcomes with any prediction
- Accuracy is percentage of right predictions over all predictions
Combining the precision and recall metrics into a single score is better, this is called the `F1` score:
```
F1 = 2 / ((1/P) + (1/R))
``

### Managing Train/Dev/Test Distributions
- Make sure dev and test sets come from the same distribution
- Old way of splitting data was 70% training, 30% test or 60% training, 20% dev, 20% test.
  - This is only valid for number of examples < 100,000
- In modern deep learning, where big data is prevalent, a  reasonable split for a million+ examples is 98% training, 1% dev, 1% test`


### Why Compete and Compare with Human Level Performance
There are two main reasons:
- Advances in deep learning have caused machine learning systems to be competitive with humans in a variety of tasks
- More importantly, the workflow of designing and building a machine learning system is more efficient when that task is something humans can also do.
![](bayes.png)

As seen in image above, there is some error, 'Bayes optimal error' that no ML system will surpass. There isn't much of a gap between human error and Bayes optimal error in a lot of tasks. So in these tasks it makes sense
- get labeled data from humans
- gain insight from manual error analysis
- better analysis of bias and variance

For natural perception tasks, humans are generally better than ML systems. For other tasks such as Online advertising, product recommendation, and loan approval, ML and DL systems are actually better than humans.

### Avoidable bias

- Suppose that the cat classification algorithm gives these results:

  | Humans             | 1%   | 7.5% |
  | ------------------ | ---- | ---- |
  | **Training error** | 8%   | 8%   |
  | **Dev Error**      | 10%  | 10%  |
  - In the left example, because the human level error is 1% then we have to focus on the **bias**.
  - In the right example, because the human level error is 7.5% then we have to focus on the **variance**.
  - The human-level error as a proxy (estimate) for Bayes optimal error. Bayes optimal error is always less (better), but human-level in most cases is not far from it.
  - You can't do better then Bayes error unless you are overfitting.
  - `Avoidable bias = Training error - Human (Bayes) error`
  - `Variance = Dev error - Training error`

### Improving your model performance

- The two fundamental asssumptions of supervised learning:
  1. You can fit the training set pretty well. This is roughly saying that you can achieve low **avoidable bias**. 
  2. The training set performance generalizes pretty well to the dev/test set. This is roughly saying that **variance** is not too bad.
- To improve your deep learning supervised system follow these guidelines:
  1. Look at the difference between human level error and the training error - **avoidable bias**.
  2. Look at the difference between the dev/test set and training set error - **Variance**.
  3. If **avoidable bias** is large you have these options:
     - Train bigger model.
     - Train longer/better optimization algorithm (like Momentum, RMSprop, Adam).
     - Find better NN architecture/hyperparameters search.
  4. If **variance** is large you have these options:
     - Get more training data.
     - Regularization (L2, Dropout, data augumentation).
     - Find better NN architecture/hyperparameters search.

