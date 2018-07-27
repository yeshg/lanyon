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

- You have a lot of ideas for how to improve the accuracy of your deep learning system:
  - Collect more data.
  - Collect more diverse training set.
  - Train algorithm longer with gradient descent.
  - Try different optimization algorithm (e.g. Adam).
  - Try bigger network.
  - Try smaller network.
  - Try dropout.
  - Add L2 regularization.
  - Change network architecture (activation functions, # of hidden units, etc.)
