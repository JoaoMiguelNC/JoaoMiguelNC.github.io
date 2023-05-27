---
title:  "Advanced examples"
mathjax: true
layout: post
categories: media
---
A variational quantum algorithm uses a classical approach to optimize the parameters $$\theta$$ of the circuit.

[variational circuit](https://raw.githubusercontent.com/JoaoMiguelNC/JoaoMiguelNC.github.io/master/Images/temporary%20circuit.png)

Given a cost function $${\cal L}(\theta)$$

We can use gradiant descent to minimize the cost function:

$$\dot{\theta}_{k+1} = \theta_k - \epsilon\nabla_\theta{\cal L}(\theta)$$
