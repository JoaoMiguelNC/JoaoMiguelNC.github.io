---
title:  "Advanced examples"
mathjax: true
layout: post
categories: media
---
# Intro
A variational quantum algorithm uses a classical approach to optimize the parameters $$\theta$$ of the circuit.

![variational circuit](https://raw.githubusercontent.com/JoaoMiguelNC/JoaoMiguelNC.github.io/master/Images/temporary%20circuit.png)

Given a cost function $${\cal L}(\theta)$$

We can use gradiant descent to minimize the cost function:

$$\dot{\theta}_{k+1} = \theta_k - \epsilon\nabla_\theta{\cal L}(\theta)$$

In [1] the authors propose doing the optimization over the matrix $$U$$ instead of the parameter space.

# Special Unitary Differential Manifold
The special unitary Lie group is the set of $$n\times n$$ unitary matrices with determinant 1, that is

$$\mathrm{SU}(n)=\{X\in\mathbb{C}^{n\times n}\!: X^\dagger X=I, \det(X)=1\}$$

$$\mathrm{SU}(n)$$ is a differential manifold and given $$U\in\mathrm{SU}(n)$$ the tangent space at $$U$$ is

$$T_U\mathrm{SU}(n)=\{V\in\mathbb{C}^{n\times n}\!: V^\dagger U + U^\dagger V = 0 \ \wedge \ \mathrm{tr}(U^\dagger V)=0\}$$

Thus, the corresponding Lie algebra, $$\mathfrak{su}(n)$$, is the tangent space at the identity

$$\mathfrak{su}(n)=\{\Omega\in\mathbb{C}^{n\times n}\!: \Omega^\dagger=-\Omega \ \wedge \ \mathrm{tr}\,\Omega=0\}$$

# References
[1] Roeland Wiersema and Nathan Killoran. Optimizing quantum circuits with riemannian gradient flow, 2022.
