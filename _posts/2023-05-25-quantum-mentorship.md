---
title:  "Advanced examples"
mathjax: true
layout: post
categories: media
---
# Intro
A variational quantum algorithm uses a classical approach to optimize the parameters $$\theta$$ of the circuit.

<p align="center">
  <img src="https://raw.githubusercontent.com/JoaoMiguelNC/JoaoMiguelNC.github.io/master/Images/temporary%20circuit.png" />
</p>


![variational circuit](https://raw.githubusercontent.com/JoaoMiguelNC/JoaoMiguelNC.github.io/master/Images/temporary%20circuit.png)

Given a cost function $${\cal L}(\theta)$$

We can use gradiant descent to minimize the cost function:

$$\dot{\theta} = - \epsilon\nabla_\theta{\cal L}(\theta)$$

And we get

$$\dot{\theta}_{k+1} = \theta_k - \epsilon\nabla_\theta{\cal L}(\theta)$$

In [1] the authors propose doing the optimization over the matrix $$U$$ instead of the parameter space.

Now the cost is a function of $$U$$ and we have (meter qualquer coisa que o grad(L) é da forma f(U)U e que por isso vamos usar grad para nos referirmos a f(U)U e a f(U) e que pelo contexto fica claro, neste caso estamos a usar a segunda)

$$\dot{U} = - \epsilon\cdot\mathrm{grad}{\cal L}(U)\cdot U$$

Which leads to

$$U_{k+1} = \exp(- \epsilon\cdot\mathrm{grad}{\cal L}(U_k))$$

PUT IMAGE HERE

# Special Unitary Differential Manifold
The special unitary Lie group is the set of $$n\times n$$ unitary matrices with determinant 1, that is

$$\mathrm{SU}(n)=\{X\in\mathbb{C}^{n\times n}\!: X^\dagger X=I, \det(X)=1\}$$

$$\mathrm{SU}(n)$$ is a differential manifold and given $$U\in\mathrm{SU}(n)$$ the tangent space at $$U$$ is

$$T_U\mathrm{SU}(n)=\{V\in\mathbb{C}^{n\times n}\!: V^\dagger U + U^\dagger V = 0 \ \wedge \ \mathrm{tr}(U^\dagger V)=0\}$$

Thus, the corresponding Lie algebra, $$\mathfrak{su}(n)$$, is the tangent space at the identity

$$\mathfrak{su}(n)=\{\Omega\in\mathbb{C}^{n\times n}\!: \Omega^\dagger=-\Omega \ \wedge \ \mathrm{tr}\,\Omega=0\}$$

# Computing the Gradient

## Cost Function 1

In [1] they compute the gradient for the cost function 

$${\cal L}(U) = \mathrm{tr}(HU\rho_oU^\dagger)$$

Obtaining 

$$\mathrm{grad}{\cal L}(U) = - [U\rho_0U^\dagger, H]U$$

For simplicity we'll call $$\mathrm{grad}{\cal L}(U)$$ to the part $$- [U\rho_0U^\dagger, H]$$ so that it fits nicelly in the equation

$$\dot{U} = - \epsilon\cdot\mathrm{grad}{\cal L}(U)\cdot U$$

## Cost Function 2

We computed the gradient for the cost function used in [2]

$${\cal L}(U) = 1 - \frac{1}{4^n}|\mathrm{tr}(V^\dagger U)|^2$$

And we got

$$\mathrm{grad}{\cal L}(U) = - \frac{1}{4^n}\left(\mathrm{tr}(VU^\dagger)UV^\dagger - \mathrm{tr}(V^\dagger U)VU^\dagger\right)U$$

## Cost Function 3

To find the solution of a linear equation $$Ax=b$$ we can obtimze a unitary $$U$$ such that 
$$U|0\rangle$$
approximates $$|x\rangle$$ using the cost function

$${\cal L}(U) = 1 - \frac{|\langle b|AU|0\rangle|^2}{\lVert AU|0\rangle \rVert^2}$$

For this function the gradient is

$$\mathrm{grad}{\cal L}(U) = \left[ U|0\rangle\langle 0|U^\dagger, \frac{fA^\dagger A - gA^\dagger|b\rangle\langle b|A}{g^2} \right]U$$

where 
$$f := \mathrm{tr}(U|0\rangle\langle 0|U^\dagger A^\dagger|b\rangle\langle b|A)$$ 
and 
$$g := U|0\rangle\langle 0|U^\dagger A^\dagger A$$.

## Cost Function 4
For the mean square error, with $$y_i\in\{-1, 1\}$$, the loss function is

$${\cal L}(U) = \frac{1}{2m}\sum_{i=1}^m \left( \mathrm{tr}(OU\rho_0(x_i)U^\dagger) - y_i \right)^2$$

And the gradient is

$$\mathrm{grad}{\cal L}(U) = -\frac{1}{2m}\left( 2(\mathrm{tr}(OU\rho_0(x_i)U^\dagger) - y_i)[U\rho_0(x_i)U^\dagger, O] \right)U$$

## Cost Function 5
For the cross entropy, with $$y_i\in\{0, 1\}$$, the loss function is

$${\cal L}(U) = -\frac{1}{m}\sum_{i = 1}^m\left(y_i\log f(U; x_i) + (1 - y_i)\log(1 - f(U; x_i))\right)$$

where $$f(U; x_i) := (\mathrm{tr}(OU\rho_0(x_i)U^\dagger) + 1)/2$$

The gradient for this loss function is

$$\mathrm{grad}{\cal L}(U) = \frac{1}{m}\left(\sum_i \frac{T_i + 1 - 2y_i}{T_i^2 - 1}[U\rho_0(x_i)U^\dagger, O] \right)U$$

where $$T_i:=\mathrm{tr}(OU\rho_0(x_i)U^\dagger)$$

# References

[1] Roeland Wiersema and Nathan Killoran. Optimizing quantum circuits with riemannian gradient flow, 2022.

[2] Sumeet Khatri, Ryan LaRose, Alexander Poremba, Lukasz Cincio, Andrew T. Sornborger, and Patrick J. Coles. Quantum-assisted quantum compiling. 2018.
