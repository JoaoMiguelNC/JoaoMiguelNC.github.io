---
title:  "QOSF Quantum Computing Mentorship Program"
mathjax: true
layout: post
categories: media
---
# Introduction

Quantum supremacy is the goal that a quantum computer can solve a problem that no classical computer can solve in a feasible amount of time. Some areas where this speed up would occur are factoring integers, simulation a quantum system, or solving linear equations. However, quantum computers are still very sensitive to errors, preventing the implementation of these complex tasks.  

There are, however, areas where simple quantum algorithms are being used, for example, quantum key distribution. But for anything complex, classical computers are still our go to. This forms the basis of Noisy Intermediate Scale Quantum (NISQ) computing - using quantum computers where they can be used and then outsource difficult calculations to a classical computer.   

Variational Quantum Algorithms (VQA) is a type of Noisy Intermediate Scale Quantum (NISQ) computing. The objective of a VQA is to approximate some value using a parameterized quantum circuit by using a quantum computer to calculate quantum values and passing them on to a classical optimization algorithm to improve the parameters. 


In [1] the authors take a different approach, instead of optimizing over the parameter space, they optimize directly over the special unitary group using Riemann gradient flow. They derived the Riemann gradient for the Variational Quantum Eigensolver (VQE) and build circuits that could compute it.  

For QOSF Quantum Computing Mentorship Program we expanded on the previous mentioned work and derived the gradient flow for other cost functions.

# Variational Quantum Algorithm

A VQA is used when we want to aproximate a quantum operator $$U$$ to estimate a value 
$$U|0\rangle$$. 
For that we consider a parameterized quantum circuit $$U(\theta)$$ whose parrameter $$\theta$$ is optimized using a classical algorithm with a relevant cost function $${\cal L}(\theta)$$ (or its gradient $$\nabla{\cal L}(\theta)$$) calculated using a quantum computer. 

![variational circuit](https://raw.githubusercontent.com/JoaoMiguelNC/JoaoMiguelNC.github.io/master/Images/U%20theta%20circuit.png)

For the optimization we are interested in the gradient descent. Thus we can minimize the cost function considering the flow

$$\dot{\theta} = - \nabla{\cal L}(\theta)$$

And we have a recursive method of calculating the next iteration of $$\theta$$

$$\theta_{k+1} = \theta_k - \epsilon\nabla{\cal L}(\theta)$$

# Riemann Gradient Flow

In [1] the authors propose doing the optimization over the matrix $$U$$ instead of the parameter space.

Now the cost is a function of $$U$$ and it turns out that the gradient can be written in the form $$f(U)U$$. With this in mind, we have

$$\dot{U} = - f(U)U$$

Which leads to

$$U_{k+1} = \exp(- \epsilon f(U_k))U_k$$

Which means that to update the circuit for $$U$$, we just need to append the exponential as can been seen in the following figure taken from [1] where the authors are identifying the gradient with just the $$f(U)$$ part.

![update circuit](https://raw.githubusercontent.com/JoaoMiguelNC/JoaoMiguelNC.github.io/master/Images/Updating%20circuit.png)

# Computing the Gradient

For the mentorship program, we redirived the gradient in [1] and derived other gradiants based on different cost functions, and tested the results using Python.

### Variational Quantum Eigensolver

In [1] the authors compute the gradient for the cost function of the variational quantum eigensolver:

$${\cal L}(U) = \mathrm{tr}(HU\rho_oU^\dagger)$$

Obtaining 

$$\mathrm{grad}{\cal L}(U) = - [U\rho_0U^\dagger, H]U$$

### Quantum-Assisted Quantum Compiling

We computed the gradient for the cost function used in quantum-assisted quantum compiling (see [2])

$${\cal L}(U) = 1 - \frac{1}{4^n}|\mathrm{tr}(V^\dagger U)|^2$$

And we got

$$\mathrm{grad}{\cal L}(U) = - \frac{1}{4^n}\left(\mathrm{tr}(VU^\dagger)UV^\dagger - \mathrm{tr}(V^\dagger U)VU^\dagger\right)U$$

### Solving Linear Equations

To find the solution of a linear equation $$Ax=b$$ we can optimize a unitary $$U$$ such that 
$$U|0\rangle$$
approximates $$|x\rangle$$ using one cost function introduced in [3]

$${\cal L}(U) = 1 - \frac{|\langle b|AU|0\rangle|^2}{\lVert AU|0\rangle \rVert^2}$$

For this function, the gradient is

$$\mathrm{grad}{\cal L}(U) = \left[ U|0\rangle\langle 0|U^\dagger, \frac{fA^\dagger A - gA^\dagger|b\rangle\langle b|A}{g^2} \right]U$$

where 
$$f := \mathrm{tr}(U|0\rangle\langle 0|U^\dagger A^\dagger|b\rangle\langle b|A)$$ 
and 
$$g := U|0\rangle\langle 0|U^\dagger A^\dagger A$$.

### Mean Square Error
For the mean square error, with $$y_i\in\{-1, 1\}$$, the loss function is

$${\cal L}(U) = \frac{1}{2m}\sum_{i=1}^m \left( \mathrm{tr}(OU\rho_0(x_i)U^\dagger) - y_i \right)^2$$

And the gradient is

$$\mathrm{grad}{\cal L}(U) = -\frac{1}{2m}\left( 2(\mathrm{tr}(OU\rho_0(x_i)U^\dagger) - y_i)[U\rho_0(x_i)U^\dagger, O] \right)U$$

### Cross Entropy
For the cross entropy, with $$y_i\in\{0, 1\}$$, the loss function is

$${\cal L}(U) = -\frac{1}{m}\sum_{i = 1}^m\left(y_i\log f(U; x_i) + (1 - y_i)\log(1 - f(U; x_i))\right)$$

where $$f(U; x_i) := (\mathrm{tr}(OU\rho_0(x_i)U^\dagger) + 1)/2$$

The gradient for this loss function is

$$\mathrm{grad}{\cal L}(U) = \frac{1}{m}\left(\sum_i \frac{T_i + 1 - 2y_i}{T_i^2 - 1}[U\rho_0(x_i)U^\dagger, O] \right)U$$

where $$T_i:=\mathrm{tr}(OU\rho_0(x_i)U^\dagger)$$

# Code and Graphs
For the quantum compiling, solving linear equations, and mean square error cases we wrote code to test the gradient functions we derived.

### Code for "Quantum Compiling"

For quantum compiling we tested it for a 6-qubit.

```python
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import unitary_group
from scipy.linalg import expm

# Useful Functions

def dag(X):
  return X.conj().T

# Cost and Gradient Functions

def cost(A, B, d):
    return 1.0 - (np.abs(np.trace(dag(A) @ B)) / d) ** 2


def grad(A, B, d):
    return (np.trace(A @ dag(B)) * (B @ dag(A)) - np.trace(B @ dag(A)) * (A @ dag(B))) / d ** 2

# Initialization

n = 6
N = 2 ** n
V = unitary_group.rvs(N)

# Runing Gradient Descent

dt = 1.0
U = np.eye(N)
costf = []
n_iter = 1000
for i in range(n_iter):
    U = expm(dt * grad(U, V, N)) @ U
    costf.append(cost(U, V, N))

# Plot Results
plt.plot(range(n_iter), costf)
plt.title('Gradient Descent for Quantum Compiling')
plt.xlabel('Interation')
plt.ylabel('Cost ${\cal L}(U)$')
```
#### Code for "Solving linear Equations"

here we tested with a linear system with size 8.

```python
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import unitary_group
from scipy.linalg import expm

# Useful Functions

def commutator(A, B):
  return A @ B - B @ A


def dag(X):
  return X.conj().T
  

def ketbra(b):
  x = b.reshape(-1, 1)
  return x @ dag(x)

# Cost and Gradient Functions

def cost(A, U, b):
  o = np.zeros(N)
  o[0] = 1.0
  psi = A @ U @ o
  return 1 - (np.abs(dag(b) @ psi) ** 2) / np.abs((dag(psi) @ psi))


def grad(A, U, b):
  ro = np.zeros((N, N))
  ro[0, 0] = 1.0

  f = np.trace(U @ ro @ dag(U) @ dag(A) @ ketbra(b) @ A)
  g = np.trace(U @ ro @ dag(U) @ dag(A) @ A)

  return commutator(
      U @ ro @ dag(U),
      (f * dag(A) @ A - g * dag(A) @ ketbra(b) @ A) / g**2,
  )

# Initialization

n = 3
N = 2 ** n

V = unitary_group.rvs(N)

o = np.zeros(N)
o[0] = 1.0
ro = np.zeros((N, N))
ro[0, 0] = 1.0

b = V @ o

A = np.array([[1, 0, 1, 0, 0, 0, 0, 0],[0, 1, 0, 0, 0, 0, 0, 0],[0, 0, 1, 1, 0, 0, 0, 0],[0, 1, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 1, 0],[0, 0, 0, 0, 0, 1, 0, 0],[0, 0, 0, 0, 0, 0, 1, 1],[0, 0, 0, 0, 0, 1, 0, 1]])

# Runing Gradient Descent

dt = 0.1
U = np.eye(N)
costf = []
n_iter = 1000
for i in range(n_iter):
    U = expm(dt * grad(A, U, b)) @ U
    costf.append(cost(A, U, b))

# Plot Results
plt.plot(range(n_iter), costf)
plt.title('Gradient Descent for Solving Linear Equations')
plt.xlabel('Interation')
plt.ylabel('Cost ${\cal L}(U)$')

```

#### Code for "Mean Square Error"

For the mean square error we randomly choose points around 2 and -2 with a gaussian.

```python
import numpy as np
import matplotlib.pyplot as plt
from random import gauss

from scipy.stats import unitary_group
from scipy.linalg import expm

# Useful Functions

def Ry(theta):
  h_t = theta / 2 # half theta
  return np.array([[np.cos(h_t), - np.sin(h_t)], [np.sin(h_t), np.cos(h_t)]])

def dag(X):
  return X.conj().T

def rho(x_i):
  x_range = 5 # ie -x_range <= x <= x_range
  theta = (x_i + x_range) * np.pi / (2*x_range)
  return Ry(theta) @ np.array([[1, 0],[0, 0]]) @ dag(Ry(theta))

def commutator(A, B):
  return A @ B - B @ A


# Cost and Gradient Functions

def cost(O, U, x, y):
  m = len(x)
  return sum([(np.trace(O @ U @ rho(x[i]) @ dag(U)) - y[i]) ** 2 for i in range(m)]) / (2*m)

def grad(O, U, x, y):
  m = len(x)

  def tr(i):
    return np.trace(O @ U @ rho(x[i]) @ dag(U)) - y[i]

  def commut(i):
    return commutator(U @ rho(x[i]) @ dag(U), O)

  return sum([ tr(i) * commut(i) for i in range(m)])/ m

# Initialization

N = 20
sig = .5
x = [gauss(-2, sig) for _ in range(N)] + [gauss(2, sig) for _ in range(N)]
y = [1 if x_i > 0 else -1 for x_i in x]

# Runing Gradient Descent

dt = 0.01
U = np.eye(2)
O = np.array([[1, 0],[0, -1]])
costf = []
n_iter = 1000
for i in range(n_iter):
    U = expm(dt * grad(O, U, x, y)) @ U
    costf.append(cost(O, U, x, y))

# Plot Results
plt.plot(range(n_iter), costf)
plt.title('Gradient Descent for Mean Square Error')
plt.xlabel('Interation')
plt.ylabel('Cost ${\cal L}(U)$')
```

### Results

The codes above produced the following graphs, in all cases the cost function decreases.

![Quantum Compiling](https://raw.githubusercontent.com/JoaoMiguelNC/JoaoMiguelNC.github.io/master/Images/Quantum%20Compiling.png)

![Solving Linear Equations](https://raw.githubusercontent.com/JoaoMiguelNC/JoaoMiguelNC.github.io/master/Images/Solving%20Linear%20Equations.png)

![Mean Square Error](https://raw.githubusercontent.com/JoaoMiguelNC/JoaoMiguelNC.github.io/master/Images/Mean%20Square%20Error.png)

# References

[1] Roeland Wiersema and Nathan Killoran. Optimizing quantum circuits with riemannian gradient flow, 2022.

[2] Sumeet Khatri, Ryan LaRose, Alexander Poremba, Lukasz Cincio, Andrew T. Sornborger, and Patrick J. Coles. Quantum-assisted quantum compiling. 2018.

[3] Carlos Bravo-Prieto, Ryan LaRose, M. Cerezo, Yigit Subasi, Lukasz Cincio, and Patrick J. Coles. Variational quantum linear solver, 2019.
