+++
date = "2022-04-22T14:07:28+01:00"
description = "Linear Algebra (Gilbert Strang): Chapter 21"
draft = false
tags = ["Linear Algebra", "Determinant", "Gilbert Strang", "Eigenvalues", "Eigenvectors", "Eigenvalue Decomposition", "Markov Matrices", "Fourier Series"]
title = "Markov Matrices and Fourier Series"
topics = ["Linear Algebra"]

+++

## 21.1 Markov Matrices

</b>Markov Matrices</b> have the following properties:
* All entries $\geq 0$
* Sum of the entries in a column equal $1$

<b>A markov matrix will always have an eigenvalue of $1$</b>. Apart from this, <b>all other eigenvalues will be $\leq 1$</b>. 

Let us consider the difference equation $u_k = A^ku_0$ where we represent $u_0$ as the combinations of eigenvectors, i.e. $u_0 = c_1x_1 + c_2x_2 + ... + c_nx_n = Sc$. Then $Au_0 = c_1Ax_1 + c_2Ax_2 + ... + c_nAx_n = c_1\lambda_1x_1 + c_2\lambda_2x_2 + ... + c_n\lambda_nx_n = \Lambda Sc$. Similarly, $A^ku_0 = c_1\lambda_1^kx_1 + c_2\lambda_2^kx_2 + ... + c_n\lambda_n^kx_n = \Lambda^kSc$.

When $A$ is a markov matrix, the steady state value for $u_k$ will be $c_1x_1$ as one the eigenvalues ($\lambda_1 = 1$) is $1$ and all the other eigenvalues will be $|\lambda_i < 1|$ when $i \neq 1$.

## 21.2 Why $\lambda = 1$ is an eigenvalue of a Markov Matrix

Let us take an example of a markov matrix $A$.

$$\begin{align}
A = \begin{bmatrix}
0.1 & 0.01 & 0.3 \\\\
0.2 & 0.99 & 0.3 \\\\
0.7 & 0 & 0.4
\end{bmatrix}
\end{align}$$

If $\lambda = 1$ is an eigenvalue, then the matrix $A-\lambda I = A-I$ will be a <b>singular matrix</b> as $|A-I| = 0$. <b>For a singualr matrix, at least one row/column is dependent</b>.

$$\begin{align}
A-I = \begin{bmatrix}
-0.9 & 0.01 & 0.3 \\\\
0.2 & -0.01 & 0.3 \\\\
0.7 & 0 & -0.6
\end{bmatrix}
\end{align}$$

As all columns of $A-I$ add to $0$, the rows are <b>linearlly dependent</b> as $row_1 + row_2 + row_3 = 0$ and hence the matrix $A-I$ is singular. This means that our assumption that $\lambda = 1$ is an eigenvalue is true. The eigenvector corresponding to the eigenvalue $\lambda = 1$ will be in the <b>null space of $A - \lambda I = A -I$</b>.

$$\begin{align}
(A-I)x = \begin{bmatrix}
-0.9 & 0.01 & 0.3 \\\\
0.2 & -0.01 & 0.3 \\\\
0.7 & 0 & -0.6
\end{bmatrix}\begin{bmatrix}
x_1 \\\\
x_2 \\\\
x_3 
\end{bmatrix} = 0 \implies
x = \begin{bmatrix}
0.6 \\\\
33 \\\\
0.7 
\end{bmatrix}
\end{align}$$

## 21.3 Applications of Markov Matrices

Markov Matrices can be viewed as the distribution of probabilities when certain thing/quantity is spread across multiple bins along a closed systme. For example, consider a closed system considering of two states: California and Washington. We are trying to build a systme to track the population of these two states such that only people from california and washington moves to each other. Let $\begin{bmatrix}
u_{c} \\\\
u_w
\end{bmatrix}(k+1)$ be the population of these states at $t=k+1$. The population movement is represented as:

$$\begin{align}
\begin{bmatrix}
u_{c} \\\\
u_w
\end{bmatrix}(k+1) = M\begin{bmatrix}
u_{c} \\\\
u_w
\end{bmatrix}(k) = \begin{bmatrix}
0.9 & 0.2 \\\\
0.2 & 0.8
\end{bmatrix}\begin{bmatrix}
u_{c} \\\\
u_w
\end{bmatrix}(k)
\end{align}$$

where $M$ is a Markov Matrix. This system of equation can be interpreted as follows:

$$\begin{align}
\begin{bmatrix}
u_c \\\\
u_w
\end{bmatrix}(k+1) = \begin{bmatrix}
0.9u_c+0.2u_w \\\\
0.2u_c+0.8u_w
\end{bmatrix}(k)
\end{align}$$

This means that at every time step, $90\\%$ of california's population stays while $20\\%$ of washington's population moves to california. Similarly, at every time step $80\\%$ of washington's population stays while $10\\%$ of california's population moves to washington. This system of equation can be solved by solving the difference equation $u_{k} = M^k u_0$.

The eigenvalues of $M$ are $\lambda_1 = 1$ (as it's a markov matrix) and $\lambda_2 = 0.7$ (as trace = $1.7=\lambda_1 + \lambda_2$). The corresponding eigenvectors are $x_1 = \begin{bmatrix}
2 \\\\
1
\end{bmatrix}$ and $x_2 = \begin{bmatrix}
-1 \\\\
1
\end{bmatrix}$. This leads us to the generic solution:

$$\begin{align}
u_k = c_1 (1)^k \begin{bmatrix}
2 \\\\
1
\end{bmatrix} + c_2 (0.7)^k \begin{bmatrix}
-1 \\\\
1
\end{bmatrix}
\end{align}$$

Assuming the initial state of $u_0 = \begin{bmatrix}
0 \\\\
1000
\end{bmatrix}$, we get the constants $c_1$ and $c_2$ as $c_1 = \frac{1000}{3};c_2 = \frac{2000}{3}$.

## 21.4 Projections with Orthonormal Basis

Let $q_1, q_2, ..., q_n$ be the $n$ orthonormal basis vectors in $n-dimensional$ space. Then any vector $v$ can be represented as:

$$\begin{align}
v = x_1q_1 + x_2q_2 + ... + x_nq_n
\end{align}$$

It should be noted that $q_1^Tv = x_1q_1^Tq_1 = x_1$, as $q_1^Tq_1 = 1$ and $q_1^Tq_i = 0$ when $i \neq 1$ as $q_is $ are orthonormal to each other. In the matrix form, $Qx = v$ where $q_is$ are the columns of $Q$. This gives us $x = Q^{-1}v$. As the columns of $Q$ are orthonormal, $Q^{-1} = Q^T$, i.e. $x = Q^Tv$. 



## 21.5 Fourier Series

The following function $f$ is called a <b>Fourier Series</b>.

$$\begin{align}
f(x) = a_0 + a_1 \cos x + b_1 \sin x + a_2 \cos 2x + b_2 \sin 2x + ...
\end{align}$$

One of the thing to note is that this series goes till $\infty$. We can claim that $1, \cos x, \sin x, \cos 2x, \sin 2x,...$ are the <b>basis</b> and they are <b>othogonal</b>. To check the claim of the orthogonality, we need to find  a way to find the inner products of functions. Till now vectors were in discrete space. These function values are in continuous space and hence, first of all, instead of addition we will have integration in the inner product. One more thing to note is that these functions are <b>periodic</b> with a period $2\pi$. Hence, the integral only needs to be computed fron $0 \to 2\pi$. Hence, for any pair of function $f$ and $g$ in the basis, their inner product can be defined as

$$\begin{align}
f^Tg = \int_{0}^{2\pi}f(x)g(x)dx
\end{align}$$

Let's consider two function $f(x) = \sin x$ and $g(x) = \cos x$, then

$$\begin{align}
f^Tg = \int_{0}^{2\pi}(\sin x \cos x) dx = 0
\end{align}$$

which means that these two basis vectors are orthogonal. The coefficients can be found the similar way we did for the projections with orthonormal basis. For example, taking the inner product with $\cos x$ on both hand side, we get the value of $a_1$ as

$$\begin{align}
\int_{0}^{2\pi}(f(x) \cos x) dx = a_1 \int_{0}^{2\pi}(\cos x)^2 dx = a_1 \pi
\end{align}$$
