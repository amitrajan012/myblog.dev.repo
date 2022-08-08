+++
date = "2022-07-25T23:07:28+01:00"
description = "Pattern Recognition (Bishop): Chapter 6"
draft = false
tags = ["Bishop", "Pattern Recognition", "Linear Models", "Dual Representations", "Kernel Function", "Design Matrix"]
title = "Kernel Methods - Dual Representations"
topics = ["Pattern Recognition"]

+++


There is a calss of pattern recognition techniques, where the training data points or a subset of them are kept and used during the prediction phase. For example, in a simple technique for classification called as nearest neighbours, each new test sample is assigned the same label as the closest example form the training set. These type of prediction algorithms typically require a metric to be defined that <b>measures the similarity of any two vectors in input space</b>, and are generally <b>fast to train</b> but <b>slow at making predictions</b> for test data points. 

Many linear parametric models can be interpreted as a <b>dual representation</b> in which the predictions are also based on linear combinations of a <b>kernel function</b> evaluated at the training data points. For models which are based on a <b>fixed nonlinear feature space</b> $\phi(X)$ mapping the kernel function is given by the relation

$$\begin{align}
k(X,X^{'}) = \phi(X)^T\phi(X^{'})
\end{align}$$

We can see that the kernel is a symmetric function of its arguments as $k(X,X^{'}) = k(X^{'},X)$.

The simplest example of a kernel function is obtained by considering the <b>identity mapping</b> for the feature space in so that $\phi(X) = X$ in which case $k(X,X^{'}) = X^TX^{'}$. We shall refer to this as the <b>linear kernel</b>. 

The concept of a kernel formulated as an <b>inner product</b> in a feature space allows us to build interesting extensions of many well-known algorithms by making use of the <b>kernel trick</b>, also known as <b>kernel substitution</b>. The general idea is, if we have an algorithm formulated in such a way that the input vector $X$ enters only in the form of scalar products, then we can replace that scalar product with some other choice of kernel.

Some kernels have the property of being a function of the difference between the arguments, such that
$k(X,X^{'}) = k(X - X^{'})$. These kernels are known as <b>stationary kernels because they are invariant to translations in input space</b>. A further specialization involves <b>homogeneous kernels</b>, also known as <b>radial basis functions</b>, which depend only on the magnitude of the distance (typically Euclidean) between the arguments so that $k(X,X^{'}) = k(||X - X^{'}||)$.

## 6.1 Dual Representations

For a linear regression model whose parameters are determined by minimizing a <b>regularized sum-of-squares erro function</b> given by

$$\begin{align}
J(W) = \frac{1}{2} \sum_{n=1}^{N} \bigg( W^T\phi(X_n) -t_n \bigg)^2 + \frac{\lambda}{2}W^TW
\end{align}$$

where $\lambda \geq 0$. Minimizing with respect to $W$, we get

$$\begin{align}
W = -\frac{1}{\lambda} \sum_{n=1}^{N} \bigg( W^T\phi(X_n) -t_n \bigg)\phi(X_n) = \sum_{n=1}^{N} a_n\phi(X_n)
\end{align}$$

where

$$\begin{align}
a_n = -\frac{1}{\lambda} \bigg( W^T\phi(X_n) -t_n \bigg)
\end{align}$$

In matrix notation, the solution can be written as

$$\begin{align}
W = \phi(X_1)a_1 + \phi(X_2)a_2 + ... + \phi(X_N)a_N = \Phi^Ta
\end{align}$$

where $a = (a_1,a_2,...,a_N)^T$ and $\Phi = (\phi(X_1)^T, \phi(X_2)^T, ..., \phi(X_N)^T)^T$. $\Phi$ is called as the <b>design matrix</b>. Replacing $W = \Phi^Ta$ in the error function, we get

$$\begin{align}
J(a) = \frac{1}{2} a^T\Phi(\Phi^T\Phi)\Phi^Ta - a^T\Phi \Phi^Tt + \frac{1}{2}t^Tt + \frac{\lambda}{2}a^T\Phi\Phi^Ta
\end{align}$$

where $t = (t_1,t_2,...,t_N)^T$. Let $K = \Phi\Phi^T$, which is called as <b>Gram Matrix</b>. $K$ is a $N \times N$ <b>symmetric matrix</b>, with elements

$$\begin{align}
K_{nm} = \phi(X_n)^T\phi(X_m) = k(X_n,X_m)
\end{align}$$

where $k(X,X^{'})$ is a <b>kernel function</b> defined as $k(X,X^{'}) = \phi(X)^T\phi(X^{'})$. In terms of the Gram matrix, the sum-of-square error function is written as

$$\begin{align}
J(a) = \frac{1}{2} a^TKKa - a^TKt + \frac{1}{2}t^Tt + \frac{\lambda}{2}a^TKa
\end{align}$$

Setting the gradient of $J(a)$ with respect to $a$ to zero, we have

$$\begin{align}
a = (K + \lambda I_{N})^{-1}t
\end{align}$$

Substituting it back to the linear regression model, we get

$$\begin{align}
y(X) = W^T\phi(X) = a^T\Phi\phi(X) =
a^T\begin{bmatrix}
\phi(X_1)^T \\\\
\phi(X_2)^T \\\\
... \\\\
\phi(X_N)^T
\end{bmatrix}\phi(X) = 
a^T\begin{bmatrix}
\phi(X_1)^T \phi(X)\\\\
\phi(X_2)^T \phi(X)\\\\
... \\\\
\phi(X_N)^T \phi(X)
\end{bmatrix}
\end{align}$$

$$\begin{align}
= a^T\begin{bmatrix}
k(X_1,X)\\\\
k(X_2,X)\\\\
... \\\\
k(X_N,X)
\end{bmatrix} = \bf{k}(X)^Ta =
\bf{k}(X)^T(K + \lambda I_{N})^{-1}t
\end{align}$$

where vector $\bf{k}(X)$ has the elements $k(X_n,X)$. Hence, the dual formulation allows the solution to the least-squares problem to be expressed entirely in terms of the kernel function $k(X,X^{'})$. Note that the prediction at $X$ is given by a linear combination of the target values from the training set.

In the dual formulation, we determine the parameter vector a by inverting an $N \times N$ matrix, whereas in the original parameter space formulation we had to invert an $M \times M$ matrix in order to determine $W$. Because $N$ is typically much larger than $M$, the dual formulation does not seem to be particularly useful. However, the advantage of the dual formulation, as we shall see, is that it is expressed entirely in terms of the kernel function $k(X,X^{'})$. We can therefore work directly in terms of kernels and avoid the explicit introduction of the feature vector $\phi(X)$ which allows us implicitly to use feature spaces of high, even infinite, dimensionality.