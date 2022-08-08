+++
date = "2022-07-15T23:07:28+01:00"
description = "Pattern Recognition (Bishop): Chapter 5"
draft = false
tags = ["Bishop", "Pattern Recognition", "Neural Networks", "Feed-forward Network", "Hessian Matrix", "Diagonal Approximation", "Outer Product Approximation"]
title = "Neural Networks - The Hessian Matrix"
topics = ["Pattern Recognition"]

+++

## 5.4 The Hessian Matrix

Backpropagation can also be used to evaluate the second derivatives of the error, given by

$$\begin{align}
H = \frac{\partial^2 E}{\partial W_{ji}\partial W_{lk}}
\end{align}$$

An important consideration for many applications of the Hessian is the efficienc with which it can be evaluated. If there are $W$ parameters (weights and biases) in the network, then the Hessian matrix has dimensions $W \times W$ and so the computational effort needed to evaluate the Hessian will scale like $O(W^2)$ for each pattern in the data set. As we shall see, there are efficient methods for evaluating the Hessian whose scaling is indeed $O(W^2)$.

### 5.4.1 Diagonal Approximation

Some of the applications for the Hessian matrix discussed above require the inverse of the Hessian, rather than the Hessian itself. For this reason, there has been some interest in using a diagonal approximation to the Hessian, in other words one that simply replaces the off-diagonal elements with zeros, because its inverse is trivial to evaluate. For a data set with $N$ data points, the Hessian can be computed by considering one pattern at a time and then summing the results over all the patterns.

The diagonal elements of the Hessian can be written as

$$\begin{align}
\frac{\partial^2 E_n}{\partial W_{ji}^2} = \frac{\partial^2 E_n}{\partial a_j^2}\frac{\partial^2 a_j}{\partial W_{ji}^2} = \frac{\partial^2 E_n}{\partial a_j^2}z_i^2
\end{align}$$

The value of $\frac{\partial^2 E_n}{\partial a_j^2}$ can be evaluated using chain rule of differentiation as

$$\begin{align}
\frac{\partial^2 E_n}{\partial a_j^2} = \frac{\partial}{\partial a_j}\bigg(\frac{\partial E_n}{\partial a_j}\bigg)
\end{align}$$

$$\begin{align}
= \frac{\partial}{\partial a_j}\bigg[\sum_k \frac{\partial E_n}{\partial a_k} \frac{\partial a_k}{\partial z_j} \frac{\partial z_j}{\partial a_j}\bigg] = \frac{\partial}{\partial a_j}\bigg[ h^{'}(a_j)\sum_k W_{kj} \frac{\partial E_n}{\partial a_k} \bigg]
\end{align}$$

$$\begin{align}
= h^{''}(a_j)\sum_k W_{kj} \frac{\partial E_n}{\partial a_k} + h^{'}(a_j)\sum_k \frac{\partial}{\partial a_j}\bigg[ W_{kj} \frac{\partial E_n}{\partial a_k} \bigg]
\end{align}$$

$$\begin{align}
\frac{\partial^2 E_n}{\partial a_j^2} = h^{''}(a_j)\sum_k W_{kj} \frac{\partial E_n}{\partial a_k} + h^{'}(a_j)^2 \sum_k \sum_{k^{'}} W_{kj}W_{k^{'}j} \frac{\partial^2 E_n}{\partial a_k\partial a_{k^{'}}}
\end{align}$$

Neglecting off-diagonal terms, we have

$$\begin{align}
\frac{\partial^2 E_n}{\partial a_j^2} = h^{''}(a_j)\sum_k W_{kj} \frac{\partial E_n}{\partial a_k} + h^{'}(a_j)^2 \sum_k W_{kj}^2 \frac{\partial^2 E_n}{\partial a_k^2}
\end{align}$$

One of the major problem with diagonal approximations, however, is that in practice the Hessian is typically found to be strongly nondiagonal, and so these approximations, which are driven mainly be computational convenience, must be treated with care.

### 5.4.2 Outer Product Approximation

For the case of neural network applied to a regression problem, error function is given as the sum-of-squares error as

$$\begin{align}
E = \frac{1}{2}\sum_{n=1}^{N}(y_n - t_n)^2
\end{align}$$

The Hessian matrix can be written as

$$\begin{align}
\nabla E = \sum_{n=1}^{N}(y_n - t_n)\nabla y_n
\end{align}$$

$$\begin{align}
H = \nabla\nabla E = \sum_{n=1}^{N}\nabla [(y_n - t_n)\nabla y_n]
\end{align}$$

$$\begin{align}
H = \nabla\nabla E = \sum_{n=1}^{N} \nabla y_n \nabla y_n + \sum_{n=1}^{N} (y_n - t_n) \nabla \nabla y_n
\end{align}$$

For a trained neural netwrok, $y_n \simeq t_n$ and hence the second term will be samll and can be ignored. By neglecting the second term, we arrive at the <b>Levenberg–Marquardt approximation or outer product approximation</b>. Hence,

$$\begin{align}
H \simeq \sum_{n=1}^{N} b_n  b_n^T
\end{align}$$

where $b_n = \nabla y_n = \nabla a_n$ for the <b>unit activation function</b> at the output layer. It should be noted that this approximation is only likely to be valid for a network that has been trained appropriately, and that for a general network mapping the second derivative terms on the right-hand side will typically not be negligible.

For the classification problem, we use the cross-entropy error function for a network with logistic sigmoid output-unit activation functions, the corresponding approximation can be derived as

$$\begin{align}
E = -\sum_{n=1}^{N} t_n \ln{y_n} + (1-t_n) \ln{(1-y_n)}
\end{align}$$

$$\begin{align}
\nabla E = -\sum_{n=1}^{N} \bigg[\frac{t_n}{y_n} \nabla y_n + \frac{1-t_n}{1-y_n}(-\nabla y_n)\bigg]
\end{align}$$

where

$$\begin{align}
\nabla y_n = \nabla h(a_n) = y_n(1-y_n)\nabla a_n
\end{align}$$

The expression for first derivative reduces to

$$\begin{align}
\nabla E = -\sum_{n=1}^{N} \bigg[\frac{t_n}{y_n} - \frac{1-t_n}{1-y_n}\bigg] \nabla y_n = -\sum_{n=1}^{N} \bigg[\frac{t_n-y_n}{y_n(1-y_n)}\bigg] \nabla y_n = -\sum_{n=1}^{N} (t_n-y_n)\nabla a_n
\end{align}$$

$$\begin{align}
\nabla\nabla E = -\sum_{n=1}^{N} -\nabla y_n \nabla a_n -\sum_{n=1}^{N} (t_n-y_n)\nabla\nabla a_n
\end{align}$$

Neglecting the second term and replacing the value of $\nabla y_n$, we get

$$\begin{align}
H = \nabla\nabla E \simeq \sum_{n=1}^{N} y_n(1-y_n) \nabla a_n\nabla a_n = \sum_{n=1}^{N} y_n(1-y_n) b_n b_n^T
\end{align}$$

### 5.4.3 Fast multiplication by the Hessian

For many applications of the Hessian, the quantity of interest is not the Hessian matrix $H$ itself but the product of $H$ with some vector $v$. The evaluation of the Hessian takes $O(W^2)$ operations, and it also requires storage that is $O(W^2)$. The vector $v^TH$ that we wish to calculate, however, has only $W$ elements, so instead of computing the Hessian as an intermediate step, we can instead try to find an efficient approach to evaluating $v^TH$ directly in a way that requires only $O(W)$ operations. The product can be written as

$$\begin{align}
v^TH = v^T\nabla (\nabla E)
\end{align}$$

This expression can be interpretted as the standard forward-propagation and backpropagation equations for the evaluation of $\nabla E$ and then use $v^T\nabla$ as the differential opertor to these equations to give a set of forward-propagation and backpropagation equations for the evaluation of $v^TH$. Let the differential operator $v^T\nabla$ is denoted as $R(.)$. As $\nabla$ is the differntial operator which finds derivative with respect to $W$, $R(W) = v$.

For a two layer netwrok, the forward-propagation equations are given as

$$\begin{align}
a_j = \sum_{i} W_{ji}X_i
\end{align}$$

$$\begin{align}
z_j = h(a_j)
\end{align}$$

$$\begin{align}
y_k = \sum_{j} W_{kj}z_j
\end{align}$$

Applying the operator $R(.)$ on these equations, we have

$$\begin{align}
R(a_j) = \sum_{i} v_{ji}X_i
\end{align}$$

$$\begin{align}
R(z_j) = h^{'}(a_j)R(a_j)
\end{align}$$

$$\begin{align}
R(y_k) = \sum_{j} W_{kj}R(z_j) + \sum_{j} v_{kj}z_j
\end{align}$$

These equations can be used to find the values of $R(a_j)$ and then $R(z_j)$ and eventually $R(y_k)$.

For the backpropogation part, assuming the sum-of-squares error function, the backpropagation equations can be written as

$$\begin{align}
\delta_k = y_k - t_k
\end{align}$$

$$\begin{align}
\delta_j = h^{'}(a_j)\sum_{k}W_{kj}\delta_k
\end{align}$$

Applying the operator $R(.)$ on these equations, we have (using differentiation by parts)

$$\begin{align}
R(\delta_k) = R(y_k)
\end{align}$$

$$\begin{align}
R(\delta_j) = h^{''}(a_j) R(a_j)\sum_{k}W_{kj}\delta_k + h^{'}(a_j)\sum_{k}v_{kj}\delta_k + h^{'}(a_j)\sum_{k}W_{kj}R(\delta_k)
\end{align}$$

The equations for the first derivative of error $\nabla E$ with respect to weights at both the layers are

$$\begin{align}
\nabla E = \frac{\partial E}{\partial W_{kj}} = (y_k - t_k) \frac{\partial y_k}{\partial W_{kj}} = (y_k - t_k) \frac{\partial a_k}{\partial W_{kj}} = (y_k - t_k) \frac{\partial W_{kj}z_j}{\partial W_{kj}} = \delta_k z_j
\end{align}$$

$$\begin{align}
\nabla E = \frac{\partial E}{\partial W_{ji}} = \delta_j X_i
\end{align}$$

Applying the operator $R(.)$ on these equations, we have

$$\begin{align}
R\bigg(\frac{\partial E}{\partial W_{kj}}\bigg) = R(\delta_k)z_j = \delta_k R(z_j)
\end{align}$$

$$\begin{align}
R\bigg(\frac{\partial E}{\partial W_{ji}}\bigg) = X_iR(\delta_j)
\end{align}$$

The implementation of this algorithm involves the introduction of additional variables $R(a_j), R(z_j), R(\delta_j)$ for the hidden units and $R(\delta_j), R(y_k)$ for the output units. For each input pattern, the values of these quantities can be found using the above results, and the elements of $v^TH$ are then given by the last two equations. An elegant aspect of this technique is that the equations for evaluating $v^TH$ mirror closely those for standard forward and backward propagation, and so the extension of existing software to compute this product is typically straightforward.

