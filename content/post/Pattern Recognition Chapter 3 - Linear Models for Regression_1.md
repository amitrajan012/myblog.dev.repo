+++
date = "2022-06-20T22:07:28+01:00"
description = "Pattern Recognition (Bishop): Chapter 3"
draft = false
tags = ["Bishop", "Pattern Recognition", "Linear Models", "Basis Function", "Gaussian Basis Function", "Maximum Likelihood", "Least Squares"]
title = "Linear Models for Regression - Linear Basis Function Models : Part 1"
topics = ["Pattern Recognition"]

+++

<b>Linear regression model</b> has the property of being linear functions of adjustable parameters. We can add more complexity in the linear regression models by taking linear combinations of a fixed set of nonlinear functions of the input variables, known as <b>basis functions</b>. In the modeling process, giveb $x$, we have to predict $t$ which can be predicted as $y(x)$. Form a probabilistic prespective, we aim to model the predictive distribution $p(t|x)$ as this expresses the uncertainty about the value of $t$ for each value of $x$. Linear models have significant limitations as practical techniques for pattern recognition, particularly for problems involving input spaces of high dimensionality.

## 3.1 Linear Basis Function Models

In the simplest linear regression model, output is the linear combination of input variables

$$\begin{align}
y(X,W) = w_0 + w_1x_1 + w_2x_2 + ... + w_Dx_D
\end{align}$$

Extending it to include the nonlinear combination of input variables, we get

$$\begin{align}
y(X,W) = w_0 + \sum_{j=1}^{M-1}w_j\phi_j(X)
\end{align}$$

where $\phi_j(X)$ is called as <b>basis function</b>. The total number of parameters in this model is $M$. $w_0$ is called as the <b>bias parameter</b>. If we define a dummy basis function for bias as $\phi_0(X)=1$, we have

$$\begin{align}
y(X,W) = \sum_{j=0}^{M-1}w_j\phi_j(X) = W^T\phi(X)
\end{align}$$

where $W = (w_0,w_1,...,w_{M-1})^T$ and $\phi = (\phi_0,\phi_1,...,\phi_{M-1})^T$. The basis function can take any form. For example, a polynomial basis function takes the form $\phi_{j}(x) = x^j$. We can also have piecewise polynomial function called as <b>spline functions</b> where we have different polynomials in different regions of input space. Another example is <b>Gaussian basis function</b>, which takes the form

$$\begin{align}
\phi_j(x) = exp\bigg[ -\frac{(x-\mu_j)^2}{2s^2}\bigg]
\end{align}$$

where $\mu_j$ govers the location and $s$ governs their spatial space. We can also have <b>sigmoidal basis function</b>, which is defined as

$$\begin{align}
\phi_j(x) = \frac{1}{1+exp[-\frac{x-\mu_j}{s}]}
\end{align}$$

### 3.1.1 Maximum Likelihood and Least Squares

The target variable $t$ is predicted as $y(X,W)$. The prediction $y(X,W)$ will have some additional noise in it and let us assume that the noise is Gaussian. Then,

$$\begin{align}
t = y(X,W) + \epsilon
\end{align}$$

where $\epsilon$ is a zero mean Gaussian with precision $\beta$. Hence,

$$\begin{align}
p(t|X,W,\beta) = N(t|y(X,W),\beta^{-1})
\end{align}$$

For a squared loss function, the optimal prediction, for a new value of $X$, will be given by the conditional mean of the target variable, i.e.

$$\begin{align}
E[t|X] = \int t p(t|X)dt = y(X,W)
\end{align}$$

Let us consider a dataset of inputs $X=\{X_1,X_2,...,X_n\}$ with the target variables $t=\{t_1,t_2,...,t_n\}$. Assuming that these data points are drawn independetly, the likelihood function is given as

$$\begin{align}
p(t|X,W,\beta) = \prod_{n=1}^{N} N(t_n|W^T\phi(X_n), \beta^{-1})
\end{align}$$

The log likelihood is given as

$$\begin{align}
\ln p(t|X,W,\beta) = \sum_{n=1}^{N} \ln N(t_n|W^T\phi(X_n), \beta^{-1})
\end{align}$$

$$\begin{align}
= \sum_{n=1}^{N} \ln \bigg[ \frac{1}{(2\pi\beta^{-1})^{1/2}} exp\bigg(\frac{-\beta}{2}(t_n - W^T\phi(X_n))^2\bigg)\bigg]
\end{align}$$

$$\begin{align}
= \frac{N}{2}\ln\beta - \frac{N}{2}\ln(2\pi) - \frac{\beta}{2}\sum_{n=1}^{N} (t_n - W^T\phi(X_n))^2
\end{align}$$

$$\begin{align}
= \frac{N}{2}\ln\beta - \frac{N}{2}\ln(2\pi) - \beta E_D(W)
\end{align}$$

where

$$\begin{align}
E_D(W) = \frac{1}{2}\sum_{n=1}^{N} (t_n - W^T\phi(X_n))^2
\end{align}$$

Taking derivative with respect to $W$ and setting this to $0$, we get

$$\begin{align}
\nabla\ln p(t|X,W,\beta) =  \sum_{n=1}^{N} [t_n - W^T\phi(X_n)]\phi(X_n)^T = 0
\end{align}$$

$$\begin{align}
\sum_{n=1}^{N} t_n\phi(X_n)^T - W^T\sum_{n=1}^{N}\phi(X_n)\phi(X_n)^T = 0
\end{align}$$

Converting it into matrix form and solving we get, 

$$\begin{align}
W_{ML} = (\phi^T\phi)^{-1}\phi^Tt
\end{align}$$

where $\phi$ is a $N \times M$ <b>design matrix</b> with $n^{th}$ row has the basis vector for $X_n$. The above equation is called as the <b>normal equation</b> for the least square problem. Taking derivative with respect to $\beta$ and equating it to $0$, we get

$$\begin{align}
\nabla\ln p(t|X,W,\beta) = \frac{N}{2\beta} - E_D(W) = 0
\end{align}$$

$$\begin{align}
\frac{1}{\beta_{ML}} = \frac{2}{N}E_D(W) = \frac{1}{N} \sum_{n=1}^{N} (t_n - W^T\phi(X_n))^2
\end{align}$$

The role of the bias parameter $W_0$ can be analyzed by setting the derivative of $\ln p(t|X,W,\beta)$ with respect to $W_0$ $0$. As $W_0$ is only in $E_D(W)$, making it explicit in $E_D(W)$, we get

$$\begin{align}
E_D(W) = \frac{1}{2}\sum_{n=1}^{N} \bigg(t_n - W_0 - \sum_{j=1}^{M-1} W_j\phi_j(X_n)\bigg)^2
\end{align}$$

$$\begin{align}
\nabla\ln p(t|X,W,\beta) = \nabla E_D(W) = \sum_{n=1}^{N} \bigg(t_n - W_0 - \sum_{j=1}^{M-1} W_j\phi_j(X_n)\bigg) = 0
\end{align}$$

$$\begin{align}
NW_0 = \sum_{n=1}^{N} \bigg(t_n - \sum_{j=1}^{M-1} W_j\phi_j(X_n)\bigg)
\end{align}$$

$$\begin{align}
W_0 = \frac{1}{N}\sum_{n=1}^{N}t_n  - \sum_{j=1}^{M-1}W_j\frac{1}{N}\sum_{n=1}^{N} \phi_j(X_n)
\end{align}$$

$$\begin{align}
W_0 = \bar{t}  - \sum_{j=1}^{M-1}W_j\bar{\phi_j}
\end{align}$$

where

$$\begin{align}
\bar{t}  = \frac{1}{N}\sum_{n=1}^{N}t_n
\end{align}$$

$$\begin{align}
\bar{\phi_j}  = \frac{1}{N}\sum_{n=1}^{N} \phi_j(X_n)
\end{align}$$

Hence the bias compensates for the difference in actual value (avearge over the training set) and the weighetd sum of of the avearage over the basis vector.