+++
date = "2022-06-25T23:07:28+01:00"
description = "Pattern Recognition (Bishop): Chapter 3"
draft = false
tags = ["Bishop", "Pattern Recognition", "Linear Models", "Bayesian Linear Regression", "Parameter Distribution"]
title = "Linear Models for Regression - Bayesian Linear Regression"
topics = ["Pattern Recognition"]

+++

## 3.3 Bayesian Linear Regression

One of the problems with frequentist approach and using maximum likelihood estimator is the issue of deciding the appropriate model complexity for the particular problem, which cannot be decided simply by maximizing the likelihood function, because this always leads to excessively complex models and over-fitting. We therefore turn to a Bayesian treatment of linear regression, which will avoid the over-fitting problem of maximum likelihood, and which will also lead to automatic methods of determining model complexity using the training data alone.

### 3.3.1 Parameter Distribution

The likelihood function $p(t|W)$ is the exponential of a quadratic function of $W$. The corresponding conjugate prior can be defined as

$$\begin{align}
p(W) = N(W|m_0,S_0)
\end{align}$$

having mean $m_0$ and covariance $S_0$. The posterior distribution can be then given as

$$\begin{align}
p(W|t) = N(W|m_N,S_N)
\end{align}$$

where

$$\begin{align}
m_N = S_N(S_0^{-1}m_0+\beta\phi^Tt)
\end{align}$$

$$\begin{align}
S_N^{-1} = S_0^{-1} + \beta\phi^T\phi
\end{align}$$

As the posterior distribution is Gaussian, its mode coincides with the mean and hence the <b>maximum posterior weight</b> is given as $W_{MAP} = m_N$. For an infinitely broad prior, the mean $m_N$ of the posterior distribution reduces to the maximum likelihood estimate $W_{ML}$. For sequential data points, the posterior distribution at any stage acts as the prior distribution for the subsequent data point.

For further analysis, we will use a zero-mean isotropic Gaussian governed by a single precision parameter $\alpha$ given as

$$\begin{align}
p(W|\alpha) = N(W|0,\alpha^{-1}I)
\end{align}$$

and the corresponding parameters for the posterior distribution are

$$\begin{align}
m_N = \beta S_N\phi^Tt
\end{align}$$

$$\begin{align}
S_N^{-1} = \alpha I + \beta\phi^T\phi
\end{align}$$

The log of the posterior distribution is the sum of the log likelihood and the log of the prior and it takes the form

$$\begin{align}
\ln p(W|t) = -\frac{\beta}{2}\sum_{n=1}^{N}(t_n - W^T\phi(X_n))^2 - \frac{\alpha}{2}W^TW + const
\end{align}$$

Hence, the maximization of the posterior distribution with respect to $W$ is equaivalent to the minimization of the sum-of-squares error function with the addition of a quadratic regularization term (in a frequentist setting the regularization parameter is $\lambda = \alpha/\beta$).