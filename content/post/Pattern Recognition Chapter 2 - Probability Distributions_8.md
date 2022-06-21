+++
date = "2022-06-18T19:07:28+01:00"
description = "Pattern Recognition (Bishop): Chapter 2"
draft = false
tags = ["Bishop", "Pattern Recognition", "Probability Distributions", "Exponential Family", "Natural Parameter", "Bernoulli Distribution", "Logistic Sigmoid Function", "Multinomial Distribution", "Softmax Function", "Noninformative Priors"]
title = "Probability Distributions - The Exponential Family"
topics = ["Pattern Recognition"]

+++

## 2.4 The Exponential Family

<b>Exponential family</b> of distributions over $X$ with parameters $\eta$ is defined as

$$\begin{align}
p(X|\eta) = h(X)g(\eta)exp[\eta^Tu(X)]
\end{align}$$

Here $\eta$ is the <b>natural parameter</b> of the distribution and $u(X)$ is some function of $X$. $g(\eta)$ ensures that the distribution is normalized and satisfies

$$\begin{align}
g(\eta) \int h(X)exp[\eta^Tu(X)] dX = 1
\end{align}$$

<b>Bernoulli distribution</b> is a common exponential distribution. It is given as

$$\begin{align}
p(x|\mu) = Bern(x|\mu) = \mu^x(1-\mu)^{1-x} = exp[x\ln \mu + (1-x)\ln(1-\mu)]
\end{align}$$

$$\begin{align}
= exp\bigg[x\ln \bigg(\frac{\mu}{1-\mu}\bigg) + \ln(1-\mu)\bigg] = (1-\mu) exp\bigg[x\ln \bigg(\frac{\mu}{1-\mu}\bigg)\bigg]
\end{align}$$

Comparing this with the expression for the exponetial family of distribution, we get

$$\begin{align}
\eta = \ln \bigg(\frac{\mu}{1-\mu}\bigg)
\end{align}$$

Or, we can say that

$$\begin{align}
\mu = \sigma(\eta) = \frac{1}{1+exp(-\eta)}
\end{align}$$

which is called as the <b>logistic sigmoid function</b>. Hence,

$$\begin{align}
1-\mu = 1-\sigma(\eta) = 1-\frac{1}{1+exp(-\eta)} = \frac{exp(-\eta)}{1+exp(-\eta)} = \frac{1}{1+exp(\eta)} = \sigma(-\eta)
\end{align}$$

Finally, the Bernoulli distribution can be represented as

$$\begin{align}
p(x|\eta) = \sigma(-\eta)exp(\eta x)
\end{align}$$

Comparing it with the general expression on the exponential distribution, we get $u(x) = x$, $h(x) = 1$ and $g(\eta) = \sigma(-\eta)$.

Another example of exponential distribution is <b>multinomial distribution</b>. For a single observation $X$, it takes the form

$$\begin{align}
p(X|\mu) = \prod_{k=1}^{M}\mu_k^{x_k} = exp\bigg[ \sum_{k=1}^{M} x_k \ln \mu_k\bigg] = exp(\eta^TX)
\end{align}$$

where $\eta = (\eta_1, \eta_2, ..., \eta_M)^T$ and $\eta_k = \ln\mu_k$. Comparing with the general equation of exponential family, we get $u(x) = X$, $h(x) = 1$ and $g(\eta) = 1$. It should be noted that the parameters $\eta_k$ are not indepndet as $\sum_k \mu_k = 1$. Hence, $\mu_M$ can be derived from rest of the values as:

$$\begin{align}
\mu_M = 1 - \sum_{k=1}^{M-1}\mu_k
\end{align}$$

Using this, the multinomial distribution can be rewritten as

$$\begin{align}
p(X|\mu) = exp\bigg[ \sum_{k=1}^{M} x_k \ln \mu_k\bigg] = exp\bigg[ \sum_{k=1}^{M-1} x_k \ln \mu_k + x_M \ln \mu_M\bigg]
\end{align}$$

$$\begin{align}
= exp\bigg[ \sum_{k=1}^{M-1} x_k \ln \mu_k + \bigg(1 - \sum_{k=1}^{M-1}x_k\bigg) \ln\bigg(1 - \sum_{k=1}^{M-1}\mu_k\bigg)\bigg]
\end{align}$$

$$\begin{align}
= exp\bigg[ \sum_{k=1}^{M-1} x_k \ln\bigg(\frac{\mu_k}{1 - \sum_{j=1}^{M-1}\mu_j}\bigg) + \ln\bigg(1 - \sum_{k=1}^{M-1}\mu_k\bigg)\bigg]
\end{align}$$

$$\begin{align}
= \bigg(1 - \sum_{k=1}^{M-1}\mu_k\bigg)exp\bigg[ \sum_{k=1}^{M-1} x_k \ln\bigg(\frac{\mu_k}{1 - \sum_{j=1}^{M-1}\mu_j}\bigg)\bigg]
\end{align}$$

Comparing this with the general expression for exponential distribution, we get 

$$\begin{align}
\eta_k = \ln\bigg(\frac{\mu_k}{1 - \sum_{j=1}^{M-1}\mu_j}\bigg)
\end{align}$$

Solving for $\mu_k$, we get

$$\begin{align}
\mu_k = \frac{exp(\eta_k)}{1+\sum_j exp(\eta_j)}
\end{align}$$

which is called as <b>softmax function or normalized exponential</b>. In this form, the multinomial distribution can be represented as

$$\begin{align}
p(X|\eta) = \bigg(1 + \sum_{k=1}^{M-1}exp(\eta_k)\bigg)^{-1}exp(\eta^TX)
\end{align}$$

<b>Gaussian</b> can also be reduced in the form of exponetial distribution.

### 2.4.1 Maximum Likelihood and Sufficient Statistics

<b>Exponential family</b> of distributions over $X$ with parameters $\eta$ satisfies

$$\begin{align}
g(\eta) \int h(X)exp[\eta^Tu(X)] dX = 1
\end{align}$$

Taking derivative with respect to $\eta$, we have

$$\begin{align}
\nabla g(\eta) \int h(X)exp[\eta^Tu(X)] dX + g(\eta) \int h(X)exp[\eta^Tu(X)]u(X) dX= 0
\end{align}$$

$$\begin{align}
-\frac{1}{g(\eta)}\nabla g(\eta) = \frac{\int h(X)exp[\eta^Tu(X)]u(X) dX}{\int h(X)exp[\eta^Tu(X)]dX}
\end{align}$$

$$\begin{align}
-\frac{1}{g(\eta)}\nabla g(\eta) = g(\eta) \int h(X)exp[\eta^Tu(X)]u(X) dX = E[u(X)]
\end{align}$$

Hence,

$$\begin{align}
E[u(X)] = -\nabla \ln g(\eta)
\end{align}$$

The maximum likelihood estimator of $\eta$, denoted as $\eta_{ML}$ will satisfy

$$\begin{align}
-\nabla \ln g(\eta_{ML}) = \frac{1}{N}\sum_{n=1}^{N}u(X_n)
\end{align}$$

The solution of the maximum likelihood estimator depends on the data through $\sum_{n=1}u(X_n)$ only, which is therefore called as <b>sufficient statistic</b> of the distribution. It should be noted that <b>we don't need to store the entire dataset itself but just the value of sufficient statistc</b>.

### 2.4.2 Noninformative Priors

In many cases, we will have little or no idea how the prior distribution should look like. We may then seek a form of prior distribution, called a <b>noninformative prior</b>, which is intended to have as little influence on the posterior distribution as possible. 

For a distribution $p(x|\lambda)$, we might be tempted to propose a prior distribution $p(\lambda) = const$. For a discrete $\lambda$ with $K$ states, we can set the prior probability of each state as $1/K$. For continuous $\lambda$, if the domain of $\lambda$ is <b>unbounded</b>, the prior distribution cannot be correctly normalized as the integral over $\lambda$ diverges. Such priors are called <b>improper</b>. Another prolem is with the non-linear transformation of the variable. Let's say we want change the variable $\lambda$ as $\lambda = \eta^2$. The new density is then give as

$$\begin{align}
p_{\eta}(\eta) = p_{\lambda}(\lambda)\bigg|\frac{d\lambda}{d\eta}\bigg| = p_{\lambda}(\eta^2)2\eta
\end{align}$$

And hence the new density will not be constant.

Let us say that a density with the parameter $\mu$, called as the <b>location parameter</b>, takes a form

$$\begin{align}
p(x|\mu) = f(x-\mu)
\end{align}$$

This family of density shows <b>translational invariance</b> as for a shifted $x$, $\hat{x} = x+c$, the new density is

$$\begin{align}
p(\hat{x}|\hat{\mu}) = f(\hat{x}-\hat{\mu})
\end{align}$$

where $\hat{\mu} = \mu+c$. This means that the density is independent of choice of origin. The density which satisfies the translational invariance property assign equal probability mass to the interval $A \leq \mu \leq B$ and the shifted interval $A-c \leq \mu \leq B-c$, i.e.

$$\begin{align}
\int_{A}^{B}p(\mu)d\mu = \int_{A-c}^{B-c}p(\mu)d\mu = \int_{A}^{B}p(\mu-c)d\mu
\end{align}$$

This should hold for all choices of $A,B$ and hence $p(\mu-c) = p(\mu)$ which implies that $p(\mu)$ is constant.

Let us consider another form of density which is defined as

$$\begin{align}
p(x|\sigma) = \frac{1}{\sigma}f\bigg(\frac{x}{\sigma}\bigg)
\end{align}$$

where $\sigma > 0$. The parameter $\sigma$ is known as <b>scale parameter</b> and the density exhibits <b>scale invariance</b> as when $x$ is scaled by a factor such that $\hat{x} = cx$, then

$$\begin{align}
p(\hat{x}|\hat{\sigma}) = \frac{1}{\hat{\sigma}}f\bigg(\frac{\hat{x}}{\hat{\sigma}}\bigg)
\end{align}$$

where $\hat{\sigma} = c\sigma$. For scale invariance, the probability for the inervals $A \leq \sigma \leq B$ and $A/c \leq \sigma \leq B/c$ should be same, i.e.

$$\begin{align}
\int_{A}^{B}p(\sigma)d\sigma = \int_{A/c}^{B/c}p(\sigma)d\sigma = \int_{A}^{B}p\bigg( \frac{\sigma}{c}\bigg)\frac{1}{c}d\sigma
\end{align}$$

For this to hold

$$\begin{align}
p(\sigma) = p\bigg( \frac{\sigma}{c}\bigg)\frac{1}{c}
\end{align}$$