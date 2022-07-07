+++
date = "2022-07-03T23:07:28+01:00"
description = "Pattern Recognition (Bishop): Chapter 4"
draft = false
tags = ["Bishop", "Pattern Recognition", "Linear Models", "Classification", "Probabilistic Generative Models", "Class Priors", "Class-conditional Densities", "Logistic Sigmoid Function"]
title = "Linear Models for Clasification - Probabilistic Generative Models"
topics = ["Pattern Recognition"]

+++

## 4.2 Probabilistic Generative Models

In the <b>probabilistic generative modelling</b> approach, we model the <b>class-conditional densities</b> $p(X|C_k)$, as well as the <b>class priors</b> $p(C_k)$, and then use these to compute <b>posterior probabilities</b> $p(C_k|X)$ through Bayes’ theorem. The posterior probability for class $C_1$ is

$$\begin{align}
p(C_1|X) = \frac{p(X|C_1)p(C_1)}{p(X)} = \frac{p(X|C_1)p(C_1)}{p(X|C_1)p(C_1) + p(X|C_2)p(C_2)}
\end{align}$$

$$\begin{align}
= \frac{1}{1+exp(-a)} = \sigma(a)
\end{align}$$

where

$$\begin{align}
a = \ln \frac{p(X|C_1)p(C_1)}{p(X|C_2)p(C_2)}
\end{align}$$

and $\sigma(a)$ is called as the <b>logistic sigmoid function</b>. It is a <b>S-shaped</b> function and also called as <b>squashing function</b> as it maps the entire real axis into a finite interval. It satisfies: $\sigma(-a) = 1 - \sigma(a)$. The inverse of the function is given as

$$\begin{align}
a = \ln \bigg(\frac{\sigma}{1 - \sigma}\bigg) = \ln \frac{p(C_1|X)}{p(C_2|X)}
\end{align}$$

and is called as <b>logit function</b> or <b>log odds</b>. For a $K>2$ class case, the posterior distribution is given as

$$\begin{align}
p(C_k|X) = \frac{p(X|C_k)p(C_k)}{\sum_{j}p(X|C_j)p(C_j)} = \frac{exp(a_k)}{\sum_{j}exp(a_j)}
\end{align}$$

and is called as <b>normalized exponential</b> or <b>softwax function</b>. The quantity $a_k$ is defined as

$$\begin{align}
a_k = \ln p(X|C_k)p(C_k)
\end{align}$$

### 4.2.1 Continuous Inputs

For the continuous input case, let us assume that the <b>class conditional densities</b> are Gaussian where all classes share the same covariance matrix. Then the density for class $C_k$ is given as

$$\begin{align}
p(X|C_k) = \frac{1}{(2\pi)^{D/2}} \frac{1}{|\Sigma|^{1/2}} exp \bigg[-\frac{1}{2}(X-\mu_k)^T \Sigma^{-1} (X - \mu_k)\bigg]
\end{align}$$

For a two class case, the posterior distribution is given as

$$\begin{align}
p(C_1|X) = \sigma(a) = \frac{1}{1+exp(-a)}
\end{align}$$

where 

$$\begin{align}
a = \ln \frac{p(X|C_1)p(C_1)}{p(X|C_2)p(C_2)}
\end{align}$$

Evaluating $a$, we have

$$\begin{align}
a = \ln \bigg(\frac{p(X|C_1)}{p(X|C_2)} \times \frac{p(C_1)}{p(C_2)}\bigg) = \ln \frac{p(X|C_1)}{p(X|C_2)} + \ln \frac{p(C_1)}{p(C_2)}
\end{align}$$

$$\begin{align}
= \ln \frac{exp \bigg[-\frac{1}{2}(X-\mu_1)^T \Sigma^{-1} (X - \mu_1)\bigg]}{exp \bigg[-\frac{1}{2}(X-\mu_2)^T \Sigma^{-1} (X - \mu_2)\bigg]} + \ln \frac{p(C_1)}{p(C_2)}
\end{align}$$

$$\begin{align}
= (\mu_1^T - \mu_2^T)\Sigma^{-1}X - \frac{1}{2}\mu_1^T\Sigma^{-1}\mu_1 + \frac{1}{2}\mu_2^T\Sigma^{-1}\mu_2 + \ln \frac{p(C_1)}{p(C_2)}
\end{align}$$

$$\begin{align}
= W^TX + W_0
\end{align}$$

Hence, the posterior distribution is given as

$$\begin{align}
p(C_1|X) = \sigma(W^TX + W_0)
\end{align}$$

where

$$\begin{align}
W = \Sigma^{-1}(\mu_1 - \mu_2)
\end{align}$$

$$\begin{align}
W_0 = - \frac{1}{2}\mu_1^T\Sigma^{-1}\mu_1 + \frac{1}{2}\mu_2^T\Sigma^{-1}\mu_2 + \ln \frac{p(C_1)}{p(C_2)}
\end{align}$$

The <b>prior probabilities</b> only affect the bias parameter $W_0$ and hence <b>change in priors have the effect of making parallel shifts of thd decision boundary</b>. For a $K>2$ class case,

$$\begin{align}
p(C_k|X) = \frac{exp(a_k)}{\sum_{j}exp(a_j)}
\end{align}$$

where 

$$\begin{align}
a_k(X) = W_k^TX + W_{k0}
\end{align}$$

$$\begin{align}
W_k = \Sigma^{-1}\mu_k
\end{align}$$

$$\begin{align}
W_{k0} = - \frac{1}{2}\mu_k^T\Sigma^{-1}\mu_k + \ln p(C_k)
\end{align}$$

If the shared covariance matrix assumption is not true, we will obtain quadratic functions of $X$, giving rise to <b>quadratic discriminant</b>.