+++
date = "2022-06-14T19:07:28+01:00"
description = "Pattern Recognition (Bishop): Chapter 2"
draft = false
tags = ["Bishop", "Pattern Recognition", "Probability Distributions", "Gaussian Distribution", "Conditional Gaussian Distribution", "Precision Matrix", "Marginal Gaussian Distribution"]
title = "Probability Distributions - The Gaussian Distribution: Part 2"
topics = ["Pattern Recognition"]

+++

### 2.3.1 Conditional Gaussian Distribution

If two sets of variables are jointly Gaussian, then the conditional distribution of one set conditioned on the othre is Gaussian. The marginal distribution of either set is also Gaussian. Let $X$ is a $D$-dimensional vector with Gaussian distribution $N(X|\mu,\Sigma)$. $X$ is partitioned into two disjoint subsets $X_a,X_b$. Without loss of generality we can assume thet $X_a$ forms the first $M$ components of $X$ and $X_b$ the remaining $D-M$, such that

$$\begin{align}
X = \begin{pmatrix}
X_a\\\\
X_b
\end{pmatrix};\mu = \begin{pmatrix}
\mu_a\\\\
\mu_b
\end{pmatrix};\Sigma = \begin{pmatrix}
\Sigma_{aa} & \Sigma_{ab}\\\\
\Sigma_{ba} & \Sigma_{bb}
\end{pmatrix}
\end{align}$$

As $\Sigma$ is symmetric, i.e. $\Sigma^T = \Sigma$. This implies that $\Sigma_{aa} = \Sigma_{bb}$ and $\Sigma_{ba} = \Sigma_{ab}^T$. It is eaasier to work with the inverse of the covarince matrix, which is called as <b>precision matrix</b> $\Lambda = \Sigma^{-1}$. The partitioned form of precision matrix is given as

$$\begin{align}
\Lambda = \begin{pmatrix}
\Lambda_{aa} & \Lambda_{ab}\\\\
\Lambda_{ba} & \Lambda_{bb}
\end{pmatrix}
\end{align}$$

where $\Lambda_{aa},\Lambda_{bb}$ are symmetric and $\Lambda_{ba} = \Lambda_{ab}^T$. As we know that the Gaussian ditribution is of the quadratic form with respect to the input $X$, it will be sufficient to show that the quadratic form of joint Gaussian when partitioned into $X_a,X_b$ and conditioned on $X_b$ ($X_a$ is variable with fixed $X_b$) takes the quadratic form for $X_a$. 

$$\begin{align}
\Delta^2 = -\frac{1}{2}(X-\mu)^T\Sigma^{-1}(X-\mu)
\end{align}$$

$$\begin{align}
= -\frac{1}{2}\begin{pmatrix}
X_a-\mu_a\\\\
X_b-\mu_b
\end{pmatrix}^T\begin{pmatrix}
\Lambda_{aa} & \Lambda_{ab}\\\\
\Lambda_{ba} & \Lambda_{bb}
\end{pmatrix}\begin{pmatrix}
X_a-\mu_a\\\\
X_b-\mu_b
\end{pmatrix}
\end{align}$$

$$\begin{align}
= -\frac{1}{2}(X_a-\mu_a)^T\Lambda_{aa}(X_a-\mu_a) -\frac{1}{2}(X_a-\mu_a)^T\Lambda_{ab}(X_b-\mu_b)
\end{align}$$
$$\begin{align}
-\frac{1}{2}(X_b-\mu_b)^T\Lambda_{ba}(X_a-\mu_a) -\frac{1}{2}(X_b-\mu_b)^T\Lambda_{bb}(X_b-\mu_b)
\end{align}$$

The above expresssion as a function of $X_a$ takes a quadratic form and hence the corresponding conditional distribution $p(X_a|X_b)$ is Gaussian.

One way to find the mean $\mu$ and the covariance/precision matirx $\Sigma$ is to compute the coefficients in the quadratic form of Gaussian. The quadractic form of Gaussian can be further decomposed as

$$\begin{align}
\Delta^2 = -\frac{1}{2}(X-\mu)^T\Sigma^{-1}(X-\mu) = -\frac{1}{2}X^T\Sigma^{-1}X + X^T\Sigma^{-1}\mu + const
\end{align}$$

Hence, from the above expressed quadratic form, we can compute $\Sigma^{-1}$ by computing the coefficint of second order term of $X$ and using the linear term of $X$, we can get $\Sigma^{-1}\mu$ from which we can obtain mean $\mu$.

Considering the expanded quadratic form of conditional distribution given above, the second order term of $X_a$ is $-\frac{1}{2}X_a^T\Lambda_{aa}X_a$. From this we can conclude that the covariance matrix for conditional distribution is given as

$$\begin{align}
\Sigma_{a|b} = \Lambda_{aa}^{-1}
\end{align}$$

Linear term in $X_a$ is

$$\begin{align}
X_a^T[\Lambda_{aa}\mu_a - \Lambda_{ab}(X_b - \mu_b)]
\end{align}$$

The coefficient of $X_a$ must equal $\Sigma_{a|b}^{-1}\mu_{a|b}$ and hence

$$\begin{align}
\Sigma_{a|b}^{-1}\mu_{a|b} = \Lambda_{aa}\mu_a - \Lambda_{ab}(X_b - \mu_b)
\end{align}$$

$$\begin{align}
\mu_{a|b} = \Sigma_{a|b}\bigg(\Lambda_{aa}\mu_a - \Lambda_{ab}(X_b - \mu_b)\bigg)
\end{align}$$

$$\begin{align}
= \Lambda_{aa}^{-1}\bigg(\Lambda_{aa}\mu_a - \Lambda_{ab}(X_b - \mu_b)\bigg) = \mu_a - \Lambda_{aa}^{-1}\Lambda_{ab}(X_b - \mu_b) 
\end{align}$$

Using matrix algebra, we can express the mean and covariance of conditional distribution in termes of partitioned mean and covariance matrix of joint distribution as

$$\begin{align}
\mu_{a|b} = \mu_{a} + \Sigma_{ab}\Sigma_{bb}^{-1}(X_b - \mu_b)
\end{align}$$

$$\begin{align}
\Sigma_{a|b} = \Sigma_{aa} - \Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba}
\end{align}$$

### 2.3.2 Marginal Gaussian Distribution

The <b>marginal distribution</b> of a joint Gaussian, given as

$$\begin{align}
p(X_a) = \int p(X_a,X_b)dX_b
\end{align}$$

is also Gaussian. It can be shown using the similar approach which is used for condition distribution above. The mean and covariance of marginal distribution is given as:

$$\begin{align}
E[X_a] = \mu_a
\end{align}$$

$$\begin{align}
Cov[X_a] = \Sigma_{aa}
\end{align}$$