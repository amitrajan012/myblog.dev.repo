+++
date = "2022-06-15T19:07:28+01:00"
description = "Pattern Recognition (Bishop): Chapter 2"
draft = false
tags = ["Bishop", "Pattern Recognition", "Probability Distributions", "Gaussian Distribution", "Sequential Estimation", "Maximum Likelihood for the Gaussian", "Bayes’ Theorem for Gaussian Variables"]
title = "Probability Distributions - The Gaussian Distribution: Part 3"
topics = ["Pattern Recognition"]

+++

### 2.3.3 Bayes' Theorem for Gaussian Variables

The applicatio of Bayes' theorem in the Gaussian setting is when we have Gaussian marginal distribution $p(X)$ and a Gaussian conditional distribution $p(Y|X)$ and we wish to find the marginal and consitional distribution $p(Y)$ and $p(X|Y)$. It should be noted that the mean and covariace of the marginal distribution $p(X)$ are constant with respect to $X$. The covarince of the conditional distribution $p(Y|X)$ is constant with respect to $X$ but its mean is a <b>linear function</b> of $X$. Hence, the marginal can consitional distribution can be represented as

$$\begin{align}
p(X) = N(X|\mu,\Lambda^{-1})
\end{align}$$

$$\begin{align}
p(Y|X) = N(Y|AX+b,L^{-1})
\end{align}$$

where $\mu,A,b$ are parameters governing the mean and $\Lambda,L$ are the precision matrices.

Let the joint distribution over $X,Y$, defined as $Z$ such that

$$\begin{align}
Z = \begin{pmatrix}
X\\
Y
\end{pmatrix}
\end{align}$$

From Bayes' theorem, $p(Z) = p(X,Y) = p(X)p(Y|X)$. This means that

$$\begin{align}
\ln p(Z) = \ln p(X) + \ln p(Y|X)
\end{align}$$

$$\begin{align}
= -\frac{1}{2}(X-\mu)^T\Lambda(X-\mu) -\frac{1}{2}(Y-AX-b)^TL(Y-AX-b) + const
\end{align}$$

The above expression is a quadratic form of the componnets of $Z$ and hence the joint distribution is Gaussian. Its mean and precison matrix ($R$) can be computed using the coefficients of the quadratic form of $Z$ (i.e. $\begin{pmatrix}
X\\
Y
\end{pmatrix}$)in a similar way and are given as

$$\begin{align}
R = \begin{pmatrix}
\Lambda + A^TLA & -A^TL\\
-LA & L
\end{pmatrix}
\end{align}$$

$$\begin{align}
Cov(Z) =R^{-1} = \begin{pmatrix}
\Lambda^{-1} & \Lambda^{-1}A^T\\
A\Lambda^{-1} & L^{-1}+A\Lambda^{-1}A^T
\end{pmatrix}
\end{align}$$

$$\begin{align}
E[Z] = R^{-1}\begin{pmatrix}
A\mu-A^TLb\\
Lb
\end{pmatrix} = \begin{pmatrix}
\mu\\
A\mu+b
\end{pmatrix}
\end{align}$$

This joint distribution can be used to find the parameters for the marginal and conditional distribution $p(Y)$ and p(X|Y). For the marginal distribution $p(Y)$, the mean and covariance are given as

$$\begin{align}
E[Y] = A\mu+b
\end{align}$$

$$\begin{align}
Cov[Y] = L^{-1}+A\Lambda^{-1}A^T
\end{align}$$

For the conditional distribution $p(X|Y)$, the mean and covariance are given as

$$\begin{align}
Cov[X|Y] = (\Lambda + A^TLA)^{-1}
\end{align}$$

$$\begin{align}
E[Y] = (\Lambda + A^TLA)^{-1} \bigg(A^TL(Y-b) + \Lambda\mu \bigg)
\end{align}$$



### 2.3.4 Maximum Likelihood for the Gaussian

For a dataset $X = (X_1, X_2,...,X_N)^T$ where the individual observations $X_n$ are assumed to be drawn independetly from a mutivariate Gaussian, the log likelihood function is given as

$$\begin{align}
\ln p(X|\mu,\Sigma) = -\frac{ND}{2}\ln(2\pi) -\frac{N}{2}\ln(|\Sigma|) -\frac{1}{2}\sum_{n=1}^{N}(X_n-\mu)^T\Sigma^{-1}(X_n-\mu)
\end{align}$$

Taking the derivative of the log likelihood function with respect to $\mu$, we get

$$\begin{align}
\frac{\delta}{\delta\mu}\ln p(X|\mu,\Sigma) = \sum_{n=1}^{N}\Sigma^{-1}(X_n-\mu)
\end{align}$$

To compute the derivative, this explanation is used: (https://math.stackexchange.com/questions/312077/differentiate-fx-xtax)]

Calculate the differential of the function $f: \Bbb R^n \to \Bbb R$ given by $$f(x) = x^T A x$$ with $A$ symmetric. Also, differentiate this function with respect to $x^T$.

As a start, things work "as usual": You calculate the difference between $f(x+h)$ and $f(x)$ and check how it depends on $h$, looking for a dominant linear part as $h\to 0$.
Here, $f(x+h)=(x+h)^TA(x+h)=x^TAx+ h^TAx+x^TAh+h^TAh=f(x)+2x^TAh+h^TAh$, so $f(x+h)-f(x)=2x^TA\cdot h + h^TAh$. The first summand is linear in $h$ with a factor $2x^TA$, the second summand is quadratic in $h$, i.e. goes to $0$ faster than the first / is negligible against the first for small $h$. So the row vector $2x^TA$ is our derivative (or transposed: $2Ax$ is the derivative with respect to $x^T$).

Computing the derivative to $0$, we get the maximum likelihood estimator of mean as

$$\begin{align}
\mu_{ML} = \frac{1}{N}\sum_{n=1}^{N}X_n
\end{align}$$

Similarly, the maximum likelihood estimator of covariance can be computed as

$$\begin{align}
\Sigma_{ML} = \frac{1}{N}\sum_{n=1}^{N}(X_n-\mu_{ML})(X_n-\mu_{ML})^T
\end{align}$$

### 2.3.5 Sequential Estimation

In a <b>sequential estimation</b> process, data points are processed one at a time and then discareded. This process is important for on-line applications and for large data sets. For maximum likelihood estimator for the mean $\mu_{ML}$, the updated mean after dissecting out contribution for the final data point $x_N$ is

$$\begin{align}
\mu_{ML}^{(N)} = \mu_{ML}^{(N-1)} + \frac{1}{N}(x_N - \mu_{ML}^{(N-1)})
\end{align}$$

The new mean can be interpreted as moving the old mean in the direction of the <b>error signal</b> $(x_N - \mu_{ML}^{(N-1)})$ by a small amount $1/N$. As $N$ inreases, the contribution from new data point gets smaller. There are more genric way to come up with a sequential learning algorithm when they can't be derived via this route.

