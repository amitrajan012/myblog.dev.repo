+++
date = "2022-06-23T23:07:28+01:00"
description = "Pattern Recognition (Bishop): Chapter 3"
draft = false
tags = ["Bishop", "Pattern Recognition", "Linear Models", "Basis Function", "Bias-Variance Decomposition", "Sequential Learning", "Regularized Least Squares", "Parameter Shrinkage"]
title = "Linear Models for Regression - Bias-Variance Decomposition"
topics = ["Pattern Recognition"]

+++

## 3.2 Bias-Variance Decomposition

We have seen that the use of maximum likelihood, or equivalently least squares, can lead to severe over-fitting if complex models are trained using data sets of limited size. However, limiting the number of basis functions in order to avoid over-fitting has the side effect of limiting the flexibility of the model to capture interesting and important trends in the data. <b>The phenomenon of over-fitting is really an unfortunate property of maximum likelihood and does not arise when we marginalize over parameters in a Bayesian setting.</b> From a frequentist point of view, the phenomena of overfitting can be analyzed using <b>bias-variance trade-off</b>.

Let us say we have modeled the conditional distribution $p(t|X)$ for any prediction problem. The optimal choice for the prediction can be given as conditional expectation which can be given as

$$\begin{align}
h(X) = E[t|X] = \int tp(t|X)dt
\end{align}$$

$h(X)$ can be seen as the optimal choice of $y(X)$ which is the prediction for $t$. The goal for any regression problem is to minimize the expected loss which can be given as

$$\begin{align}
E[L] == \int \int L(t,y(X))p(X,t)dXdt
\end{align}$$

For a sum-of-squares error function, the expected loss is 

$$\begin{align}
E[L] == \int \int [y(X) - t]^2 p(X,t)dXdt = \int \int [(y(X) - h(X)) - (h(X) - t)]^2 p(t|X)p(X)dXdt
\end{align}$$

$$\begin{align}
\int [y(X) - h(X)]^2 p(X)dX + \int \int [y(X) - h(X)]^2 p(X,t)dXdt
\end{align}$$

The second term, which is independent of $y(X)$ is the <b>minimum achievable</b> value of the expected loss and arises from the intrisic noise in the data. The first term depend on the solution $y(X)$ and the goal is find $y(X)$ which minimizes this. The smallest that we can hope for it to be is $0$. If we have infinite supply of data and computation power, we can find a solution $y(X)$ which will make the first term negligible. In practice, we have a limited supply of data set $D$ having $N$ data points and hence we do not know the $h(X)$ exactly.

Let us say that we model $h(X)$ using a parametric function $y(X,W)$. In a frequentist approach, we try to interpret the uncertainity of the estimate through the following thought experiment. Suppose we had a large number of data sets each of size $N$ and each drawn independently from the distribution $p(t,X)$. For any given data set $D$, we can run our learning algorithm and obtain a prediction function $y(X;D)$. Different data sets from the ensemble will give different functions and consequently different values of the squared loss. The performance of a particular learning algorithm is then assessed by taking the average over this ensemble of data sets. For a particular data set $D$, the first term takes the form

$$\begin{align}
[y(X;D) - h(X)]^2
\end{align}$$

As this quantity will depend on a particular data set $D$, we take its average over the ensemble of data sets. If we add and subtract the term $E_D[y(X;D)]$ in the expression above, it reduces to

$$\begin{align}
[y(X;D) - E_D[y(X;D)] + E_D[y(X;D)]- h(X)]^2
\end{align}$$

$$\begin{align}
= (y(X;D) - E_D[y(X;D)])^2 + (E_D[y(X;D)]- h(X))^2
\end{align}$$

$$\begin{align}
\+ 2(y(X;D) - E_D[y(X;D)])(E_D[y(X;D)]- h(X))
\end{align}$$

If we take the expectation with respect to $D$, the second term is constant. The third term reduces to 

$$\begin{align}
E_D\bigg[2(y(X;D) - E_D[y(X;D)])(E_D[y(X;D)]- h(X))\bigg]
\end{align}$$

$$\begin{align}
= 2(E_D[y(X;D)] - E_D[y(X;D)])(E_D[y(X;D)]- h(X)) = 0
\end{align}$$

Hence, the expression reduces to

$$\begin{align}
E_D\bigg[(y(X;D) - h(X))^2\bigg]
\end{align}$$

$$\begin{align}
= \bigg(E_D[y(X;D)]- h(X)\bigg)^2 + E_D\bigg[(y(X;D) - E_D[y(X;D)])^2\bigg]
\end{align}$$

$$\begin{align}
= \bigg(bias\bigg)^2 + variance
\end{align}$$

The first term, called as <b>bias</b> represents <b>the extent to which the average prediction over all data sets $E_D[y(X;D)]$ differs from the desired regression function $h(X)$</b>. The second term, called the <b>variance, measures the extent to which the solutions for individual data sets $y(X;D)$ vary around their average $E_D[y(X;D)$, and hence this measures the extent to which the function $y(X;D)$ is sensitive to the particular choice of data set</b>. Hence, <b>expected loss is the sum of squared bias, variance and noise</b>.

Although the bias-variance decomposition may provide some interesting insights into the model complexity issue from a frequentist perspective, it is of limited practical value, because the bias-variance decomposition is based on averages with respect to ensembles of data sets, whereas in practice we have only the single observed data set. If we had a large number of independent training sets of a given size, we would be better off combining them into a single large training set, which of course would reduce the level of over-fitting for a given model complexity.