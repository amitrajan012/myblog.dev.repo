+++
date = "2022-06-27T23:07:28+01:00"
description = "Pattern Recognition (Bishop): Chapter 3"
draft = false
tags = ["Bishop", "Pattern Recognition", "Linear Models", "Evidence Approximation", "Fixed Basis Functions"]
title = "Linear Models for Regression - Evidence Approximation & Limitations of Fixed Basis Function"
topics = ["Pattern Recognition"]

+++

## 3.5 The Evidence Approximation

In a fully Bayesian treatment of the linear basis function model, we would introduce prior distributions over the hyperparameters $\alpha$ and $\beta$ and make predictions by marginalizing with respect to these hyperparameters as well as with respect to the parameters $W$. However, although we can integrate analytically over either $W$ or over the hyperparameters, the complete marginalization over all of these variables is analytically intractable. Our goal is to find the predictive distribution for each of the models $p(t|X,M_i)$. If we introduce <b>hyperpriors</b>(priors over hyperparameters) over $\alpha,\beta$, the predictive distribution is obtained by marzinalizing over $W,\alpha,\beta$ and is given as

$$\begin{align}
p(t|X,M_i) = \int\int\int p(t|W,\beta,X,M_i)p(W|t,\alpha,\beta,X,M_i)p(\alpha,\beta|t,X,M_i)dW d\alpha d\beta
\end{align}$$

The first two terms are tractable and in a Gaussian setting, the product will take the form of a Gaussiam. The last term in integral is complex to compute as we don't have much knowledge about the hyperparameters $\alpha,\beta$. We can instead assume that the posterior distribution of hyperparameters is shraply peaked at $\alpha^{*},\beta^{*}$. Hence, the expression reduces to

$$\begin{align}
p(t|X,M_i) = \int\int p(t|\alpha,\beta,X,M_i)p(\alpha,\beta|t,X,M_i) d\alpha d\beta
\end{align}$$

which can be further reduced to (assuming that the posterior distribution of hyperparameters is shraply peaked at $\alpha^{\*},\beta^{\*}$)

$$\begin{align}
p(t|X,M_i) \simeq  p(t|\alpha^{\*},\beta^{\*},X,M_i)
\end{align}$$

where

$$\begin{align}
\alpha^{\*},\beta^{\*} = argmax_{\alpha,\beta}\bigg(p(\alpha,\beta|t,X,M_i)\bigg)
\end{align}$$

Using Bayes rule (posterior is the product of likelihood and prior), it further reduces to

$$\begin{align}
\alpha^{\*},\beta^{\*} = argmax_{\alpha,\beta}\bigg(p(t|\alpha,\beta,X,M_i) p(\alpha,\beta,M_i)\bigg)
\end{align}$$

Assuming a flat prior, we can ignore the second term in the optimization problem. Hence the above expression reduces to 

$$\begin{align}
\alpha^{\*},\beta^{\*} = argmax_{\alpha,\beta}\bigg(p(t|\alpha,\beta,X,M_i)\bigg)
\end{align}$$

which is like selecting the hyperparameters which maximizes the marginal likelihood (as it is marginalized over $W$).

## 3.6 Limitations of Fixed Basis Functions

One of the main limitation of a fixed basis function (i.e. basis functions $\phi_j(X)$ are fixed before the training data set is observed) is as the dimensionality $D$ of the input space rises, the number of basis functions need to grow rapidly. But, there are two properties of real data sets that we can exploit to help alleviate this problem. First of all, the data vectors ${X_n}$ typically lie close to a nonlinear manifold whose intrinsic dimensionality is smaller than that of the input space as a result of strong correlations between the input variables. Instead, we can use localized basis functions and arrange them to be scattered in input space only in regions containing data. The second property is that target variables may
have significant dependence on only a small number of possible directions within the data manifold. Neural networks can exploit this property by choosing the directions in input space to which the basis functions respond.

