+++
date = "2022-11-04T23:07:28+01:00"
description = "Pattern Recognition (Bishop): Chapter 9"
draft = false
tags = ["Bishop", "Pattern Recognition", "Mixture Models", "Expectation Maximization", "Mixtures of Gaussians", "Complete Data", "K-means", "Bernoulli Distributions", "Latent Class Analysis", "Resposibility"]
title = "Mixture Models and Expectation Maximization - An Alternative View of EM"
topics = ["Pattern Recognition"]

+++

## 9.3 An Alternative View of EM

The goal of the EM algorithm is to find the maximum likelihood solutions for the models having latent variables. The set of all observed data is denoted by $X$ where $n^{th}$ row represents $x_n^T$ and the latent variables by $Z$ where the $n^{th}$ row represets $z_n^T$. Let the set of all model parameters be denoted by $\theta$. Then, we have

$$\begin{align}
\ln p(X|\theta) = \ln \bigg[ \sum_Z p(X,Z|\theta) \bigg]
\end{align}$$

The expression will be valid for continuous variable with summation replaced with integration. The presence of summation inside the logarithm prevents the logarithm to directly act on the joint distribution and resulting in complicated expression for maximum likelihood solution. For each observation in $X$, we have a correspondoing value of latent variable $Z$. The data set $\{X,Z\}$ is called as the <b>complete data set</b> and maximization of log likelihood function for the complete data set $\ln p(X,Z|\theta)$ is straightforward. 

### 9.3.1 Gaussian Mixtures Revisited

For the EM algorithm for Gaussian mixture model, our goal is to maximize the log likelihood function which is computed using the observed data set. This is difficult due to the presence of summation inside the logarithm. The distribution for latent variable and the conditional distribution for $X$ given the latent variable is shown below.

$$\begin{align}
p(Z) = \prod_{k=1}^{K} \pi_{k}^{z_k}
\end{align}$$

$$\begin{align}
p(X|Z) = \prod_{k=1}^{K} N(X|\mu_k,\Sigma_k)^{z_k}
\end{align}$$

The likelihood function for the combined data set $\{X,Z\}$ then takes the form

$$\begin{align}
p(X,Z|\mu,\Sigma,\pi) = \prod_{n=1}^{N}\prod_{k=1}^{K} \pi_{k}^{z_{nk}} N(X_n|\mu_k,\Sigma_k)^{z_{nk}}
\end{align}$$

where $z_{nk}$ denotes the $k^{th}$ component of $z_n$. Taking the logarithm, we have

$$\begin{align}
\ln p(X,Z|\mu,\Sigma,\pi) = \sum_{n=1}^{N} \sum_{k=1}^{K} z_{nk}[\ln \pi_k + \ln N(X_n|\mu_k,\Sigma_k)] 
\end{align}$$

The logarithm now acts directly on the Gaussian distribution. Consider first the maximization with respect to mean and covariance. $z_n$ is a $K$-dimensional vector with all elements equal to $0$ except for one having a value of $1$. Hence, the complete-data log likelihood function is simply a sum of $K$ independent contributions, one for each mixture component.  Hence, the maximization with respect to mean and variance is a exactly as for a single Gaussian except that it involves a subset of data points that are assigned to that component. Maximization problem with respect to mixing component can be solved using Lagrange multiplier and leads to the result

$$\begin{align}
\pi_k = \frac{1}{N}\sum_{n=1}^{N} z_{nk}
\end{align}$$

which is equal to the fraction of data points assigned to the corresponding component.

### 9.3.2 Relation to K-means

The $K$-means algorithm perform a hard assignment of data points to the clusters whereas the EM algorithm makes a soft assignment. We can derive the $K$-means algorithm as a particular limit of EM for Gaussian mixture. Consider a Gaussian mixture model in which the covariance matrices of the mixture components are given by $\epsilon I$, where $\epsilon$ is a variance parameter which is shared by all the components. This gives us

$$\begin{align}
p(X|\mu_k, \Sigma_k) = \frac{1}{(2\pi\epsilon)^{1/2}} \exp \bigg( -\frac{1}{2\epsilon} ||X-\mu_k||^2\bigg)
\end{align}$$

The resposnibility for a particular data point for a mixture of Gaussian is given as

$$\begin{align}
\gamma(z_{nk}) = \frac{\pi_k N(X_n | \mu_k,\Sigma_k)}{\sum_{j} \pi_j N(X_n | \mu_j,\Sigma_j)}
\end{align}$$

Considering $\epsilon$ as constant and replacing the values in above expression, we have

$$\begin{align}
\gamma(z_{nk}) = \frac{\pi_k \exp \bigg( -\frac{1}{2\epsilon} ||X_n-\mu_k||^2\bigg)}{\sum_{j} \pi_j \exp \bigg( -\frac{1}{2\epsilon} ||X_n-\mu_j||^2\bigg)}
\end{align}$$

Considering the limit $\epsilon \to 0$, in the denominator the term for which $||X_n-\mu_j||^2$ is smallest will go to zero most slowly and hence the resposibilities $\gamma(z_{nk})$ for data points $X_n$ all go to zero except for term $j$, for which the responsibility will go to unity. Hence, in this limit, we get the hard assignment of data points as in $K$-means. Each data point is assigned to the cluster having the closest mean.

### 9.3.3 Mixtures of Bernoulli Distributions

Let us discuss the mixture of discrete binary variables described by Bernoulli distributions. This model is also known as <b>latent class analysis</b>. Consider a set of $D$ binary variables $x_i$ where $i = 1,,...,D$, each of which is governed by a Bernoulli distribution with parameter $\mu_i$, such that

$$\begin{align}
p(X|\mu) = \prod_{i=1}^{D} \mu_i^{x_i} (1 - \mu_i)^{(1 - x_i)}
\end{align}$$

where $X = (x_1,x_2,...,x_D)^T$ and $\mu = (\mu_1, \mu_2,..., \mu_D)^T$. The mean and covariance of the distribution is 

$$\begin{align}
E[X] = \mu
\end{align}$$

$$\begin{align}
Cov[X] = \text{diag} \{\mu_i(1-\mu_i)\}
\end{align}$$

A finite mixture of these distributions is given as

$$\begin{align}
p(X|\mu,\pi) = \sum_{k=1}^{K} \pi_k p(X|\mu_k)
\end{align}$$

where

$$\begin{align}
p(X|\mu_k) = \prod_{i=1}^{D} \mu_{ki}^{x_i} (1 - \mu_{ki})^{(1 - x_i)}
\end{align}$$

The mean and covariance of this mixture distribution is given as

$$\begin{align}
E[X] = \sum_{k=1}^{K} \pi_k \mu_k
\end{align}$$

$$\begin{align}
Cov[X] = \sum_{k=1}^{K} \pi_k\{\Sigma_k + \mu_k \mu_k^T\} - E[X]E[X]^T
\end{align}$$

where $\Sigma_k = diag\{\mu_{ki}(1-\mu_{ki})\}$. As covariance matrix is no longer diagonal, the distribution can capture the correlation between the variables. For a data set $X = \{X_1, X_2, ..., X_N\}$, the log likelihood fucntion is given as

$$\begin{align}
\ln p(X|\mu,\pi) = \ln \bigg[ \prod_{n=1}^{N} \sum_{k=1}^{K} \pi_k p(X_n|\mu_k)\bigg] = \sum_{n=1}^{N} \ln \bigg[\sum_{k=1}^{K} \pi_k p(X_n|\mu_k)\bigg]
\end{align}$$

which again has summation inside logarithm and hence doesn't have a closed form solution. Now, to use the EM algorithm for the maximization of the log likelihood function, we introduce a latent variable $Z$ associated with each instance of $X$, which is a binary $K$-dimensional variable having a single component equal to $1$, with all other components equal to $0$. The conditional distribution of $X$ given the latent variable can be written as

$$\begin{align}
p(X|Z,\mu) = \prod_{k=1}^{K} p(X|\mu_k)^{z_k}
\end{align}$$

where the prior distribution of latent variable is given as

$$\begin{align}
p(Z|\pi) = \prod_{k=1}^{K} \pi_k^{z_k}
\end{align}$$

The complete-data log likelihood function is given as

$$\begin{align}
p(X,Z|\mu, \pi) = \prod_{n=1}^{N} p(X_n|Z,\mu) p(Z|\pi) = \prod_{n=1}^{N} p(Z|\pi) \prod_{k=1}^{K} p(X_n|\mu_k)^{z_{nk}}
\end{align}$$

$$\begin{align}
= \prod_{n=1}^{N} p(Z|\pi) \prod_{k=1}^{K} \bigg[ \prod_{i=1}^{D} \mu_{ki}^{x_{ni}} (1 - \mu_{ki})^{(1 - x_{ni})} \bigg]^{z_{nk}}
\end{align}$$

$$\begin{align}
= \prod_{n=1}^{N} \prod_{k=1}^{K} \pi_k^{z_{nk}} \bigg[ \prod_{i=1}^{D} \mu_{ki}^{x_{ni}} (1 - \mu_{ki})^{(1 - x_{ni})} \bigg]^{z_{nk}}
\end{align}$$

Taking logarithm, we get the log likelihood function as

$$\begin{align}
\ln p(X,Z|\mu, \pi) = \sum_{n=1}^{N} \sum_{k=1}^{K} z_{nk} \bigg[ \ln \pi_k + \sum_{i=1}^{D} [x_{ni} \ln \mu_{ki} + (1 - x_{ni}) \ln (1 - \mu_{ki}) \bigg]
\end{align}$$

Taking expectation with respect to the latent variable, we have

$$\begin{align}
E_Z[\ln p(X,Z|\mu, \pi)] = \sum_{n=1}^{N} \sum_{k=1}^{K} \gamma(z_{nk}) \bigg[ \ln \pi_k + \sum_{i=1}^{D} [x_{ni} \ln \mu_{ki} + (1 - x_{ni}) \ln (1 - \mu_{ki}) \bigg]
\end{align}$$

where $E[z_{nk}] = \gamma(z_{nk})$ is the responsibility and is given as

$$\begin{align}
\gamma(z_{nk}) = \frac{\pi_k p(X_n | \mu_k)}{\sum_{j=1}^{K} \pi_j p(X_n | \mu_j)}
\end{align}$$

Considering the sum over $n$ in the expression for expectation with respect to latent variable, responsibility enters over two terms, which can be expressed as

$$\begin{align}
N_k = \sum_{n=1}^N \gamma(z_{nk})
\end{align}$$

$$\begin{align}
\overline{X_k} = \frac{1}{N_k} \sum_{n=1}^N \gamma(z_{nk}) X_n
\end{align}$$

$N_k$ is the effective number of data point associated with the comonent $k$. In the M step, the expected complete-data log likelihood is maximized with respect to $\mu_k$ and $\pi$. Using derivatives and equating them to $0$, we have

$$\begin{align}
\mu_k = \overline{X}_k
\end{align}$$

$$\begin{align}
\pi_k = \frac{N_k}{N}
\end{align}$$

This means that the mean of the component $k$ is given as the weighted mean of the data where weighing coefficients given by the responsibilities that component $k$ takes for data points.
