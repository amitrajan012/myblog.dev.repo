+++
date = "2022-07-04T23:07:28+01:00"
description = "Pattern Recognition (Bishop): Chapter 4"
draft = false
tags = ["Bishop", "Pattern Recognition", "Linear Models", "Classification", "Probabilistic Generative Models", "Class Priors", "Class-conditional Densities", "Maximum Likelihood Solution"]
title = "Linear Models for Clasification - Probabilistic Generative Models (Maximum Likelihood Solution)"
topics = ["Pattern Recognition"]

+++

### 4.2.2 Maximum Likelihood Solution

Once the class-conditional densities $p(X|C_k)$ are expressed in a parametric form, the value of the parameter can be determinied using a maximum likelihood approach. This requires a data set having input $X$ together with their class labels. Suppose we have a data set $\{X_n,t_n\}$ where $n=1,2,...,N$ and $t_n=1$ for class $C_1$ and $t_n=0$ for class $C_2$. Let the prior class probablities be $p(C_1) = \pi$ and $p(C_2) = 1-\pi$. The joint densities are given as

$$\begin{align}
p(X_n,C_1) = p(C_1)p(X_n|C_1) = \pi N(X_n|\mu_1,\Sigma)
\end{align}$$

$$\begin{align}
p(X_n,C_2) = p(C_2)p(X_n|C_2) = (1-\pi) N(X_n|\mu_2,\Sigma)
\end{align}$$

assuming shared variance across class-conditional densities. Assuming independence, the likelihood function is given as

$$\begin{align}
p(t|\pi,\mu_1,\mu_2,\Sigma) = \prod_{n=1}^{N} [\pi N(X_n|\mu_1,\Sigma)]^{t_n}[(1-\pi) N(X_n|\mu_2,\Sigma)]^{1-t_n}
\end{align}$$

The log likelihood can be then maximized with respect to different parameters. For Maximizing log likelihhod with respect to $\pi$ , the part of it which depends on $\pi$ is

$$\begin{align}
\sum_{n=1}^{N} [t_n \ln(\pi) + (1-t_n) \ln(1-\pi)]
\end{align}$$

Differentiating with respect to $\pi$ and equating it to $0$, we get 

$$\begin{align}
\sum_{n=1}^{N} \bigg[\frac{t_n}{\pi} - \frac{1-t_n}{1-\pi}\bigg] = \sum_{n=1}^{N} \frac{t_n-\pi}{\pi(1-\pi)} = 0
\end{align}$$

$$\begin{align}
\pi = \frac{1}{N}\sum_{n=1}^{N} t_n = \frac{N_1}{N} = \frac{N_1}{N_1 + N_2}
\end{align}$$

where $N_1,N_2$ are the total number of data points in class $C_1,C_2$. For Maximizing log likelihhod with respect to $\mu_1$ , the part of it which depends on $\mu_1$ is

$$\begin{align}
\sum_{n=1}^{N} t_n \ln N(X_n|\mu_1,\Sigma) = -\frac{1}{2}\sum_{n=1}^{N} t_n (X_n-\mu_1)^T \Sigma^{-1} (X_n-\mu_1) + const
\end{align}$$

Differentiating with respect to $\mu_1$ and equating it to $0$, we get

$$\begin{align}
-\frac{1}{2}\sum_{n=1}^{N} 2t_n \Sigma^{-1}(X_n-\mu_1) = -\Sigma^{-1}\sum_{n=1}^{N} t_n(X_n-\mu_1) = 0
\end{align}$$

$$\begin{align}
\sum_{n=1}^{N} t_n\mu_1 = \sum_{n=1}^{N} t_nX_n \implies N_1\mu_1 = \sum_{n=1}^{N} t_nX_n
\end{align}$$

$$\begin{align}
\mu_1 = \frac{1}{N_1} \sum_{n=1}^{N} t_nX_n
\end{align}$$

Similarly, maximizing with respect to $\mu_2$, we have

$$\begin{align}
\mu_2 = \frac{1}{N_2} \sum_{n=1}^{N} (1-t_n)X_n
\end{align}$$

Hence, the maximum likelihood solutions for the means are the mean of the input vectors assigned to the respective classes. The maximum likelihood solution for the variance is given as (can be derived)

$$\begin{align}
\Sigma = \frac{N_1}{N}S_1 + \frac{N_2}{N}S_2
\end{align}$$

$$\begin{align}
S_1 = \frac{1}{N_1} \sum_{n \in C_1} (X_n - \mu_1)(X_n - \mu_1)^T
\end{align}$$

$$\begin{align}
S_2 = \frac{1}{N_2} \sum_{n \in C_2} (X_n - \mu_2)(X_n - \mu_2)^T
\end{align}$$

which <b>represents a weighted average of the covariance matrices associated with each of the two classes separately</b>.