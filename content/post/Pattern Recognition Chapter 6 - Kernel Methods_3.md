+++
date = "2022-08-04T23:07:28+01:00"
description = "Pattern Recognition (Bishop): Chapter 6"
draft = false
tags = ["Bishop", "Pattern Recognition", "Linear Models", "Gaussian Process", "Gram Matrix", "Automatic Relevance Determination"]
title = "Kernel Methods - Gaussian Process"
topics = ["Pattern Recognition"]

+++

## 6.4 Gaussian Process

For a linear regression models of the form $y(X,W) = W^T\phi(X)$ in which $W$ is a vector of parameters and $\phi(X)$ is a vector of fixed nonlinear basis functions that depend on the input vector $X$, we showed that a prior distribution over $W$ induced a corresponding prior distribution over functions $y(X,W)$. Given a training data set, we then evaluated the posterior distribution over $W$ and thereby obtained the corresponding posterior distribution over regression functions, which in turn (with the addition of noise) implies a predictive distribution $p(t|X)$ for new input vectors $X$.

### 6.4.1 Linear Regression Revisited

Consider a model defined in terms of a linear combination of $M$ fixed basis functions given by the elements of the vector $\phi(X)$ so that

$$\begin{align}
y(X) = W^T\phi(X)
\end{align}$$

where $X$ is the input vector and $W$ is the $M$-dimensional weight vector. Consider a prior distribution over $W$ given by an <b>isotropic Gaussian</b> of the form

$$\begin{align}
p(W) = N(0, \alpha^{-1}I)
\end{align}$$

governed by the hyperparameter $\alpha$, which represents the precision (inverse variance) of the distribution. <b>For any given value of $W$, $y(X) = W^T\phi(X)$ defines a particular function of $X$</b>. The probability distribution over $W$, which is given as $p(W)$ therefore induces a probability distribution over functions $y(X)$. In practice, we wish to evaluate this function at specific values of $X$, for example at the training data points $X_1,X_2,...,X_N$. We are hence interested in the joint distribution of the function values $y(X_1),y(X_2),...,y(X_N)$, which is denoted by the vector $y$ with elements $y_n = y(X_n)$ for $n=1,2,...,N$. This vector is given as

$$\begin{align}
y = \Phi W
\end{align}$$

where $\Phi$ is the <b>design matrix</b> with elements $\Phi_{nk} = \phi_k(X_n)$.

The probablity distribution of $y$ can be found as follows. First of all, $y$ is a linear combination of Gaussian distributed variables given by the elements of $W$ and hence itself is Gaussian. The mean and covariance of $y$ is then given as

$$\begin{align}
E[y] = \Phi E[W] = 0
\end{align}$$

$$\begin{align}
cov[y] = E[yy^T] = E[\Phi WW^T\Phi^T] = \Phi E[ WW^T]\Phi^T = \Phi \alpha^{-1} I \Phi^T = \frac{1}{\alpha}\Phi\Phi^T = K
\end{align}$$

where $K$ is the <b>Gram matrix</b> with elements

$$\begin{align}
K_{nm} = k(X_n,X_m) = \frac{1}{\alpha}\phi(X_n)^T\phi(X_m)
\end{align}$$

where $k(X_n,X_m)$ is the <b>kernel function</b>. This model provides us with a particular example of a Gaussian process. In general, a Gaussian process is defined as a probability distribution over functions $y(X)$ such that the set of values of $y(X)$ evaluated at an arbitrary set of points $X_1,X_2,...,X_N$ jointly have a Gaussian distribution.

### 6.4.2 Gaussian Process for Regression

In order to apply Gaussian process models to the problem of regression, we need to take into account the noise on the observed target variables, which are given by

$$\begin{align}
y_n = t_n + \epsilon_n
\end{align}$$

where $y_n = y(X_n)$ and $\epsilon_n$ is a random noise variable whose value is choosen independently for each observation $n$. Considering the Gaussian distribution for noise, we have

$$\begin{align}
p(t_n|y_n) = N(t_n|y_n, \beta^{-1})
\end{align}$$

where $\beta$ is the hyperparameter representing <b>precision</b> of the noise. As the noise is independent for all datapoints, the joint distribution of the target variable $t=(t_1,t_2,...,t_N)^T$ conditioned on the values of $y=(y_1,y_2,...,y_N)^T$ is given by an isotropic Gaussian of the form

$$\begin{align}
p(t|y) = N(t|y, \beta^{-1}I_N)
\end{align}$$

From the definition of Gaussian process, the marginal distribution $p(y)$ is given by the Gaussian whose mean is zero and whose covariance is defined by the Gram matrix $K$ so that

$$\begin{align}
p(y) = N(y|0, K)
\end{align}$$

The kernel function that determines $K$ is typically chosen to express the property that, for points
$X_n$ and $X_m$ that are similar, the corresponding values $y(X_n)$ and $y(X_m)$ will be more strongly correlated than for dissimilar points.

The marginal distribution $p(t)$ can be given as

$$\begin{align}
p(t) = \int p(t|y)p(y)dy = N(t|0,C)
\end{align}$$

where the covariance matrix $C$ has elements

$$\begin{align}
C_{nm} = C(X_n,X_m) = k(X_n,X_m) + \beta^{-1}\delta_{nm}
\end{align}$$

This result reflects the fact that the two Gaussian sources of randomness, namely that associated with $y(X)$ and that associated with $\epsilon$, are independent and so their covariances simply add.

So far, we have used the Gaussian process viewpoint to build a model of the joint distribution over sets of data points. Our goal in regression, however, is to make predictions of the target variables for new inputs, given a set of training data. Let us suppose that $T_n = (t_1,t_2,...,t_N)^T$, corresponding to input values $X_1,X_2,...,X_N$ is the observed training set, and our goal is to predict the target variable $t_{N+1}$ for a new input vector $X_{N+1}$. This requires that we evaluate the predictive distribution $p(t_{N+1}|T_{N+1})$. Note that this distribution is conditioned also on the variables $X_1,X_2,...,X_N$ and $X_{N+1}$.

To find the conditional distribution $p(t_{N+1}|T_{N+1})$, we begin with the joint distribution $p(T_{N+1})$, where $T_{N+1}$ denotes the vector $(t_1,t_2,...,t_N, t_{N+1})^T$. This joint distribution over $t_1,t_2,...,t_N,t_{N+1}$ will be given by

$$\begin{align}
p(T_{N+1}) = N(T_{N+1}|0,C_{N+1})
\end{align}$$

where $C_{N+1}$ is a covariance matrix with elements given as

$$\begin{align}
C_{nm} = C(X_n,X_m) = k(X_n,X_m) + \beta^{-1}\delta_{nm}
\end{align}$$

To find the conditional distribution $p(t_{N+1}|T_N)$, we can use the results in [https://amitrajan012.github.io/post/pattern-recognition-chapter-2-probability-distributions_4/] to partition the variable as $(t_{N+1}, T_N)^T$, This gives the partitioned covariance matrix as

$$\begin{align}
C_{N+1}= \begin{pmatrix}
c && k^T\\\\
k && C_N
\end{pmatrix}
\end{align}$$

where $c=k(X_{N+1}, X_{N+1}) + \beta^{-1}$ and the vector $k$ has elements $k(X_n,X_{N+1})$, with the mean of both partitions as zero. This leaves us with the mean and the covariance of the Gaussian conditional distribution $p(t_{N+1},T_N)$ as

$$\begin{align}
m(X_{N+1}) = k^TC_N^{-1}T_N = k^TC_N^{-1}t
\end{align}$$

$$\begin{align}
\sigma^2(X_{N+1}) = c - k^TC_N^{-1}k
\end{align}$$

These are the key results that define Gaussian process regression. Because the vector $k$ is a function of the test point input value $X_{N+1}$, we see that the predictive distribution is a Gaussian whose mean and variance both depend on $X_{N+1}$. The only restriction on the kernel function is that the covariance matrix $C$ should be positive semidefinite.

The mean of the predictive distribuition can be written as a function of $X_{N+1}$ in the form

$$\begin{align}
m(X_{N+1}) = \sum_{n=1}^{N}a_n k(X_n,X_{N+1})
\end{align}$$

where $a_n$ is the $n^{th}$ element of $C_N^{-1}T_N$.

The above results define the predictive distribution for Gaussian process regression with an arbitrary kernel function $k(X_n,X_m)$. We can therefore obtain the predictive distribution either by taking a parameter space viewpoint and using the linear regression result or by taking a function space viewpoint and using the Gaussian process result.

### 6.4.3 Learning the Hyperparameters

The predictions of a Gaussian process model will depend, in part, on the choice of covariance function. In practice, rather than fixing the covariance function, we may prefer to use a parametric family of functions and then infer the parameter values from the data.

Techniques for learning the hyperparameters are based on the evaluation of the likelihood function $p(t|\theta)$ where $\theta$ denotes the hyperparameters of the Gaussian process model. The simplest approach is to make a point estimate of $\theta$ by maximizing the log likelihood function. The predictive distribution for the Gaussian process regression model is given as

$$\begin{align}
p(t|\theta) = N(t|0,C_N) = \frac{1}{(2\pi)^{N/2}|C_N|^{1/2}} \exp\bigg( -\frac{1}{2} t^T C_N^{-1} t\bigg)
\end{align}$$

The log likelihood function for a Gaussian process regression model is then evaluated as

$$\begin{align}
\ln p(t|\theta) = -\frac{N}{2}\ln (2\pi)  -\frac{1}{2}\ln|C_N| -\frac{1}{2} t^T C_N^{-1} t
\end{align}$$

which can then be maximized with respect to $\theta$ to get the optimal value of hyperparameters.

### 6.4.4 Automatic Relevance Determination

Optimization of parameter values of Gaussian process model by maximum likelihood allows the relative importance of different inputs to be inferred from the data. Consider a Gaussian process with a two dimensional input space $X = (X_1,X_2)$, having a kernel function of the form

$$\begin{align}
k(X,X^{'}) = \theta_0 \exp\bigg(-\frac{1}{2}\sum_{i=1}^{2}\eta_i(X_i - X_i^{'})^2 \bigg)
\end{align}$$

In the above kernel function, as a particular parameter $\eta_i$ becomes small, the function becomes relatively insensitive to the corresponding input variable $X_i$. By adapting these parameters to a data set using maximum likelihood, it becomes possible to detect input variables that have little
effect on the predictive distribution, because the corresponding values of $\eta_i$ will be small. This can be useful in practice because it allows such inputs to be discarded. 

### 6.4.5 Gaussian Process for Classification

In a probabilistic approach to classification, our goal is to model the posterior target variable for a new input vector, given a set of training data. These probabilities must lie in the interval $(0,1)$, whereas a Gaussian process model makes predictions that lie on the entire real axis. We can easily
adapt Gaussian processes to classification problems by transforming the output of the Gaussian process using an appropriate nonlinear activation function.

Consider a two-class problem with a target variable $t \in \{0,1\}$. If we define a Gaussian process over a function $a(X)$ and then transform the function using a logistic sigmoid $y = \sigma(a)$, then we will obtain a non-Gaussian stochastic process over functions $y(X)$ where $y \in (0,1)$. The probability distribution over the target variable $t$ is then given by the <b>Bernoulli</b> distribution

$$\begin{align}
p(t|a) = \sigma(a)^t(1 - \sigma(a))^{1-t}
\end{align}$$

For a training set $X_1,X_2,...,X_N$ with te corresponding target variable $t = (t_1,t_2,...,t_N)^T$ and a single test point $X_{N+1}$ with target $t_{N+1}$, our goal is to determine the predictive distribution $p(t_{N+1}|t)$, where the conditioning on input variables is left implicit. To do this, we introduce a Gaussian process prior over vector $a_{N+1}$ as

$$\begin{align}
p(a_{N+1}) = N(a_{N+1}|0,C_{N+1})
\end{align}$$

Unlike the regression case, the covariance matrix no longer includes a noise term because we assume that all of the training data points are correctly labelled. However, for numerical reasons it is convenient to introduce a noise-like term governed by a parameter $v$ that ensures that the covariance matrix is positive definite. Thus the covariance matrix $C_{N+1}$ has elements given by

$$\begin{align}
C(X_n,X_m) = k(X_n,X_m) + v\delta_{nm}
\end{align}$$

where $k(X_n,X_m)$ is any positive semidefinite kernel function and value of $v$ is fixed in advance. For two-class problems, it will be sufficient to predict $p(t_{N+1}=1|t)$, which is given as

$$\begin{align}
p(t_{N+1}=1|t) = \int p(t_{N+1}=1|a_{N+1}) p(a_{N+1}|t) da_{N+1}
\end{align}$$

where $p(t_{N+1}=1|a_{N+1}) = \sigma(a_{N+1})$. This integral is analytically intractable, and so may be approximated using sampling methods. The approximation fot the convolution of a logistic sigmoid with a Gaussian distribution can be done easily. This approximation can be used provided we have a Gaussian approximation to the posterior distribution $p(a_{N+1}|t)$. The usual justification for a Gaussian approximation to a posterior distribution is that the true posterior will tend to a Gaussian as the number of data points increases as a consequence of the central limit theorem. In the case of Gaussian processes, the number of variables grows with the number of data points, and so this argument does not apply directly. However, if we consider increasing the number of data points falling in a fixed region of $X$ space, then the corresponding uncertainty in the function $a(X)$ will decrease, again leading asymptotically to a Gaussian.
