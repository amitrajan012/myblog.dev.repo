+++
date = "2022-07-06T23:07:28+01:00"
description = "Pattern Recognition (Bishop): Chapter 4"
draft = false
tags = ["Bishop", "Pattern Recognition", "Linear Models", "Classification", "Laplace Approximation", "Model Comparison", "BIC", "Occam factor", "Bayesian Logistic Regression"]
title = "Linear Models for Clasification - The Laplace Approximation & Bayesian Logistic Regression"
topics = ["Pattern Recognition"]

+++


## 4.4 The Laplace Approximation

In the Bayesian treatment of logistic regression, we can not integrate exactly over the parameter $W$ as the posterior distribution is no longer Gaussian. <b>Laplace approximation</b> aims to find a Gaussian approximation to a probability density defined over a set of continuous variables. Let for a single continuous variable $z$, the distribution is defined as

$$\begin{align}
p(z) = \frac{1}{Z}f(z)
\end{align}$$

where $Z = \int f(z)dz$ is the <b>normalizing coefficient</b>. In the Laplace method, the goal is to find a Gaussian approximation $q(z)$ of $p(z)$ which is centred on the mode of the distribution $p(z)$. The first step is to find the model of $p(z)$, i.e. the point $z_0$ such that $p^{'}(z_0) = 0$, or 

$$\begin{align}
\frac{df(z)}{dz}\bigg|_{z=z_0} = 0
\end{align}$$

As the logarithm of Gaussian distribution is a quadratic function of the variables, we can consider its <b>Taylor expansion</b> centerd around the mode $z_0$ (the first order term vanishes as it contains the first order differential of $f(z)$ at mode $z_0$ which is $0$) and is given as

$$\begin{align}
\ln f(z) \simeq \ln f(z_0) - \frac{1}{2}A(z-z_0)^2
\end{align}$$

where 

$$\begin{align}
A = - \frac{d^2}{dz^2} \ln f(z)\bigg|_{z=z_0}
\end{align}$$

After taking the exponential, the expression reduces to

$$\begin{align}
f(z) \simeq f(z_0) exp\bigg[-\frac{A}{2}(z-z_0)^2\bigg]
\end{align}$$

This takes the form of a Gaussian and the normalization coefficient $f(z_0)$ can be determined by comaring it to standard Gaussian distribution. The Gaussian approximation is then given as

$$\begin{align}
q(z) = \bigg(\frac{A}{2\pi}\bigg)^{1/2} exp\bigg[-\frac{A}{2}(z-z_0)^2\bigg]
\end{align}$$

Note that the Gaussian approximation will only be well defined if its precision $A > 0$, in other words the stationary point $z_0$ must be a local maximum, so that the second derivative of $f(z)$ at the point $z_0$ is negative.

For a distribution $p(Z) = f(Z)/C$ defined over a $M$-dimensional space, the Laplace approximation is obtained in a similar fashion and is given as

$$\begin{align}
q(Z) = \frac{|A|^{1/2}}{(2\pi)^{M/2}} exp\bigg[-\frac{1}{2}(Z-Z_0)^TA(Z-Z_0)\bigg]
\end{align}$$

where $A$ is a $M \times M$ <b>Hessian</b> matrix given as

$$\begin{align}
A = -\nabla\nabla \ln f(Z)\bigg|_{Z=Z_0}
\end{align}$$

Hence, in order to apply the Laplace approximation we first need to find the mode $Z_0$, and then evaluate the Hessian matrix at that mode.

Many of the distributions encountered in practice will be multimodal and so there will be different Laplace approximations according to which mode is being considered. Note that the normalization constant $C$ of the true distribution does not need to be known in order to apply the Laplace method. As a result of the central limit theorem, the posterior distribution for a model is expected to become increasingly better approximated by a Gaussian as the number of observed data points is increased, and so we would expect the Laplace approximation to be most useful in situations where the number of data points is relatively large.

### 4.4.1 Model Comparison and BIC

Using Laplace method to approximate the distribution $p(Z)$, we can also obtain an approximation for the normalizing coefficient $C$ as

$$\begin{align}
C = \int f(Z)dZ \simeq f(Z_0) \int exp\bigg[-\frac{1}{2}(Z-Z_0)^TA(Z-Z_0)\bigg] dZ
\end{align}$$

$$\begin{align}
= f(Z_0) \frac{(2\pi)^{M/2}}{|A|^{1/2}}
\end{align}$$

For the purpose of <b>Bayesian model comparison</b>, consider a data set $D$ and a set of models $\{M_i\}$ having parameters $\{\theta_i\}$. For each model, we are interested in calculating the <b>model evidence</b> $p(D|M_i)$. The model evidence be further decomposed as the product of likelihood function $p(D|M_i,\theta_i)$ and the prior over parameters $p(\theta_i|M_i)$. Hence,

$$\begin{align}
p(D|M_i) = \int p(D|M_i,\theta_i)p(\theta_i|M_i) d\theta_i
\end{align}$$

Remove $M_i$ from the condition and replacing $\theta_i$ with $\theta$ to keep the notations uncluttered, we have

$$\begin{align}
p(D) = \int p(D|\theta)p(\theta) d\theta
\end{align}$$

Comparing it with general distribution equation, we have the normalizing coefficient $C=p(D)$ and the distribution function without normalizing coefficeint $f(\theta) = p(D|\theta)p(\theta)$. Using Laplace method of finding the normalizing coefficient $C$, we have

$$\begin{align}
C = p(D) = f(\theta_0) \frac{(2\pi)^{M/2}}{|A|^{1/2}} = f(\theta_{MAP}) \frac{(2\pi)^{M/2}}{|A|^{1/2}}
\end{align}$$

$$\begin{align}
C = p(D) = p(D|\theta_{MAP})p(\theta_{MAP}) \frac{(2\pi)^{M/2}}{|A|^{1/2}}
\end{align}$$

It should be noted that $\theta_0 = \theta_{MAP}$. Taking logarithm, we have

$$\begin{align}
\ln p(D) = \ln p(D|\theta_{MAP}) + \ln p(\theta_{MAP}) + \frac{M}{2} \ln(2\pi) - \frac{1}{2} \ln|A|
\end{align}$$

where $A$ is the Hessian matrix given as

$$\begin{align}
A = - \nabla\nabla \ln f(\theta)\bigg|_{\theta=\theta(MAP)} 
\end{align}$$

$$\begin{align}
\implies A = - \nabla\nabla \ln p(D|\theta)p(\theta)\bigg|_{\theta=\theta(MAP)}
\end{align}$$

$$\begin{align}
\implies A =  - \nabla\nabla \ln p(\theta|D)\bigg|_{\theta=\theta(MAP)}
\end{align}$$

The term $\ln p(D|\theta_{MAP})$ is the log likelihood evaluated at optimized parameter and the remaining term is called as the <b>Occam factor</b> which penalizes model complexity

$$\begin{align}
\text{Occam factor} = \ln p(\theta_{MAP}) + \frac{M}{2} \ln(2\pi) - \frac{1}{2} \ln|A|
\end{align}$$

Under certain conditions, the expression can further approximated as

$$\begin{align}
\ln p(D) \simeq \ln p(D|\theta_{MAP}) - \frac{M}{2} \ln N + const
\end{align}$$

where $N$ is the number of data points, $M$ is the number of parameters in $\theta$. This is called as <b>Bayesian Informatio Criterion(BIC)</b> or the <b>Schwarz criterion</b>.

## 4.5 Bayesian Logistic Regression

Exact Bayesian inference for logistic regression is intractable. In particular, evaluation of the posterior distribution would require normalization of the product of a prior distribution and a likelihood function that itself comprises a product of logistic sigmoid functions, one for every data point. Instead we can use Laplace approximation to fit the model. It is explained earlier that the Laplace approximation is obtained by finding the mode of the posterior distribution and then fitting a Gaussian centred at that mode. This requires evaluation of the second derivatives of the log posterior, which is equivalent to finding the Hessian matrix.

Assuming a Gaussian prior for the parameters which is given as

$$\begin{align}
p(W) = N(W|m_0,S_0)
\end{align}$$

the posterior distribution over parameter $W$ is given as

$$\begin{align}
p(W|t) \propto p(W)p(t|W)
\end{align}$$

where the likelihood function $p(t|W)$ is given as

$$\begin{align}
p(t|W) = \prod_{n=1}^{N} y_n^{t_n}(1-y_n)^{1-t_n}
\end{align}$$

Substituting the value of prior distribution and likelihood function in the expression of posterior distribution and taking logarithm, we get

$$\begin{align}
\ln p(W|t) = -\frac{1}{2}(W-m_0)^TS_0^{-1}(W-m_0) + \sum_{n=1}^N t_n \ln y_n + (1-t_n) \ln (1-y_n) + const
\end{align}$$

where $y_n = \sigma(W^T\phi_n)$. The Gaussian approximation of the posterior distribution can be obtained using Laplace method. We first maximize the posterior distribution to get the maximum posterior solution $W_{MAP}$ which serves as the mean of the Gaussian. Covariance is given as the Hessian mtrix

$$\begin{align}
S_N = -\nabla\nabla \ln p(W|t) = S_0^{-1} + \sum_{n=1}^N y_n(1-y_n)\phi_n\phi_n^T
\end{align}$$

Finally, the Gaussian approximation of the posterior distribution takes the form

$$\begin{align}
q(W) = p(W|t) = N(W|W_{MAP},S_N)
\end{align}$$

Once we have obtained the posterior distribution, we get the final predictions by marginalizing with respect to the obtained posterior distribution. For a two-class case, the predictive distribution for class $C_1$ given a new feature vector $\phi$ is obtained by marginalizing with respect to the posterior distribution $p(W|t)$ and is given as

$$\begin{align}
p(C_1|\phi,t) = \int p(C_1|\phi,W) p(W|t) dW = \int \sigma(W^T\phi) q(W) dW
\end{align}$$

The probability for class $C_2$ is then given as $p(C_2|\phi,t) = 1 - p(C_1|\phi,t)$.
