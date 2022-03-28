+++
date = "2018-10-28T17:11:47+01:00"
description = "Maximum Likelihood Estimation"
draft = false
tags = ["Maximum Likelihood Estimation", "MLE", "Normal Distribution", "Estimation", "Bernoulli Distribution", "Log likelihood"]
title = "Maximum Likelihood Estimation"
topics = ["Estimation"]

+++

</br>
### Introduction :

<b>Maximum Likelihood Estimation</b> is the method of estimating the <b>parameters</b> of a <b>statistical model</b>, given the observations. It attempts to find the parameter values that maximize the <b>likelihood function</b>. The process can be viewed as finding the parameters that maximize the likelihood of getting the data we observed for a particular set of statistical models.

Suppose we have the data points (random samples) $X_1, X_2, ..., X_n$ which belong to a distribution which depends on one or more unknown parameters $\theta_1, \theta_2, ..., \theta_m$ with probability density (or mass) function $f(x_i; \theta_1, \theta_2, ..., \theta_m)$. Here, $x_i$s are the observed values for $X_i$s. Our task is to find the value of parameters that maximize the probability or likelihood of getting the observed value of the data. i.e., We need to maximize the following quantity (which is called as the <b>Likelihood function</b>):

$$L(\theta) = P(X_1=x_1, X_2=x_2, ..., X_n=x_n; \theta) = \prod _{i=1}^{n}f(x_i; \theta)$$

One assumption that is made while formulating the above equation is: the samples drawn are <b>independent</b> of each other and hence, we can multiply the individual probabilities. One way to find the parameters that maximize the above quantity is by taking the derivative of the likelihood function and evaluating it to 0. We can transform the multiplicative form of the function to additive by taking the <b>natural logarithm</b> of the expression. As the logarithm is a <b>monotonically increasing function</b>, this will not dilute our objective. Taking the logarithm, we get the <b>log likelihood function</b> as:

$$ln(L(\theta)) = \sum _{i=1}^{n}ln(f(x_i; \theta))$$

</br>
### Examples :

 - Suppose we have three data points given as <b>1, 1.5</b> and <b>2.5</b>, generated from a <b>Gaussian distribution</b>. We need to calculate the <b>maximum likelihood estimate</b> of the parameters ($\mu$ and $\sigma$) of the distribution.

The probability density of a Gaussian distribution is given as:

$$P(x; \mu, \sigma) = \frac{1}{\sqrt{2\pi \sigma^2}} exp \bigg( -\frac{(x-\mu)^2}{2\sigma^2} \bigg)$$

The corresponding <b>log likelihood function</b> can be given as:

$$ln(P(x; \mu, \sigma)) = \sum \bigg[ -ln(\sigma) -\frac{ln(2\pi)}{2} -\frac{(x-\mu)^2}{2\sigma^2}\bigg]$$

This expression can be differentiated to find the <b>maximum likelihood estimator</b> of $\mu$ and $\sigma$. Substituting the values of data points and differentiating with respect to $\mu$ and evaluating it to 0, we get

$$\frac{1}{\sigma^2} [1+1.5+2.5 - 3\mu] = 0$$

Hence, maximum likelihood estimator for mean $\mu$ is <b>1.67</b>.


 - Suppose we have a random sample $X_1, X_2, ..., X_n$ where, $X_i=0,1$ when a randomly selected student does not own and own a sports car respectively. Assuming that $X_i$ are independent <b>Bernoulli</b> random variables with unknown paramtere $p$, find the maximum likelihood estimator of $p$, the proportion of students who own a sports car.

The probability mass funaction for Bernoulli distribution is given as:

$$f(x_i;p) = p^{x_i} (1-p)^{1-x_i}$$

The likelihood function is given as:

$$L(p) = \prod _{i=1}^{n}f(x_i;p) = \prod _{i=1}^{n}p^{x_i} (1-p)^{1-x_i} = p^{\sum x_i}(1-p)^{n-\sum x_i}$$

The <b>log likelihood function</b> is given as:

$$ln(L(p)) = (\sum x_i) ln(p) + (n-\sum x_i)ln(1-p)$$

Taking derivative and evaluating it to 0, we get the maximum likelihood estimator for $p$ as

$$\frac{(\sum x_i)}{p} - \frac{(n-\sum x_i)}{1-p} = 0$$

$$\sum x_i - np = 0$$

$$p = \frac{\sum x_i}{n}$$


</br>
#### Reference :

https://onlinecourses.science.psu.edu/stat414/node/191/

https://towardsdatascience.com/probability-concepts-explained-maximum-likelihood-estimation-c7b4342fdbb1
