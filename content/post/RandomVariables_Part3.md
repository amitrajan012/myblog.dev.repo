+++
date = "2018-11-13T06:11:47+01:00"
description = "Jointly Distributed Random Variables"
draft = false
tags = ["Random Variables", "Covariance", "Correlation", "Conditional Distributions", "Marginal Probability Density", "IID"]
title = "Random Variables (Part 3: Jointly Distributed Random Variables)"
topics = ["Random Variables"]
+++

</br>
#### Independent Random Variables :

If $X$ and $Y$ are <b>independent random variables</b> and $S$ and $T$ are sets of numbers then,

$$P(X \in S \ and \ Y \in T) = P(X \in S) P(Y \in T)$$

Hence, for independent random variables $X_1, X_2, ..., X_n$ and constants $c_1, c_2, ..., c_n$, the variance of linear combination $c_1X_1 + c_2X_2 + ... + c_nX_n$ is given as:

$$\sigma _{c_1X_1 + c_2X_2 + ... + c_nX_n}^2 = c_1^2\sigma _{X_1}^2 + c_2^2\sigma _{X_2}^2 + ... + c_n^2\sigma _{X_n}^2$$

</br>
#### Independence and Simple Random Samples :

When a simple random sample of numerical values is drawn from a population, each item in the sample can be thought of as a random variable. The items in a simple random sample may be treated as independent, except when the sample is a large proportion (more than 5%) of a finite population. Hence, if $X_1, X_2, ..., X_n$ are simple random samples, then $X_1, X_2, ..., X_n$ may be treated as <b>independent random variables</b>, all with the <b>same distribution</b> and is said that they are <b>independent and identically distributed (i.i.d.)</b>. The <b>sample mean</b>, denoted as $\overline{X}$, can be treated as the linear combination of means of different samples and is given as:

$$\overline{X} = \frac{1}{n}X_1 + \frac{1}{n}X_2 + ... + \frac{1}{n}X_n$$

Hence, the <b>mean</b> and <b>variance</b> of $\overline{X}$ can be computed as:

$$\mu _{\overline{X}} = \mu _{\frac{1}{n}X_1 + \frac{1}{n}X_2 + ... + \frac{1}{n}X_n} = \frac{1}{n}\mu _{X_1} + \frac{1}{n}\mu _{X_2} + ... + \frac{1}{n}\mu _{X_n} = \frac{1}{n}\mu + ... + \frac{1}{n}\mu = \mu$$

$$\sigma _{\overline{X}}^2 = \sigma _{\frac{1}{n}X_1 + \frac{1}{n}X_2 + ... + \frac{1}{n}X_n}^2 = \frac{1}{n^2}\sigma _{X_1}^2 + \frac{1}{n^2}\sigma _{X_2}^2 + ... + \frac{1}{n^2}\sigma _{X_n}^2 = \frac{1}{n}\sigma^2 + ... + \frac{1}{n}\sigma^2 = \frac{\sigma^2}{n}$$

This technique of reducing variance can be used in various statistical models to tackle the problem of overfitting.

</br>
#### Jointly Distributed Random Variables :

When two or more random variables are associated with each item in a population, the random variables are said to be <b>jointly distributed</b>. For <b>jointly distributed discrete RVs</b>, the <b>joint probability mass function</b> is given as: $p(x, y) = P(X=x \ and \ Y=y)$. The <b>marginal probabilities</b> can be computed as:

$$p_X(x) = P(X=x) = \sum _{y} p(x,y)$$

$$p_Y(y) = P(Y=y) = \sum _{x} p(x,y)$$

where the sum is taken over all the possible values of $Y$ and $X$. For a <b>jointly distributed continuous RVs</b>, the probability can be given as:

$$P(a \leq X \leq b \ and \ c \leq Y \leq d) = \int _{a}^{b} \int _{c}^{d} f(x, y) \ dy \ dx$$

The <b>marginal Probability densities</b> can be computed as:

$$f_X(x) = \int _{-\infty}^{\infty} f(x, y) \ dy$$

$$f_Y(y) = \int _{-\infty}^{\infty} f(x, y) \ dx$$

These concepts can be extended for the case of more than two random variables as well.

</br>
#### Conditional Distributions :

For two jointly discrete random variables $X$ and $Y$, the <b>conditional probability mass function</b> of $Y$ given $X=x$ can be given as:

$$p _{Y|X}(y|x) = \frac{p(x, y)}{p_X(x)}$$

where $p(x, y)$ is the joint probability mass function and $p_X(x)$ is the marginal probability mass function of $X$. Similarly, for two jointly continuous random variables $X$ and $Y$, the <b>conditional probability density function</b> of $Y$ given $X=x$ can be given as:

$$f _{Y|X}(y|x) = \frac{f(x, y)}{f_X(x)}$$

where all the terms have the usual meaning.

Two RVs $X$ and $Y$ are independent if $p(x, y) = p_X(x) p_Y(y)$ (for discrete RVs) and $f(x, y) = f_X(x) f_Y(y)$ (for continuous RVs).

</br>
#### Covariance  and Correlation :

The <b>covariance</b> of two RVs $X$ and $Y$ is given as:

$$Cov(X, Y) = \mu _{(X - \mu_X)(Y - \mu_Y)} = \mu _{XY} - \mu_X \mu_Y$$

It is to be noted that covariance has a unit which is the unit of $X$ multiplied by the unit of $Y$. Hence, covariance can not be used to compare the strength of relationship between different set of random variables. The <b>correlation</b> is the scaled version of covariance. To compute the correlation between $X$ and $Y$, the covariance between them is computed and then the unit is handled by dividing the covariance by the standard deviation of $X$ and $Y$. Hence, correlation is given as:

$$\rho _{X, Y} = \frac{Cov(X, Y)}{\sigma_X \sigma_Y}$$

For <b>non-independent</b> random variables $X_1, X_2, ..., X_n$, mean and variance can be computed as:

$$\mu _{c_1X_1 + c_2X_2 + ... + c_nX_n} = c_1\mu _{X_1} + c_2\mu _{X_2} + ... + c_n\mu _{X_n}$$

$$\sigma _{c_1X_1 + c_2X_2 + ... + c_nX_n}^2 = c_1^2\sigma _{X_1}^2 + c_2^2\sigma _{X_2}^2 + ... + c_n^2\sigma _{X_n}^2 + 2 \sum _{\forall i, j} c_i c_j Cov(X_i, X_j)$$


</br>
### Reference :

https://www.mheducation.com/highered/product/statistics-engineers-scientists-navidi/M0073401331.html
