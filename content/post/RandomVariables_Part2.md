+++
date = "2018-11-12T03:01:07+01:00"
description = "Continuous Random Variables"
draft = false
tags = ["Random Variables", "Continuous Random Variables", "PMF", "CDF", "Mean", "Variance"]
title = "Random Variables (Part 2: Continuous Random Variables)"
topics = ["Random Variables"]
+++

</br>
#### Continuous Random Variables :

A <b>continuous random variable</b> is a random variable which can take infinitely many values. The probabilities associated with a continuous RV is defined by probability density function(PDF).

</br>
#### Probability Density Function (PDF) :

As a continuous RV takes infinite values, the probability $P(X=x)$ for it can not be defined and takes a value of 0. Instead we define a <b>probability density funaction</b>, which intutively depicts probability per unit space, where space is defined by the range of the underlying random variable. For a continuous RV $X$ with a PDF $f(x)$, the probability can be given as:

$$P(a \leq X \leq b) = \int _{a}^{b} f(x) dx$$

If we integrate the PDF for the entire range of $X$, it will be 1, i.e. $\int _{-\infty}^{\infty} f(x) dx = 1$

</br>
#### Cumulative Distribution Function (Continuous RV) :

The <b>cumulative distribution funaction (CDF)</b> of a continuous random variable is given as:

$$F(x) = P(X \leq x) = \int _{-\infty}^{x} f(t) dt$$

</br>
#### Mean, Median and Variance of Continuous RV :

The <b>mean</b> and <b>variance</b> of a continuous RV is defined in a similar way as discrete RV.

$$\mu_X = \int _{-\infty}^{\infty} x f(x) dx$$

$$\sigma_X^2 = \int _{-\infty}^{\infty} (x-\mu_X)^2 f(x) dx$$

The <b>median</b> is the point which divides the dataset into two equal halves. Hence, it can be calculted by solving the following equation (where $x_m$ is the median):

$$F(x_m) = P(X \leq x_m) = \int _{-\infty}^{x_m} f(x) dx = 0.5$$

</br>
#### Linear Functions of Random Variables :

If $X$ is a random variable and $a$ and $b$ are constants:

$$\mu _{aX+b} = a\mu_X + b$$

$$\sigma _{aX+b}^2 = a^2\sigma_X^2$$

$$\sigma _{aX+b} = \lvert a \rvert\sigma_X$$

</br>
### Reference :

https://www.mheducation.com/highered/product/statistics-engineers-scientists-navidi/M0073401331.html
