+++
date = "2018-11-14T03:08:14+01:00"
description = "Measurement and Propagation of Error"
draft = false
tags = ["Independent Measurements", "Repeated Measurements", "Dependent Measurements", "Taylor Series", "Propagation of Error", "Neural Networks"]
title = "Measurement and Propagation of Error (Part 2)"
topics = ["Measurement", "Propagation of Error"]
+++

</br>
#### Linear Combinations of Dependent Measurements :

In the case of dependent measurement, to quantify the uncertainty, we need to know the value of <b>covariance</b> for all the possible pairs of measurements. This is practically not feasible. In this case, an upper bound can be placed on the uncertainty. If $X_1, X_2, ..., X_n$ are $n$ dependent measurements and $c_1, c_2, ..., c_n$ are constants, then the uncertainty of $c_1X_1 + c_2X_2 + ... + c_nX_n$ can be bounded as:

$$\sigma _{c_1X_1 + c_2X_2 + ... + c_nX_n} \leq |c_1|\sigma _{X_1} + |c_2|\sigma _{X_2} + ... + |c_n|\sigma _{X_n}$$

</br>
#### Uncertainties for Functions of One Measurement :

If $X$ is a measurement whose uncertainty $\sigma_X$ is <b>small</b> and if $U$ is a function of $X$, then uncertainty in the measurement of $U$ can be approximated as (where the derivative is computed at the observed measurement $X$):

$$\sigma_U \approx \bigg| \frac{dU}{dX} \bigg| \sigma_X$$

The above approximation can be proved by using a simple mathematical technique. A <b>Taylor Series</b> of a function $f(X)$, which is <b>infinitely differentiable</b> at $a$ is given as:

$$f(a) + \frac{f^{'}(a)}{1!}(x-a) + \frac{f^{''}(a)}{2!}(x-a)^2 + \frac{f^{'''}(a)}{3!}(x-a)^3 + ...$$

In our case, if $X$ is close to $\mu_X$, then the <b>first-order Taylor Series approximation</b> for $U(X)$ can be given as:

$$U(X) \approx U(\mu_X) + \frac{U^{'}(\mu_X)}{1!}(x-\mu_X) = U(\mu_X) + \frac{dU}{dX}(X-\mu_X)$$

where $\frac{dU}{dX}$ is evaluated at $\mu_X$. It should be noted that for any reasonable precise measurement, $X$ will be close enough to $\mu _{X}$ for the Taylor series approximation to be valid. Rearranging the above expression, we get

$$U(X) \approx  \bigg( U(\mu_X) - \frac{dU}{dX} \mu _{X} \bigg) + \frac{dU}{dX}X$$

As $\frac{dU}{dX}$ is measured at $\mu_X$, it is constant and hence the quantity inside the bracket is constant. This gives the uncertainty (<b>standard deviation</b>) as:

$$\sigma_U \approx \bigg| \frac{dU}{dX} \bigg| \sigma_X$$

This is the <b>propagation of error</b> formula and is applied in almost all the back-propagation algorithms in <b>neural-networks</b> as well.

</br>
#### Uncertainties for Functions of Several Measurements :

If $X_1, X_2, ..., X_n$ are <b>independent measurements</b> whose uncertainties $\sigma _{X_1}, \sigma _{X_2}, ..., \sigma _{X_n}$ are <b>small</b> and if $U = U(X_1, X_2, ..., X_n)$ is a function of $X_1, X_2, ..., X_n$, then

$$\sigma_U \approx \sqrt{\bigg( \frac{\partial U}{\partial X_1}\bigg)^2 \sigma _{X_1}^2 + \bigg( \frac{\partial U}{\partial X_2}\bigg)^2 \sigma _{X_2}^2 + ... + \bigg( \frac{\partial U}{\partial X_n}\bigg)^2 \sigma _{X_n}^2}$$

This is the <b>multivariate propagation of error formula</b> and is valid only when the measurements are independent. This formula can be derived in a similar way as the one discussed for the one measurement case.

<b>Example: </b>Two resistors with resistances R1 and R2 are connected in parallel. The combined resistance R is given by R = (R1R2)/(R1+ R2). If R1 is measured to be 100±10, and R2 is measured to be 20±1, estimate R and find the uncertainty in the estimate.

<b>Sol:</b> The estimate of $R$ can be given as $\frac{100 \times 20}{100 + 20}$ = <b>16.67</b>. First of all, we need to compute the partial derivative of R as:

$$\frac{\partial R}{\partial R1} = \bigg( \frac{R2}{R1+R2}\bigg)^2 = 0.0278$$

$$\frac{\partial R}{\partial R2} = \bigg( \frac{R1}{R1+R2}\bigg)^2 = 0.694$$

Now, $\sigma _{R1} = 10$ and $\sigma _{R2} = 1$, and hence

$$\sigma _{R} = \sqrt{\bigg( \frac{\partial R}{\partial R1}\bigg)^2 \sigma _{R1}^2 + \bigg( \frac{\partial R}{\partial R2}\bigg)^2 \sigma _{R2}^2} = 0.75$$

Hence, the combined resistance is <b>16.67 ± 0.75</b>.

</br>
#### Uncertainties for Functions of Dependent Measurements :

In the case of the <b>dependent</b> measurements, the uncertainty can be calculated accurately if the covariance of each pair of measurements is known. Instead, we can give a conservative estimate for uncertainty as (where the terms have usual meaning):

$$\sigma _{U} \leq \bigg| \frac{\partial U}{\partial X_1}\bigg| \sigma _{X_1} + \bigg| \frac{\partial U}{\partial X_2}\bigg| \sigma _{X_2} + ... + \bigg| \frac{\partial U}{\partial X_n}\bigg| \sigma _{X_n}$$

This inequality is valid for almost all the cases of dependent measurements.

</br>
#### Reference:

https://www.mheducation.com/highered/product/statistics-engineers-scientists-navidi/M0073401331.html
