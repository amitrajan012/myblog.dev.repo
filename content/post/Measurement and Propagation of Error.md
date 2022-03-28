+++
date = "2018-11-14T02:18:29+01:00"
description = "Measurement and Propagation of Error"
draft = false
tags = ["Independent Measurements", "Repeated Measurements", "Dependent Measurements", "Propagation of Error", "Neural Networks"]
title = "Measurement and Propagation of Error (Part 1)"
topics = ["Measurement", "Propagation of Error"]
+++

</br>
Any measurement in a scientific or an engineering process, consists of two error parts: <b>systematic error</b> or <b>bias</b> and <b>random error</b>. The bias is the part of the error that is same for every measurement. Random error varies from measurement to measurement and averages out to 0 in the long run. Hence, the measured value can be written as:

$$Measured \ Value = True \ Value + Bias + Random \ Error$$

The <b>mean</b> $\mu$ of the population represents the part of the measurement that is the same for every measurement. Hence, $\mu$ is the <b>sum of the true value and the bias</b>. The smaller the bias, the more accurate the measuring process is. If the mean is equal to the true value, the measuring process is said to be <b>unbiased</b>. The <b>standard deviation</b> $\sigma$ of the population is the the standard deviation of the random error. The <b>precision</b> of the measurement is determined by the standard deviation of the measurement process. The smaller the value of $\sigma$, the more precise the process. $\sigma$ is often referred to as the <b>uncertainty</b> in the measuring process.

For example, if $X_1, X_2, ..., X_n$ are the independent measurements, all made on the same quantity by the same process, the <b>sample standard deviation</b> $s$ can be used to estimate the <b>uncertainty</b> in the process. If the <b>true value</b> of the measuring quantity is known, the sample mean $\overline{X}$ can be used to estimate the bias as $Bias \approx \overline{X} - True \ Value$. If the true value is unknown, the bias can not be estimated from the repeated measurements. If bias has been reduced to a negligible level, the measurements can be described as:

$$Measured \ Value \pm \sigma$$

</br>
#### Linear Combinations of Independent Measurements :

If $X$ is a measurement and $c$ a constant, then the <b>uncertainty</b> in the measurement of $cX$ can be given as:

$$\sigma _{cX} = \left|c\right| \sigma _{X}$$

For the independet measurements $X_1,X_2, ..., X_n$ and constants $c_1, c_2, ..., c_n$, the <b>uncertainty</b> of the measurement $c_1X_1 + c_2X_2 + ... + c_nX_n$ is given as:

$$\sigma _{c_1X_1 + c_2X_2 + ... + c_nX_n} = \sqrt{c_1^2\sigma _{X_1}^2 + c_2^2\sigma _{X_2}^2 + ... + c_n^2\sigma _{X_n}^2}$$

The above mentioned calculation of uncertainties is a simple implication of the linear combination of random variables.

</br>
#### Repeated Measurements :

Taking repeated independent measurements of the same quantity is a good way to reduce the overall uncertainty or variance of the measurement. The average of these measurements will have the same mean but the standard deviation will be reduced by a factor of square root of number of measurements. Hence, for $n$ <b>independent measurements</b> $X_1,X_2, ..., X_n$, each with mean $\mu$ and uncertainty $\sigma$, the sample mean $\overline{X}$ is a measurement with mean

$$\mu _{\overline{X}} = \mu$$

and with uncertainty

$$\sigma _{\overline{X}} = \frac{\sigma}{\sqrt{n}}$$

For the case of the repeated measurements $X_1,X_2, ..., X_n$ with different uncertainties $\sigma _{X_1}, \sigma _{X_2}, ..., \sigma _{X_n}$, the ovearall uncertainty of the <b>average measurement</b> can be given as:

$$\sigma _{avg} = \sqrt{\frac{1}{n^2}\sigma _{X_1}^2 + \frac{1}{n^2}\sigma _{X_2}^2 + ... + \frac{1}{n^2}\sigma _{X_n}^2}$$

Instead, we can also take the <b>weighted average</b> by taking fractions $c_1, c_2, ..., c_n$ measurements each individual one such that $c_1 + c_2 + ... + c_n = 1$. The uncertainty of the final weighted average measurement will be:

$$\sigma _{weighted \ average} = \sqrt{c_1^2\sigma _{X_1}^2 + c_2^2\sigma _{X_2}^2 + ... + c_n^2\sigma _{X_n}^2}$$

<b>Example: </b> An engineer measures the period of a pendulum (in seconds) to be 2.0 ± 0.2 s. Another independent measurement is made with a more precise clock, and the result is 2.2 ± 0.1 s. The average of these two measurements is 2.1 s. Find the uncertainty in this quantity.

<b>Sol: </b> The uncertainty will be:

$$\sigma _{avg} = \sqrt{\frac{1}{n^2}\sigma _{X}^2 + \frac{1}{n^2}\sigma _{Y}^2} = \sqrt{\frac{1}{4}(0.2)^2 + \frac{1}{4}(0.1)^2} = 0.11s$$

<b>Example: </b> In the above scenario, another engineer suggests that since Y is a more precise measurement than X, a weighted average in which Y is weighted more heavily than X might be more precise than the unweighted average. Express the uncertainty in the weighted average $cX + (1 − c)Y$ in terms of $c$, and find the value of c that minimizes the uncertainty

<b>Sol: </b> The uncertainty for the weighted average is given as:

$$\sigma _{weighted \ average} = \sqrt{c^2\sigma _{X}^2 + (1-c)^2\sigma _{Y}^2} = \sqrt{0.04c^2 + 0.01(1-c)^2} = \sqrt{0.05c^2 - 0.02c + 0.01}$$

Taking derivative with respect to $c$ and computing it to 0, we get

$$\frac{d\sigma _{weighted \ average}}{dc} = 0.10c - 0.02 = 0$$

and hence, the required value of $c$ is <b>0.2</b>. The new mean and uncertainty is

$$\mu _{weighted \ average} = 0.2X + 0.8Y = 2.16$$

$$\sigma _{weighted \ average} = \sqrt{0.04(0.2)^2 + 0.01(0.8)^2} = 0.09s$$

</br>
#### Reference:

https://www.mheducation.com/highered/product/statistics-engineers-scientists-navidi/M0073401331.html
