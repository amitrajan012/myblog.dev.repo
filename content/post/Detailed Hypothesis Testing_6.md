+++
date = "2018-11-26T07:26:51+05:30"
description = "Tests for Variances and Power of a Test"
draft = false
tags = ["Hypothesis Testing", "Normal Distribution", "Equality of Variances", "F Test", "Type-1 Error", "Type-2 Error", "Power"]
title = "Hypothesis Testing (Part 6)"
topics = ["Hypothesis Testing"]
+++

</br>
### Tests for Variances of Normal Populations :

Let $X_1, X_2, ..., X_n$ be a simple random sample from a normal population given as $N(\mu, \sigma^2)$. The sample variance $s^2$ is given as:

$$s^2 = \frac{1}{n-1}\sum_{i=1}^{n} (X_i - \overline{X})^2$$

Then the test statistic $\frac{(n-1)s^2}{\sigma_0^2}$ follows a <b>chi-square distribution</b> with $n-1$ degrees of freedom. The null hypothesis can take any of the form:

$$H_0: \sigma^2 \leq \sigma_0^2;\ \sigma^2 = \sigma_0^2;\ \sigma^2 \geq \sigma_0^2$$

<b>Example:</b> To check the reliability of a scale in a butcher shop, a test weight known to weigh 400 grams was weighed 16 times. For the scale to be considered reliable, the variance of repeated measurements must be less than 1. The sample variance of the 16 measured weights was $s^2$ = 0.81. Assume that the measured weights are independent and
follow a normal distribution. Can we conclude that the population variance of the measurements is less than 1?

<b>Sol:</b> The null and alternate hypothesis are given as:

$$H_0: \sigma \geq 1$$

$$H_A: \sigma < 1$$

The test statistic is $\frac{(n-1)s^2}{\sigma_0^2} = \frac{15 \times 0.81}{1^2} = 12.15$

From the table of chi-square distribution for degrees of freedom $n-1 = 15$, we find that the lower 10% point is 8.547, and hence we can conclude that the $p-value > 0.10$ and can not reject the null hypothesis stating that: <b>We can not conclude that the scale is reliable</b>.

</br>
### The F Test for Equality of Variance :

Let $X_1, X_2, ..., X_m$ and $Y_1, Y_2, ..., Y_n$ be simple random samples from normal populations $N(\mu_1, \sigma_1^2)$ and $N(\mu_2, \sigma_2^2)$. Let $s_1^2$ and $s_2^2$ be the sample variances which are given as:

$$s _1^2 = \frac{1}{m-1} \sum _{i=1}^{m} (X_i - \overline{X})^2$$

$$s_1^2 = \frac{1}{n-1} \sum _{i=1}^{n} (Y_i - \overline{Y})^2$$

The null and alternate hypothesis for the equality of variance can be formulated as:

$$H_0: \frac{\sigma_1^2}{\sigma_2^2} \leq 1; \ \frac{\sigma_1^2}{\sigma_2^2} \geq 1; \ \frac{\sigma_1^2}{\sigma_2^2} = 1$$

$$H_A: \frac{\sigma_1^2}{\sigma_2^2} > 1; \ \frac{\sigma_1^2}{\sigma_2^2} < 1; \ \frac{\sigma_1^2}{\sigma_2^2} \neq 1$$

The test statistic is given as:

$$F = \frac{s_1^2}{s_2^2}$$

The distribution of $F$ under null hypotheis follows <b>F distribution</b>. The F distribution has two degrees of freedom: one associated with the numerator and another with the denominator. Let us look at an example of the test for equality of variance.

<b>Example:</b> In a series of experiments to determine the absorption rate of certain pesticides into skin, measured amounts of two pesticides were applied to several skin specimens. After a time, the amounts absorbed (in μg) were measured. For pesticide A, the variance of the amounts absorbed in 6 specimens was 2.3, while for pesticide B, the variance of the amounts absorbed in 10 specimens was 0.6. Assume that for each pesticide, the amounts absorbed are a simple random sample from a normal population. Can we conclude that the variance in the amount absorbed is greater for
pesticide A than for pesticide B?

<b>Sol:</b> First of all we need to test that whether variance in the amount absorbed is greater for pesticide A than for pesticide B. Hence, the null and alternate hypothesis is formulated as:

$$H_A: \sigma_A^2 > \sigma_B^2$$

$$H_0: \sigma_A^2 \leq \sigma_B^2 \equiv \frac{\sigma_A^2}{\sigma_B^2} \leq 1$$

The test statistic is given as:

$$F = \frac{s_A^2}{s_B^2} = \frac{2.3}{0.6} = 3.83$$

Under the null hypothesis, the test statistic follows F distribution with degrees of freedoms 5 and 9 $(F_{5, 9})$. If $H_0$ is true, $s_A^2$ will on average be smaller than $s_B^2$. Hence, larger the value of $F$, stronger the evidence against the null hypothesis. From the F distribution table, we find that the upper 5% point is 3.48 and the upper 1% point is 6.06. Hence, we can conclude that $0.01 < p-value < 0.05$, i.e. we can reject the null hypothesis stating that there is an evidence that the amount absorbed is greater for pesticide A than for pesticide B.

As the F-table contains only the large values (greater than 1) for the F statistic, it is not feasible obtain a p-value for the tests, where F statistic is less then 1. In this case we can flip the test statistic (use $\frac{s_2^2}{s_1^2}$ instead of $\frac{s_1^2}{s_2^2}$). The new test statistic follows the F distribution with degrees of freedoms $n-1, m-1$ instead.

For a two-tailed test, both a small and large value of the test statistic provide evidence against null hypothesis. We can use either of $\frac{s_1^2}{s_2^2}$ or $\frac{s_2^2}{s_1^2}$ (whichever is greater than 1) as the test statistic. The <b>p-value</b> for the test is twice of the p-value for the one-tailed test. The F test is quite sensitive to the departure from the normality.

</br>
### Power :

<b>Type-2 error</b> is the error when the null hypothesis $H_0$ is not rejected when it is false. The <b>power</b> of a test is the probability of rejecting $H_0$ when it is false. Hence, it can be given as:

$$Power = 1 - P(Type \ 2 \ Error)$$

<b>Type-1 error</b> is the error when null hypothesis is rejected even if it is true. We can minimize the type-1 error by choosing a small value for the significance level $\alpha$. For a test to be efficient, the type-1 and 2 error should be minimized, i.e. the power of the test should be larger. To compute the power, first of all we need to compute the rejection region. Then we compute the probability that the test statistic falls in the rejection region given that the alternate hypothesis is true. This probability gives the power of the test. Let us look at an example.

<b>Example:</b> Find the power of the 5% level test of $H_0 :μ ≤ 80$ versus $H_A :μ > 80$ for the mean yield of the new process under the alternative $μ = 82$, assuming n = 50 and σ = 5.

<b>Sol:</b> First of all we beed to find the rejection region. Here, $\sigma = 5$, hence $\sigma_{\overline{X}} = \frac{\sigma}{\sqrt{50}} = 0.707$. The rejection region for 5% level test is given as $z \geq 1.645$ and hence the rejection region is $\overline{X} \geq 80+1.645(0.707) = 81.16$. Now, under the assumption that the alternate hypothesis is true, the z-statistic is:

$$z = \frac{81.16 - 82}{0.707} = -1.19$$

The area to the right of this is <b>0.8830</b>, which is the power of the test.

As the alternate mean is far away from the null mean, the power of the test will be almost equal to 1. For the value of the alternate mean closer to the null mean, the power of the test will be almost equal to the significance level of the test. Power of test is increased by increasing the sample size as well.

</br>
### Reference :

https://www.mheducation.com/highered/product/statistics-engineers-scientists-navidi/M0073401331.html
