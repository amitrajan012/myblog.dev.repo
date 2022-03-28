+++
date = "2018-11-23T19:38:29+05:30"
description = "Tests for a Population Proportion"
draft = false
tags = ["Hypothesis Testing", "Proportion", "Null Hypothesis", "Alternate Hypothesis", "One-tailed Test", "Two-tailed Test"]
title = "Hypothesis Testing (Part 2)"
topics = ["Hypothesis Testing"]
+++

</br>
### Tests for a Population Proportion :

The hypothesis testing for population proportion can be conducted in a similar manner. Here are some examples to depict it.

<b>Example:</b> A supplier of semiconductor wafers claims that of all the wafers he supplies, no more than 10% are defective. A sample of 400 wafers is tested, and 50 of them, or 12.5%, are defective. Can we conclude that the claim is false?

<b>Sol:</b> First of all, the supplier claims that the percentage of defective wafer is less than 10%. Let us denote the proportion of defective wafer by $p$, then the suppliers claim is: $p \leq 0.1$. As the sample size is large, we can apply the central limit theorem and can say that the sample proportion comes from a normal distribution denoted as:

$$\widehat{p} \sim N\bigg(p, \frac{p(1-p)}{n}\bigg)$$

where $p$ is the population proportion and $n = 400$ is the sample size. Our goal is to find that whether the supplier's claim is false or not. Hence, the null and alternate hypothesis can be defined as:

$$H_0: p \leq 0.1$$

$$H_A: p > 0.1$$

To perform hypothesis test, we assume that null hypothesis is true and hence $p=0.1$. The <b>null distribution</b> comes out to be:

$$\widehat{p} \sim N(0.1, 2.25 \times 10^{-4})$$

The standard deviation and the observed value of $\widehat{p}$ is $\sqrt{2.25 \times 10^{-4}} = 0.015$ and $50 /400 = 0.125$. Hence, the <b>z-score</b> is:

$$z = \frac{0.125 - 0.1}{0.015} = 1.67$$

The corresponding <b>p-value</b> is <b>0.0475</b>. For a <b>significance level</b> of 5%, we can <b>reject the null hypothesis</b> and say that the supplier's claim is false.

The only necessary condition to conduct the mentioned test is the fact that sample size must be large, or there should be <b>more than 10</b> outcomes for both the classes, i.e. $np_0 > 10$ and $n(1-p_0) > 10$.

</br>
### Reference :

https://www.mheducation.com/highered/product/statistics-engineers-scientists-navidi/M0073401331.html
