+++
date = "2018-10-29T02:01:07+01:00"
description = "Hypothesis testing"
draft = false
tags = ["Hypothesis testing", "p-value", "z-statistic", "t-statistic", "Power", "F-test"]
title = "Hypothesis testing"
topics = ["Hypothesis testing"]

+++

</br>
### Introduction :

<b>Hypothesis testing</b> is a procedure that is used to determine that whether a made statistical statement (known as <b>hypothesis</b>) is a reasonable one and should not be rejected, or is unreasonable and should be rejected. Hypothesis testing setup is initialized by formulating a <b>Null Hypothesis</b> ($H_0$), which is the hypothesis associated with a contradiction to the theory that one would like to prove and <b>Alternate Hypothesis</b> ($H_A$), which is the hypothesis associated with the theory that one would like to prove. Then an appropriate <b>test statistic</b> and <b>level of significance</b> is chosen.

The choice of test statistic depends on the problem statement. For example:

 - When the population standard deviation is <b>known</b>, and if either the data is <b>normally distributed</b> or the <b>sample size > 30</b>, <b>z-statistic</b> (normal distribution) is used.


 - When the population standard deviation is <b>unknown</b>, and if either the data is <b>normally distributed</b> or the <b>sample size > 30</b>, <b>t-statistic</b> (t-distribution) is used.

The <b>rejection region</b> for the null hypothesis is decided by the <b>level of significance</b> ($\alpha$) of the test. If the <b>test statistic</b> falls in the rejection region, <b>null hypothesis is rejected</b>.

Two types of error can result from a hypothesis test. <b>Type I error</b> occurs when a null hypothesis is rejected even if it is true. Probability of Type I error is given by the <b>level of significance</b> ($\alpha$). <b>Type II error</b> occurs if we fail to reject a null hypothesis which is false. The probability of commiting Type II error is denoted by $\beta$. The probability of not committing the Type II error is called as the <b>power</b> of the test and is equal to $(1 - \beta)$.

<b>P-value</b> measures the strength of evidence in the support of a null hypothesis. If the p-value is <b>less than the level of significance</b>, we reject the null hypothesis.

Furthermore, a statistical test can be classified as one-tailed and two-tailed tests. In a <b>one-tailed test</b>, the rejection region is only on one side. For example, the null-hypothesis $H_0: \mu > 0$ is an example of one-tailed test. When the rejection region is on the both sides of the distribution, the test is called as <b>two-tailed test</b>. A null hypotghesis of $H_0: \mu = 0$ is an example of a two-tailed test. For a two-tailed test, the level of significance is halved.

</br>
### Example :

 1. The average score of all sixth graders in school District A on a math aptitude exam is 75 with a standard deviation of 8.1. A random sample of 100 students in one school was taken. The mean score of these 100 students was 71. Does this indicate that the students of this school are significantly <b>less skilled</b> in their mathematical abilities than the average student in the district? (Use a 5% level of significance.)

The test statistic can be formulated as:

 - Null Hypothesis $H_0: \mu \geq 75$
 - Alternate Hypothesis $H_A: \mu < 75$

Level of significance can be given as $\alpha = 0.05$. We are going to use <b>z-test</b> as population standard deviation is known and the sample size is greater than 30. The z-score can be given as:

$$z = \frac{x - \mu}{\sigma /\ \sqrt{n}} = \frac{71 - 75}{8.1 /\ \sqrt{100}} = -4.938$$

We can find the rejection region from the z-distribution table. The rejection region comes out to be $z < -1.645$ as $P(z < -1.645) = 0.05$ and hence we can <b>reject</b> the null hypothesis stating that the the students of this school are significantly <b>less skilled</b> in their mathematical abilities than the average student in the district.

</br>
### Factors That Affect Power :

The power of the test is highly affected by the sample size. Other things being equal, <b>greater the sample size, greater the power of the test</b>. If we decrease the significance level, we are narrowing the rejection region and hence we are less likely to reject the null hypothesis. Hence, <b>lower the significance level, lower the power of the test</b>.

</br>
### F-test :

An <b>F-test</b> is any statistical test in which the test statistic has an <b>F-distribution</b> under the null hypothesis. It is used to compare statistical models that have been fitted to a data-set in order to identify the model that best fits the population from which the data-set has been sampled. Another use is in the test for the null hypothesis that the two normal populations have the <b>same variance</b>. The formula for the one-way analysis of variance is given as:

$$F = \frac{explained \ variance}{unexplained \ variance}$$

In a regression setting, the F-statistic is given as (where the terms have usual meaning):

$$F = \frac{(TSS - RSS) / (p-1)}{RSS / (n-p)}$$

</br>
### Reference :

http://cfcc.edu/faculty/cmoore/0801-HypothesisTests.pdf

https://stattrek.com/hypothesis-test/hypothesis-testing.aspx

https://en.wikipedia.org/wiki/F-test
