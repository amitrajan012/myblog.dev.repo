+++
date = "2018-11-22T09:17:41+05:30"
description = "Tests for a Population Mean (Large and Small Samples)"
draft = false
tags = ["Hypothesis Testing", "Mean", "Null Hypothesis", "Alternate Hypothesis", "One-tailed Test", "Two-tailed Test", "Student's t Distribution"]
title = "Hypothesis Testing (Part 1)"
topics = ["Hypothesis Testing"]
+++

</br>
<b>Hypothesis Testing</b> is a method by which we can test an assumption made for a population parameter. For example, the statement $\mu > 10$ is an assumption or <b>hypothesis</b> about the population mean $\mu$. To check the validity of this hypothesis, we must conduct a <b>hypothesis test</b>. Hypothesis tests are closely related to confidence intervals.

Prior to performing a hypothesis test, we need to formulate the <b>null</b> and <b>alternate hypothesis</b>. <b>Null hypothesis</b> states that the effect indicated by the sample is only due to chance or random variation. The <b>alternate hypothesis</b> indicates that the effect indicated by the sample is real and it accurately represents the whole population. For example, if we want to test that whether the population mean $\mu > 10$ by analyzing the data from a sample, the null and alternate hypothesis will be:

$$H_0 : \mu \leq 10$$

$$H_A : \mu > 10$$

While performing a hypothesis test, we put the null hypothesis on trial. We make an assumption that the null hypothesis is true. We then obtain the plausibility of the null hypothesis being true from the random samples (which can be termed as the evidence). This plausibility is denoted by a number between 0 and 1 and is called as <b>p-value</b>. <b>p-value can also be seen as the probability of observing the value of statistic which is under test (here population mean) whose disagreement with $H_0$ is at least as great as the observed value of statistic (here $\overline{X}$) under the assumption that null hypothesis is true</b>. Hence, <b>smaller the p-value, stronger the evidence against null hypothesis $H_0$</b>. If p-value is sufficiently small, we can <b>reject the null hypothesis</b> and can say that the alternate hypothesis is true.

</br>
### Tests for a Population Mean (Large-Sample) :

Below are few examples which show how to conduct a hypothesis test for a population mean for large samples.

<b>Example: </b> A certain type of automobile engine emits a mean of 100 mg of oxides of nitrogen per second at 100 horsepower. A modification to the engine design has been proposed that may reduce the emissions. The new design will be put into production if it can be demonstrated that its mean emission rate is less than 100 mg/s. A sample of 50 modified engines are built and tested. The sample mean emission is 92 mg/s, and the sample standard deviation is 21 mg/s. Conduct a test to validate the claim (new mean emission is less than 100mg/s).

<b>Sol:</b> First of all, our interest is in the validation of the fact that whether the mean emission rate is less than 100mg/s. So this can be termed as an alternate hypothesis. Generally, <b>what is claimed is termed as alternate hypothesis</b>. Hence,

$$H_0: \mu \geq 100$$

$$H_A: \mu < 100$$

Now, we make the assumption that the null hypothesis is true and will compute the distribution of sample mean $\overline{X}$. As $\overline{X}$ is the mean of a large sample ($n = 50$), from central limit theorem, it comes from a normal distribution whose mean is $\mu$ and variance is $\sigma^2 /50$, where $\mu$ is population mean and $\sigma^2$ is population variance. Now, we need to find the values of $\mu$ and $\sigma^2$.

We are assuming that the null hypothesis is true, i.e. $\mu \geq 100$. This does not provide a specific value of $\mu$. So, we take the value of $\mu$ as close as possible to alternate hypothesis and hence $\mu = 100$. As the sample is large, we can estimate the population variance as the sample variance $s = 21 mg/s$ and hence the distribution under the assumption that null hypothesis is true (also called as <b>null distribution</b>) is $\overline{X} \sim N(100, (\frac{21}{\sqrt{50}})^2) = N(100, 2.97^2)$

Now, we need to calculate the <b>p-value</b> for the test. <b>p-value can be seen as the probability of observing the value of statistic which is under test (here population mean) whose disagreement with $H_0$ is at least as great as the observed value of statistic (here $\overline{X}$) under the assumption that null hypothesis is true</b>. Here, $\overline{X} = 92$, and we need to find the probability that a number drawn from the null distribution (which is $N(100, 2.97^2)$) is <b>less than or equal to</b> 92. This can be computed by determining the <b>z-score</b>:

$$z = \frac{92 - 100}{2.97} = -2.69$$

The corresponding p-value is <b>0.0036</b>. The p-value can be interpreted in two ways: <b>Either $H_0$ is false</b> or <b>$H_0$ is true, which implies that out of all the samples that can be drawn from the null distribution, only 0.36% can have the sample mean as small as the observed value</b>. In practice, the second conclusion is not feasible and hence we can assume that the null hypothesis is false and can <b>reject the null hypothesis</b>. Here, the calculation of p-value is done on the basis of z-score and hence it is called as <b>test statistic</b>. p-value is also called as the <b>observed significance level</b>.

<b>Example:</b> A scale is to be calibrated by weighing a 1000 g test weight 60 times. The 60 scale readings have mean 1000.6 g and standard deviation 2 g. Find the p-value for testing $H_0 :\mu = 1000$ versus $H_A : \mu \neq 1000$.

<b>Sol:</b> Here, the null distribution has mean 1000 and standard deviation $\frac{2}{\sqrt{60}} = 0.258$. Hence the z-score is:

$$z = \frac{1000.6 - 1000}{0.258} = 2.32$$

The corresponding probability is <b>0.0102</b>. Now, as $H_0$ is specified by $\mu = 1000$, regions in both the tails of the normal curve is in disagreement with $H_0$. Hence, the p-value is the sum of both the probabilities which is <b>0.0204</b>. Hence, we have a strong evidence against $H_0$ and it can be rejected. This is called as <b>two-sided</b> or <b>two-tailed</b> test. The test conducted in the previous example was <b>one-sided</b> or <b>one-tailed</b> test.

There often arises some sort of misunderstanding about the conclusions that can be drawn from a hypothesis test. Only two conclusions that can be drawn from the test are: either <b>$H_0$ is false</b> or <b>$H_0$ is plausible</b>. We can never conclude that $H_0$ is true. To reject $H_0$, we decide a <b>significance level</b> of the test and reject the null hypothesis is p-value is <b>less than</b> the significance level of the test. In general, smaller the p-value, less plausible is the null hypothesis. It is often in the best practice to report the p-value rather than just reporting the result of the hypothesis test (whether the null hypothesis is rejected or not). It should be noted that <b>p-value is not the probability of $H_0$ being true</b>.

One common misconception about hypothesis testing is the fact that if a result is statistically significant, it is practically significant as well. Let us look at this with the help of an example.

<b>Example:</b> Assume that a process used to manufacture synthetic fibers is known to produce fibers with a mean breaking strength of 50 N. A new process, which would require considerable retooling to implement, has been developed. In a sample of 1000 fibers produced by this new method, the average breaking strength was 50.1 N, and the standard deviation was 1 N. Can we conclude that the new process produces fibers with greater mean breaking strength?

<b>Sol:</b> The null and alternate hypothesis can be given as:

$$H_0: \mu \leq 50$$

$$H_A: \mu > 50$$

The calculated z-score is:

$$z = \frac{50.1 - 50}{1/\sqrt{1000}} = \frac{0.1}{0.0316} = 3.16$$

Hence, the p-value is <b>0.0008</b>, the null hypothesis can be rejected and can be said that the result is statistically significant. But as the increase in the breaking strength is almost negligible, the result obtained is not practically significant. The main reason for this is the <b>lower value of standard deviation</b>.

</br>
### Tests for a Population Mean (Small-Sample) :

When the sample size is small, we can not estimate the population standard deviation from sample standard deviation, and hence the large-sample method can not be applied. However, when the population is approximately normal, the <b>Student's t distribution</b> can be used.

<b>Example:</b> Spacer collars for a transmission countershaft have a thickness specification of 38.98–39.02 mm. The process that manufactures the collars is supposed to be calibrated so that the mean thickness is 39.00 mm, which is in the center of the specification window. A sample of six collars is drawn and measured for thickness. The six thicknesses are 39.030, 38.997, 39.012, 39.008, 39.019, and 39.002. Assume that the population of thicknesses of the collars is approximately normal. Can we conclude that the process need recalibration?

<b>Sol:</b> Here, $\overline{X} = 39.01133$ and $s = 0.011928$. We need to find that whether the process needs recalibration or not. The process will need the recalibration when the population mean will be different from 39.00 mm. Hence, the null and alternate hypothesis can be given ad:

$$H_0: \mu = 39.00$$

$$H_A: \mu \neq 39.00$$

The t-statistic is:

$$t = \frac{\overline{X} - \mu}{s/\sqrt{n}} = 2.327$$

This is a <b>two-tailed</b> test and hence the p-value is the sum of area under the curves $t < 2.327$ and $t < -2.327$, i.e. $0.05 < p-value < 0.1$. Hence, we can not reject the null hypothesis and can not conclusively state that the process is out of calibration.

When a small sample is taken from a normal population whose standard deviation $\sigma$ is <b>known</b>, we can use <b>z-statistic</b> instead of t-statistic as we are not approximating $\sigma$ with $s$.

</br>
### Reference :

https://www.mheducation.com/highered/product/statistics-engineers-scientists-navidi/M0073401331.html
