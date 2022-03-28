+++
date = "2018-12-07T07:26:51+05:30"
description = "The Wilcoxon Signed-Rank Test: Derivation of Mean and Variance"
draft = false
tags = ["Wilcoxon Signed-Rank Test", "Distribution-Free Tests", "Hypothesis Testing"]
title = "The Wilcoxon Signed-Rank Test"
topics = ["The Wilcoxon Signed-Rank Test"]
+++

</br>

The <b>Wilcoxon Signed-Rank Test</b> is a distribution free test or non-parametric test which is used to test the population mean when the samples do not come from any specific distribution. The idea behind The Wilcoxon Signed-Rank Test is to median-center the samples and give signed-ranks to the individual values. If the sum of the negative ranks (S-) is larger, we can conclude that the population mean is lesser than the mentioned value. If the sum of the positive ranks(S+) is larger, the population mean is greater than the mentioned value. For detailed explanation, the post 
<a href="https://amitrajan012.github.io/post/detailed-hypothesis-testing_4/">  The Wilcoxon Signed-Rank Test </a> can be followed.

If the sample size is larger, the test statistic S+ is approximately normally distributed with mean $n(n+1)/4$ and variance $n(n+1)(2n+1)/4$. In this case, the Wilcoxon signed-rank test can be performed by computing the <b>z-score</b> of S+ and then the normal table can be used to find the <b>p-value</b>. The z-score is:

$$z = \frac{(S+) - n(n+1)/4}{\sqrt{n(n+1)(2n+1)/24}}$$

Let us look at the derivation for the distribution parameters.

</br>
#### Derivation of distribution mean :

The statistic used in the Wilcoxon Signed-Rank Test(let us say it $W$ which is S+) can be denoted as:

$$W = \sum _{i=1}^{n} I _{i}R _{i}$$

where $I_i$ is the indicator variable defined as 0 when $x_i - y_i$ is negative and 1 otherwise and $R_i$ is their ranks. When the sample size is $n$, the distribution of the sum of positive ranks is also equivalent to distribute the ranks from $1$ to $n$ randomly into two subsets, one denoting the positive ranks and other the negative ranks. Hence, this can be denoted as:

$$U = \sum _{i=1}^{n} U _{i}$$

where $U _{i}=i$, when the rank belongs to the positive ranks and $U _{i}=0$ otherwise (when the rank belongs to the negative ranks). As these two distributions are similar:

$$E(W) = E(U)$$

Now, as we are randomly distributing the ranks amongst the positive and negative ranks, the probability of each of the rank being negative or positive will be same and will be equal to 1/2, i.e.

$$P(U_i = 0) = P(U_i = i) = 0.5$$

The expected value of $U$ can be calculated as:

$$E(U) = \sum _{i=1}^{n} E(U_i) = \sum _{i=1}^{n} \big[\frac{1}{2}0 + \frac{1}{2}i \big] = \frac{1}{2} \sum _{i=1}^{n} i = \frac{n(n+1)}{4}$$

Hence,

$$E(W) = E(U) = \frac{n(n+1)}{4}$$

</br>
#### Derivation of distribution variance :

For the calculation of variance, we need to calculate the individual variances of each of the $U_i$s and sum them all. As, $Var(X) = E[X^2] - E^{2}[X]$, the variance of $U_i$ can be given as:

$$Var(U_i) = E[U _{i}^2] - E^{2}[U_i]$$

Expected value of $U_i$ and $U_i^2$ can be calculated as:

$$E[U_i] = 0.\frac{1}{2} + i.\frac{1}{2} = \frac{i}{2}$$

$$E[U_i^2] = 0^2.\frac{1}{2} + i^2.\frac{1}{2} = \frac{i^2}{2}$$

Hence,

$$Var(U_i) = \frac{i^2}{2} - \big(\frac{i}{2} \big)^2 = \frac{i^2}{4}$$

The variance of $W$ can be given as:

$$Var(W) = Var(U) = \sum _{i=1}^{n} Var(U_i) = \sum _{i=1}^{n} \frac{i^2}{4} = \frac{n(n+1)(2n+1)}{24}$$

</br>
#### Reference :

https://www.physicsforums.com/threads/expected-value-and-variance-for-wilcoxon-signed-rank-test.775604/
