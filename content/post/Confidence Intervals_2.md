+++
date = "2018-11-18T05:07:15+05:30"
description = "Confidence Intervals for Proportions and the Difference"
draft = false
tags = ["Confidence Intervals", "Proportions", "Binomial Distribution", "Bernoulli Distribution", "Student's t Distribution"]
title = "Confidence Intervals (Part 2)"
topics = ["Confidence Intervals"]
+++

</br>
#### Confidence Intervals for Proportions :

Let $X$ be the number of successes in $n$ independent Bernoulli trials with success probability $p$, where <b>the number of trials $n$ is large enough</b>, such that $X \sim Bin(n, p)$. Then the $100(1 - \alpha) \%$ confidence interval for $p$ is:

$$\widehat{p} \pm z_{\alpha/2}\sqrt{\frac{\widehat{p}(1-\widehat{p})}{n}}$$

where $\widehat{p}$ is the <b>sample proportion</b> and can be estimated as $\frac{X}{n}$. It should be noted that the quantity under the square root is the <b>sample variance</b>. This method of computation of the confidence interval for the proportion works for large sample size or we can say that the experiment should containt at least 10 successes and 10 failures.

To tackle the problem of <b>smaller sample size</b>, we need to make certain changes in the above estimate. Let $\widetilde{n} = n+4$ and $\widetilde{p} = \frac{X+2}{\widetilde{n}}$, a $100(1-\alpha)\%$ confidence interval for $p$ is given as:

$$\widetilde{p} \pm z_{\alpha/2}\sqrt{\frac{\widetilde{p}(1-\widetilde{p})}{\widetilde{n}}}$$

If the interval exceeds from the limit, we should limit it by 0 and 1 as lower and upper limit respectively. The above described method is usually preferred for the computation of confidence interval for population proportion of any sample size. The <b>one-sided</b> confidence interval is given as:

$$Lower \ bound  = \widetilde{p} - z_{\alpha}\sqrt{\frac{\widetilde{p}(1-\widetilde{p})}{\widetilde{n}}}$$

$$Upper \ bound  = \widetilde{p} + z_{\alpha}\sqrt{\frac{\widetilde{p}(1-\widetilde{p})}{\widetilde{n}}}$$

where the symbols have the usual meaning.

</br>
#### Confidence Intervals for the Difference Between Two Means :

Let $X$ and $Y$ be independent random variables, with $X \sim N(\mu_X, \sigma_X^2)$ and $Y \sim N(\mu_Y, \sigma_Y^2)$, then

$$X+Y \sim N(\mu_X + \mu_Y, \sigma_X^2 + \sigma_Y^2)$$

$$X-Y \sim N(\mu_X - \mu_Y, \sigma_X^2 + \sigma_Y^2)$$

As the distribution of the difference of two independent normal random variables is normal, we can use this property with the central limit theorem to obtain a confidence interval for the difference of the means.

Let $X _1, X_2, ..., X _{n_X}$ be a <b>large random sample</b> of size $n_X$ from a population with mean $\mu_X$ and standard deviation $\sigma_X$, and let $Y_1, Y_2, ..., Y _{n_Y}$ be a <b>large random sample</b> of size $n_Y$ from a population with mean $\mu_Y$ and standard deviation $\sigma_Y$. If the two samples are <b>independent</b>, then a $100(1-\alpha) \%$ confidence interval for the difference of means $\mu_X - \mu_Y$ is given as:

$$\overline{X} - \overline{Y} \pm z _{\alpha/2} \sqrt{\frac{\sigma_X^2}{n_X} + \frac{\sigma_Y^2}{n_Y}}$$

If the values of population standard deviations $\sigma_X$ and $\sigma_Y$ are unknown, they can be replaced with sample standard deviation $s_X$ and $s_Y$.

</br>
#### Confidence Intervals for the Difference Between Two Proportions :

The confidence interval for the difference between two proportions can be computed in a similar way as described above. Let $X$ and $Y$ be the number of successes in $n_X$ and $n_Y$ independent Bernoulli trials with success probabilities $p_X$ and $p_Y$ respectively, such that $X \sim Bin(n_X, p_X)$ and $Y \sim Bin(n_Y, p_Y)$. Let $\tilde{n_X} = n_X+2$, $\tilde{n_Y} = n_Y+2$, $\tilde{p_X} = \frac{X+1}{\tilde{n_X}}$ and $\tilde{p_Y} = \frac{Y+1}{\tilde{n_Y}}$, then a $100(1 - \alpha) \%$ confidence interval for the difference $p_X - p_Y$ is given as:

$$\tilde{p_X} - \tilde{p_Y} \pm z _{\alpha/2} \sqrt{\frac{\tilde{p_X}(1- \tilde{p_X})}{n_X} + \frac{\tilde{p_Y}(1- \tilde{p_Y})}{n_Y}}$$

If the lower and upper limit go beyond -1 and 1, they should be replaced with -1 and 1 respectively. The adjustments used is similar as the one described for the confidence interval of single proportion, instead they are equally divided amongst the two samples.

If $n_X$ and $n_y$ are <b>large</b> enough, the confidence interval can be given as:

$$\widehat{p_X} - \widehat{p_Y} \pm z _{\alpha/2} \sqrt{\frac{\widehat{p_X}(1- \widehat{p_X})}{n_X} + \frac{\widehat{p_Y}(1- \widehat{p_Y})}{n_Y}}$$

where $\widehat{p_X}$ and $\widehat{p_Y}$ are the probability of successes. This method can only be used if both the samples contain at least 10 successes and 10 failures.

</br>
#### Confidence Intervals for the Difference Between Two Means (Small-Sample) :

The confidence interval for the difference between two means for small sample sizes can be computed using the <b>Student's t distribution</b>. As the sample size is small, the central limit theorem can not be applied and given that both the population is <b>normal</b>, the Student's t distribution can be used to compute the confidence interval.

Let $X_1, X_2, ..., X _{n_X}$ and $Y_1, Y_2, ..., Y _{n_Y}$ be two independent random samples of size $n_X$ and $n_Y$ from a <b>normal</b> population with mean $\mu_X$ and $\mu_Y$ respectively, then a $100(1-\alpha) \%$ confidence interval for $\mu_X - \mu_Y$ is given as:

$$\overline{X} - \overline{Y} \pm t _{v, \alpha/2} \sqrt{\frac{s_X^2}{n_X} + \frac{s_Y^2}{n_Y}}$$

where the population standard deviations are estimated from sample standard deviations as $s_X$ and $s_Y$ respectively and the number of <b>degree of freedoms</b> $v$ of the $t$-distribution is given as:

$$v = \frac{\big( \frac{s_X^2}{n_X} + \frac{s_Y^2}{n_Y}\big)^2}{\frac{(\frac{s_X^2}{n_X})^2}{n_X-1} + \frac{(\frac{s_Y^2}{n_Y})^2}{n_Y-1}}$$

When the <b>two population have equal variances</b>, we can use the pooled sample variances as the estimate of the population variance. The confidence interval in this case is given as:

$$\overline{X} - \overline{Y} \pm t _{n_X+n_Y-2, \alpha/2} \times s_p \sqrt{\frac{1}{n_X} + \frac{1}{n_Y}}$$

where $s_p$ is the <b>pooled standard deviation</b> and is given as:

$$s_p = \sqrt{\frac{(n_X-1)s_X^2 + (n_Y-1)s_Y^2}{n_X + n_Y - 2}}$$

One common miss-conception while using this method is the assumption that if the sample variances are equal, the population variances are equal. For a smaller sample size, sample variance does not serve as a good approximation for the population variance and hence this assumption will not be true.


</br>
#### Reference :

https://www.mheducation.com/highered/product/statistics-engineers-scientists-navidi/M0073401331.html
