+++
date = "2018-05-18T02:26:28+01:00"
description = "ISLR Resampling Methods"
draft = false
tags = ["ISLR", "Resampling", "The Bootstrap"]
title = "ISLR Chapter 5: Resampling Methods (Part 2: The Bootstrap)"
topics = ["ISLR"]

+++


### 5.2 The Bootstrap

<b>Bootstrap</b> can be used to to quantify the uncertainty associated with a given statistical model. For example, bootstrap can be used to estimate <b>standard errors (which measures the uncertainty)</b> of the coefficients from a linear regression fit. Bootstrap can be applied to a wide range of statistical learning methods. The method of bootstrap is explained below via an example:

Suppose we wish to invest money in two financial assests which yield returns of $X$ and $Y$. We will invest fraction $\alpha$ of our money in $X$ and $1-\alpha$ in $Y$. As there is a variablility associated with these returns, we wish to choose $\alpha$ which minimizes the total risk or variance of our investment. i.e. <b>We want to minimize $Var(\alpha X + (1-\alpha) Y)$</b>. The value that minimizes this is given as:

$$\alpha = \frac{\sigma_Y^2 - \sigma _{XY}}{\sigma_X^2 + \sigma_Y^2 - 2\sigma _{XY}}$$

where $\sigma_X^2 = Var(X)$, $\sigma_Y^2 = Var(Y)$ and $\sigma _{XY} = Cov(X, Y)$. As these quantities are unknown, we can make estimates for them using the past measurements of $X$ and $Y$ and hence the estimate of $\alpha$ can be computed.

One way to quantify the accuracy of the estimate of $\alpha$ is to repetedly draw 100 pairs of observations $(X, Y)$ 1000 times and hence estimating $\alpha$ 1000 times. Suppose for the simulations, the parameters were set to $\sigma_X^2 = 1$, $\sigma_Y^2 = 1.25$ and $\sigma _{XY} = 0.5$, then the true value of $\alpha$ is 0.6. If the estimated vlaue of $\alpha$ (from the simulation) turns out to be <b>0.5996</b> with standard deviation of <b>0.083</b>, we may say that the estimated value of $\alpha$ (which is $\widehat{\alpha}$) differs from the true value by 0.08 on an average.

In practice, for real data, we can not generate new samples from the original population and hence the above explained procedure is not feasible. <b>In bootstrap method, instead of obtaining independent data sets from the population, we obtain distinct data set by repeatedly sampling observations from the original data set.</b> For a sample with $n$ observations, each bootstrap data set contains $n$ observations <b>(sampled with replacement)</b> from the original dataset. A total of $B$ bootstrap data sets labeled as $Z^{*1}$, $Z^{*2}$, ..., $Z^{*B}$ are generated and corresponding B estimates of $\alpha$ ($\widehat{\alpha}^{*1}$, $\widehat{\alpha}^{*2}$, ..., $\widehat{\alpha}^{*B}$) are obtained. The standard error of these bootstrap estimates can be obtained by:

$$SE _{B}(\widehat{\alpha}) = \sqrt{\frac{1}{B-1} \sum _{r=1}^{B} \bigg( \widehat{\alpha}^{*r} - \frac{1}{B} \sum _{r^{'}=1}^{B} \widehat{\alpha}^{*r^{'}} \bigg ) ^2}$$

This way, a bootstrap process can be used very effictevly to estimate the variability associated with an estimated parameter.
