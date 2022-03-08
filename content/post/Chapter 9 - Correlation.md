+++
date = "2018-09-10T11:13:31+01:00"
description = "Think Stats: Chapter 9"
draft = false
tags = ["Think Stats", "Standard scores", "Covariance", "Correlation", "Spearman’s rank correlation", "Least squares fit", "Goodness of fit"]
title = "Think Stats: Chapter 9"
topics = ["Think Stats"]

+++





### 9.1 Standard scores

The main challenge in measuring correlation is that the variables we want to compare might not be expressed in the same units. There are two common solutions to this problem:

 - Transform all values to <b>standard scores</b>. This leads to <b>Pearson coefficient of correlation</b>.
 - Transform all values to their <b>percentile ranks</b>. This leads to <b>Spearman coefficient</b>.

<b>Normalizing</b> the score means subtracting mean from every value and dividing it by standard deviation.

$$z_i = \frac{(x_i - \mu)}{\sigma}$$

If X is skewed, distribution Z will be skewed as well. In that case it is more apt to use percentile ranks as they will always be uniformly distributed between 0 and 100.

### 9.2 Covariance

<b>Covariance</b> is a measure of the tendeny of two variables to vary together. For two series X and Y, the covariance is given as:

$$Cov(X, Y) = \frac{1}{n}\sum (x_i - \mu_X)(y_i - \mu_Y)$$

where n is the length of the two series (they have to be of same length). When X and Y are same, Cov(X, X) = Var(X).

### 9.3 Correlation

When Covariance is divided by standard deviations, we get a more standard notion for the measure of the tendency of two variables to vary together.

$$\rho = \frac{1}{n} \sum \frac{(x_i - \mu_X)(y_i - \mu_Y)}{\sigma_X \sigma_Y} = \frac{Cov(X, Y)}{\sigma_X \sigma_Y}$$

This value is called <b>Pearson's Correlation</b> coefficient. Pearson'e correlation coefficient is always between -1 and 1 inclusive. For $\rho = 1$, values are perfectly correlated and for $\rho = -1$, values are negatively correlated but for the purpose of prediction, negative correlation is just as good as positive one. Pearson'e correlation only measures <b>linear</b> relationship. If there is non-linear relationship, $\rho$ understates the strength of the dependence and hence one should always look at the scatterplot of the data before computing the correlation coefficient.

### 9.5 Spearman’s rank correlation

<b>Pearson’s correlation</b> works well if the relationship between variables is linear and if the variables are roughly normal. But it is not robust in the presence of outliers. <b>Spearman's rank correlation</b> is an alternative that mitigates the effect of outliers and skewed distributions. To compute Spearman's correlation, we need to compute the rank of each variable, which is the index of the sorted sample and then we compute Pearson's correlation of the ranks.

An alternate of the Spearman's correleation is to apply a transform tha makes the data more nearly normal and then compute Pearson's coefficient. For example if the data is lognormal, we can take the log making the data normal and then compute the correlation of log values.

### 9.6 Least squares fit

Correlation coefficient measures the strength and sign of a relationship. They do not measure the slope. The most common way to estimate the slope is a <b>linear least squares fit</b>. A linear least squares fit is a line that models the relationship between variables and minimizes the mean squared error(MSE) between the line and the data.

Suppose we have a sequence of points Y that we want to express as a function of another sequence X. Let this relationship is expressed such that the prediction of $y_i$ is $\alpha + \beta x_i$. Then the <b>deviation</b> or <b>residual</b> is:

$$\epsilon_i = (\alpha + \beta x_i) - y_i$$

and then we can minimize the sum of squared residuals:

$$min(\alpha, \beta) \sum \epsilon_i^2$$

There are certain advantage of minimizing the squared residuals, such as:

 - Squaring treats negative and positive residual the same.
 - It gives more weight to large residuals.
 - The value of $\alpha$ and $\beta$ can be computed efficiently.

The least square fit can be given as:

$$\beta = \frac{Cov(X, Y)}{Var(X)}, \ \ \ \ \alpha = \bar{y} - \beta \bar{x} $$

where $\bar{y}$ and $\bar{x}$ are sample means.

### 9.7 Goodness of fit

In the context of prediction, the quantity we are trying to guess is called as the <b>dependent variable</b> and the one used to make the guess is called the <b>explanatory</b> or <b>independent variable</b>. The <b>predictive power</b> of a model can be measured by <b>coefficient if determination</b> (also called as <b>R-squared</b>), which is given as:

$$R^2 = 1 - \frac{Var(\epsilon)}{Var(Y)}$$

A plausible explanation of $R^2$ is that suppose we have to guess Y and we don't have any information, our best guess will be the mean $\bar{y}$ ane hence the MSE will be:

$$MSE = \frac{1}{n} \sum{\bar{y} - y_i}^2 = Var(Y)$$

But when we know about the relation between X and Y, our guess will be $\alpha + \beta x_i$. In this case the MSE will be ($\epsilon$ has a normal distribution with $\mu = 0$):

$$MSE = \frac{1}{n} \sum{\alpha + \beta x_i - y_i}^2 = Var(\epsilon)$$

Hence the term, Var($\epsilon$)/Var(Y) is the ratio of MSE with and without the explanatory variable, which is the fraction of variability which is unexplained by the model and hence it's complement is the fraction of variability explained.

In a <b>linear least square model</b>, <b>coefficient of determination</b> and <b>Pearson's correlation coefficient</b> $\rho$ is related as:

$$R^2 = \rho^2$$

### 9.8 Correlation and Causation

Correlation does not imply causation.
