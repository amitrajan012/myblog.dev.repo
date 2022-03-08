+++
date = "2018-05-21T13:08:40+01:00"
description = "ISLR Linear Model Selection and Regularization"
draft = false
tags = ["ISLR", "Resampling", "Linear Model Selection", "Regularization", "Subset Selection"]
title = "ISLR Chapter 6: Linear Model Selection and Regularization (Part 1: Subset Selection)"
topics = ["ISLR"]

+++


<h1><center>Linear Model Selection and Regularization</center></h1>

Befor moving to non-linear models, there are certain other fitting procedures through which a plain linear model can be improved. These alternate fitting procedures can yield better <b>prediction accuracy</b> and <b>model interpretability</b>.

 - <b>Prediction Accuracy:</b> Provided that the relationship between predictors and response is linear, the <b>least square estimates will have low bias</b>. If n >> p, this model will have low variance as well. If n is not much larger than p, the model will have a lot of variability, resulting in overfitting and poor prediction accuracy. If p > n, the variance is almost infinite and hence the plain least square method can not be used at all.


 - <b>Model Interpretability:</b> There may be the case that some of the predictors used in the linear model are not significant. Including such <b>irrelevant variables</b> may lead to unnecessary complexity in the model. These variables can be excluded from the model by <b>feature selection</b> or <b>variable selection</b> and hence a better model interpretability.

Some methods that can be used to improve least square fit are:

 - <b>Subset Selection:</b> The predictors that are statistically significant are identified and the model using least squares is then fit on these reduced set of variables.


 - <b>Shrinkage:</b> The model is fit using all $p$ predictors and later on the estimated coefficients are shrunken towards 0 relative to the least square estimates. This shrinkage (also called as <b>regularization</b>) can reduce the variance and hence results in better fit.


 - <b>Dimension Reduction:</b> This method can be used to project the $p$ predictors into a $M$-dimensional space, where $M < p$. These $M$ predictors are then used to fit a least squares linear regression model.

### 6.1 Subset Selection

#### 6.1.1 Best Subset Selection

In <b>best subset selection</b>, a seperate least squares regression is fitted for each possible combination of $p$ predictors. We then identify the best model out of these. The algorithm for this is as follows:


 - Let $M_0$ denotes the <b>null model</b>, which contains no predictors. This model simply predicts the sample means of each observation.

 - For $K=1, 2, ..., p$, all $\dbinom{p}{k}$ models are fitted and the best among them is denoted as $M_k$. The bset is the one which has the <b>smallest RSS</b> or <b>largest R$^2$</b>.

 - Single best model out of $M_0, M_1, ..., M_k$ is selected using <b>cross-validation, AIC, BIC</b> or <b>adjusted R$^2$</b>.


After first two steps, the problem is reduced from $2^p$ possible models to $p+1$ possible models. Selecting one of the best models out of these $p+1$ models can be misleading. As number of predictors in the model increases, the <b>RSS monotonically decreases</b> and <b>R$^2$ monotonically increases</b> (as they depend on training error and training error improves as we keep on adding new predictors). Hence, we need to choose a model based on <b>test error</b>. This is the reason <b>cross-validation, $C_p$, BIC</b> or <b>adjusted R$^2$</b> is used in this step.

In the best subset selection for logistic regression, insted of RSS or R$^2$, we use the <b>deviance</b>. Deviance is negative two times the maximized log-likelihood. The <b>smaller the deviance, the better the fit</b>. For large number of predictors, best subset selection is <b>computationally inefficient</b>.

#### 6.1.2 Stepwise Selection
##### Forward Stepwise Selection

<b>Forward Stepwise Selection</b> begins with no predictors and then add them to the model one-at-a-time, until all the predictors are in the model. At each step, the variable that gives the greatest <b>additional</b> improvement to the fit is added to the model. The algorithm is as follows:

 - Let $M_0$ denotes the null model.
 - Add a single predictor to the model which gives the <b>smallest RSS</b> or <b>largest R^$2$</b> for it. The model is denoted as $M_1$.
 - Gradually get the models $M_0, M_1, M_2, ... , M_p$ by following the same procedure.
 - Select the single best model out of these based on <b>cross-validation, C$_p$, BIC</b> or <b>adjusted R$^2$</b>.


For $p$ predictors, the forward stepwise selection requires fitting of $1+\frac{p(p+1)}{2}$ models. This is a substantial improvemenet over best subset selection. Though forward stepwise selection do well in practice, it does not guarantee the selection of best possible model out of all the possible $2^p$ models as all of them are not analyzed in this case.

Forward stepwise selection can be applied in the case when $n<p$ as well. But in this case the constructed sub-models range from $M_0$ to $M _{n-1}$, as least square method does not give a unique solution if $p \geq n$.

##### Backward Stepwise Selection

<b>Backward Stepwise Selection</b> begins with the model containing all the $p$ predictors and it iteratively removes the least useful predictor one at a time. The algorithms is similar to the ones descibed above. Like forward stepwise selection, we need to fit only $1+\frac{p(p+1)}{2}$ models in backward stepwise selection. It does not guarantee to yield the best model as well. As in this case, we start from fitting the model with all the $p$ predictors, it is needed that: $n > p$.

##### Hybrid Approaches

In <b>Hybrid Approaches</b>, forward and backward stepwise selection are used to mimic best subset selection. In this method while doing forward stepwise selection, the least significant predictor (the variable that no longer provide any improvement in the model fit) is removed as well.

#### 6.1.3 Choosing the Optimal Model

In the last step of the algorithms described above, we need to choose the best model out of possible $(p+1)$ models. If we do this selection based on the parameters which are derived on the basis of training error (RSS or R$^2$), we will always end up selecting the model which has all the parameters. For the selection of the best model on the basis of <b>test error</b>, test error can be estimated as:

 - By doing <b>adjustment</b> to the training error.
 - can be directly estimated using a cross-validation approach.

##### $C_p$, AIC, BIC, and Adjusted R$^2$

In general, <b>training MSE</b> (MSE = $\frac{RSS}{n}$) is an underestimate of test MSE and hence the training RSS and R$^2$ can not be used to select the best model out of all possible ones. There are several techniques that can be used for <b>adjusting</b> the training error to get an estimate of test error. Some of these are : <b>$C_p$, Akaike information criterion (AIC), Bayesian information criterion (BIC)</b> and <b>Adjusted R$^2$</b>.

$C_p$ estimate of the test MSE is given as:

$$C_p = \frac{1}{n}(RSS + 2d\widehat{\sigma}^2)$$

where $d$ is the number of predictors and $\widehat{\sigma}^2$ is the estimate of the variance of error $\epsilon$ associated with each response in the linear model given as:

$$Y = \beta_0 X_1 + \beta_2 X_2 + ... + \beta_p X_p + \epsilon$$

The penality added is $2d\widehat{\sigma}^2$, which increases as the number of predictors increases. <b>If $\widehat{\sigma}^2$ is an unbiased estimator of $\sigma^2$, then $C_p$ is an unbiased estimate of test MSE.</b> Finally,  model with the lowest value of $C_p$ is chosen.

The <b>AIC</b> criterion is used for the models fit by maximum likelihood. AIC is given as:

$$C_p = \frac{1}{n\widehat{\sigma}^2}(RSS + 2d\widehat{\sigma}^2)$$

where the terms has the same meaning as explained above. $C_p$ and AIC are proportional to each other.

<b>BIC</b> is derived from a Bayesian point of view. For a least square model, BIC is given as:

$$BIC = \frac{1}{n}(RSS + log(n)d\widehat{\sigma}^2)$$

As instead of 2, BIC has a factor of $log(n)$, for values of $n>7$, BIC ends up penalizing models with higher predictors more and hence results in a selection of smaller models than $C_p$.

The <b>Adjusted R$^2$</b> statistic is another measure for model selection. $R^2$ is defined as $1 - \frac{RSS}{TSS}$, where RSS is <b>Residual sum of squares</b> and TSS is <b>Total sum of squares</b>. TSS is given as $\sum(y_i - \bar{y})^2$. For a least squares model with $d$ predictors, Adjusted R$^2$ statistic is given as:

$$Adjusted \ R^2 = 1 - \frac{RSS/(n-d-1)}{TSS/(n-1)}$$

A <b>large value of adjusted R$^2$ indicates smaller test error</b>. The concept behind adjusted R$^2$ is the fact that once the model is saturated (in terms of predictor), adding more predictors will only decrease RSS by a lower value but as the denominator $(n-d-1)$ decreases by a larger proportion, the value $\frac{RSS}{n-d-1}$ will overall increase, decreasing the value of adjusted R$^2$.

Here the expression for AIC, BIC and $C_p$ is given for linear model fit with least squares. The general expression for them can be computed as well.

##### Validation and Cross-Validation

Instead of computing the above parameters, we can directly estimate the test error by cross-validation methods. This process has an advantage comared to the methods discussed above as it provides a direct estimate of the test error and makes fewer assumption about the underlying model. It can also be used in the cases when it is hard to identify the number of predictors in the model or estimation of error variance $\sigma^2$ is difficult.

<b>As the estimate of test MSE depends on the validation set or the method of cross-validation(the estimate will be different fro 5-fold and 10-fold cross-validation), for the models with slight variance in the estimated test MSE across the variation in the number of predictors, there is a chance that if the validation process is repeated with different split of data, the model selection will change.</b> For this reason, we follow <b>one-standard-error rule</b>. The rule suggests that the lowest point in the curve for the estimated test MSE is found, and eventually, the smallest model (model with smallest number of predictors) for which the estimated test error is within one standard error away from the lowest point is selected as the best model.
