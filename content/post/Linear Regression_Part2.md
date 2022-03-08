+++
date = "2018-05-08T06:12:03+01:00"
description = "ISLR Linear Regression"
draft = false
tags = ["ISLR", "Linear Regression", "Multiple Linear Regression"]
title = "ISLR Chapter 3: Linear Regression (Part 2: Multiple Linear Regression)"
topics = ["ISLR"]

+++



### 3.2 Multiple Linear Regression

In general, suppose we have $p$ distinct predictors, the multiple linear regression takes the form:

$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_p X_p + \epsilon$$

where $\beta_j$ can be interpreted as the average effect on $Y$ of a one unit increase in $X_j$, <b>holding all other predictors fixed.</b>

#### 3.2.1 Estimating the Regression Coefficients

Given the estimates, $\widehat{\beta_0}, \widehat{\beta_1},..., \widehat{\beta_p}$, predictions can be done as:

$$\widehat{y} = \widehat{\beta_0} + \widehat{\beta_1}x_1 + \widehat{\beta_2}x_2 + ... + \widehat{\beta_p}x_p$$

$\beta$s can be estimated by minimizing the sum of squared residuals:

$$RSS = \sum _{i=1}^{n}(y_i - \widehat{y_i})^2 = \sum _{i=1}^{n}(y_i - \widehat{\beta_0} + \widehat{\beta_1}x _{i1} + \widehat{\beta_2}x _{i2} + ... + \widehat{\beta_p}x _{ip})^2$$

The model coefficients for multiple linear regression for advertisement data is calculated below. It is observed that the coefficient for newspaper is almost equal to 0. As we analyze the correlation table for the data, the correlation coefficient between radio and newspaper is 0.35 and hence the rise in sales due to newspaper may arise due to the radio advertising, though if we fit a model for sales and newspaper the coefficient will not be 0.


```python
from sklearn import linear_model

reg = linear_model.LinearRegression(fit_intercept=True)
reg.fit (adv[['TV', 'Radio', 'Newspaper']], adv['Sales'])
print("Model Coefficients: " + str(reg.coef_))
print("Intercept: " + str(reg.intercept_))
print("R-statistics: " + str(reg.score(adv[['TV', 'Radio', 'Newspaper']], adv['Sales'])))
```

    Model Coefficients: [ 0.04576465  0.18853002 -0.00103749]
    Intercept: 2.9388893694594085
    R-statistics: 0.8972106381789521


    /Users/amitrajan/Desktop/PythonVirtualEnv/Python3_VirtualEnv/lib/python3.6/site-packages/sklearn/linear_model/base.py:509: RuntimeWarning: internal gelsd driver lwork query error, required iwork dimension not returned. This is likely the result of LAPACK bug 0038, fixed in LAPACK 3.2.2 (released July 21, 2010). Falling back to 'gelss' driver.
      linalg.lstsq(X, y)



```python
print("Correlation Coefficients:")
adv[['TV', 'Radio', 'Newspaper', 'Sales']].corr()
```

    Correlation Coefficients:





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TV</th>
      <th>Radio</th>
      <th>Newspaper</th>
      <th>Sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>TV</th>
      <td>1.000000</td>
      <td>0.054809</td>
      <td>0.056648</td>
      <td>0.782224</td>
    </tr>
    <tr>
      <th>Radio</th>
      <td>0.054809</td>
      <td>1.000000</td>
      <td>0.354104</td>
      <td>0.576223</td>
    </tr>
    <tr>
      <th>Newspaper</th>
      <td>0.056648</td>
      <td>0.354104</td>
      <td>1.000000</td>
      <td>0.228299</td>
    </tr>
    <tr>
      <th>Sales</th>
      <td>0.782224</td>
      <td>0.576223</td>
      <td>0.228299</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



#### Is There a Relationship Between the Response and Predictors?

To check whether there is a relationship between response and predictors, we need to check whether $\beta_1 = \beta_2 = ... = \beta_p = 0$. The hypothesis is as follows:

 - $H_0$: $\beta_1 = \beta_2 = ... = \beta_p = 0$
 - $H_A$: at least one of $\beta_j$ is non-zero.

The hypothesis test is performed by computing the <b>F-statistic</b>:

$$F = \frac{(TSS-RSS) \ / \ p}{RSS \ / \ (n-p-1)}$$

where TSS is <b>total sum of squares</b> and RSS is <b>residual sum of squares</b>. If linear model assumption is true the denominator equals $\sigma^2$. If the <b>null hypothesis is true</b>, the numerator equals $\sigma^2$ as well and hence value of <b>F-statistic equals 1</b>. If the alternate hypothesis is true <b>F is greater than 1.</b> How large does the F-statistic have to be to reject $H_0$? For large value of $n$, F-statistic that is little larger than $n$ provides the evidence against $H_0$. A large F-statistic is needed to reject $H_0$ if n is small. When $H_0$ is true and the errors $\epsilon_i$ have a normal distribution, F-statistics follows an <b>F-distribution</b> and hence <b>p-value</b> can be calculated from that.

Sometimes we want to test that whether a particular set of predictors have a relationship with response. This corresponds to null hypothesis:

 - $H_0$: $\beta _{p-q+1} = \beta _{p-q+2} = ... = \beta _{p} = 0$

In this case we can fit a model that uses all the variables except these last $q$. Suppose the residual sum of squares for this model is $RSS_0$, then the F-statistic is defined as:

$$F = \frac{(RSS_0 - RSS)\ / \ q }{RSS\ /\ (n-p-1)}$$

For <b>very large number of variables</b> (p > n), we can not fit the multiple linear regression and F-statistic can not be used as well.

#### Deciding on Important Variables

The task of determining which predictors are associated with the response, in order to fit a single model involving only those predictors, is referred to as <b>variable selection.</b> One approach is to use all the possible combinations of predictors, build the model and select the one which fits best. But for large value of p, this approach is not feasible. There are three calssic approaches for this:

 - <b>Forward Selection:</b> We begin with a <b>null model</b> (only with intercept) and fit $p$ linear regressions and add to the null model the variable that results in the <b>lowest RSS</b>. Then we can add to the model (with one variable) the variable that contributes lowest RSS for the new two-variable model. This process is continued until some stopping criteria is satisfied.


 - <b>Backward Selection:</b> This works in the same way but in the reverse order. We start with a p-variable model and remove the variable with <b>largest p-value</b> resulting in a (p-1)-variable model. We can continue further until all remaining variables have a p-value below some threshold.


 - <b>Mixed Selection:</b> This is a combination of forward and backward selection.

Backward selection can not be used if $p > n$, but forward selection can always be used.

#### Model Fit

Two most common numerical measure that can be used to describe the model fit are <b>RSE</b> and <b>R</b>^2. These quantities can be interpreted in the same way for multiple linear regression with one difference. For simple linear regression, $R^2$ is equal to the square of the correlation between response and the predictor. In the case of multiple linear regression, it equals $Cor(Y, \widehat{Y})^2$ instead. <b>$R^2$ will always increase if more variables are added to the model, even if those variables are weakly associated with the response.</b> Analysing the plot of the data is also a good way to check the model fit.

#### Predictions

We can use the least square plane to make the prediction for the response variable but there are some uncertainty associated with the prediction:

 - The model coefficients are only an estimation of the true population regression plane.

 - In practice, assuming a linear model for $f(X)$ is an approximation and hence there is an additional reducible error which was called as <b>model bias.</b>

 - Even if $f(X)$ is known, there are some irreducible errors due to $\epsilon$, which can not be predicted.
