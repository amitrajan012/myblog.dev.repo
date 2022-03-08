+++
date = "2018-05-09T11:11:48+01:00"
description = "ISLR Linear Regression"
draft = false
tags = ["ISLR", "Linear Regression", "Other Considerations in the Regression Model"]
title = "ISLR Chapter 3: Linear Regression (Part 3: Other Considerations in the Regression Model)"
topics = ["ISLR"]

+++


### 3.3 Other Considerations in the Regression Model

#### 3.3.1 Qualitative Predictors

There can be a case when predictor variables can be <b>qualitative.</b>

#### Predictors with Only Two Levels

For the predictors with only two values, we can create an <b>indicator</b> or <b>dummy variable</b> with values 0 and 1 and use it in the regression model. The final prediction will not depend on the coding scheme. Only difference will be in the model coefficients and the way they are interpreted.

#### Qualitative Predictors with More than Two Levels

When a qualitative predictor has more than two levels, we can use more than one single dummy variable to encode them. There will always be one less dummy variable than the number of levels.

#### 3.3.2 Extensions of the Linear Model

Standard linear regression provides results that work quite well on real world problems. However, it makes two restrictive assumptions:

 - <b>Additive:</b> Relationship between response and predictor is additive, which means that the effect of change in the predictor $X_i$ on the response $Y$ is independent of the values of other predictors.


 - <b>Linear:</b> Change in response $Y$ due to one unit change in $X_j$ is constant.

#### Removing the Additive Assumption

A <b>synergy</b> or an <b>interaction</b> effect is described as the phenomenon when two predictors can interact while deciding on response. Linear model can be extended and take into account an <b>interaction term</b>($X_1X_2$) as follows:

$$Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + \beta_3X_1X_2 + \epsilon$$

There may be a case when interaction term has a very small p-value but the associated main effects do not. The <b>hierarchial principal</b> states that if we include the interaction term in the model, we should also include the main effect, even if the associated p-values are not significant.

Interaction effect of qualitative with quantitative variables can be incorporated in the same way.

#### Non-linear Relationships

<b>Polynomial Regression</b> can be used to extend the linear model to accomodate the non-linear relationship. The various regression models for miles per gallon vs horsepower for auto data is shown below. A simple way to incorporate non-linear associations in a linear model is by adding transformed versions of the predictors as follows (order 2):

$$mpg = \beta_0 + \beta_1 \times horsepower + \beta_2 \times horsepower^2 + \epsilon $$

This approach is called as <b>polynomial regression.</b>


```python
auto = pd.read_csv("data/Auto.csv")
auto.dropna(inplace=True)
auto = auto[auto['horsepower'] != '?']
auto['horsepower'] = auto['horsepower'].astype(int)

fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(111)
sns.regplot(x="horsepower", y="mpg", color='r', fit_reg=True, data=auto, order=1, scatter_kws={'alpha':0.1},
            line_kws={'color':'blue', 'alpha':0.7, 'label':'Order 1'})
sns.regplot(x="horsepower", y="mpg", color='r', fit_reg=True, data=auto, order=2, scatter_kws={'alpha':0.1},
            line_kws={'color':'g', 'alpha':0.7, 'label':'Order 2'})
sns.regplot(x="horsepower", y="mpg", color='r', fit_reg=True, data=auto, order=5, scatter_kws={'alpha':0.1},
            line_kws={'color':'orange', 'alpha':0.7, 'label':'Order 5'})

ax.set_xlabel('Horsepower')
ax.set_ylabel('Miles Per Gallon')
ax.set_title('Regression Model')
ax.set_ylim(8, 50)
ax.set_xlim(40, 250)
ax.legend()

plt.show()
```

    /Users/amitrajan/Desktop/PythonVirtualEnv/Python3_VirtualEnv/lib/python3.6/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval

{{% fluid_img "/img/Linear%20Regression_files/Linear%20Regression_39_1.png" %}}



#### 3.3.3 Potential Problems

The problems which arise when we fit a linear regression to a particular data set are as follows:

 - <b>Non-linearity of the Data:</b>

    <b>Residual plots</b> are a useful graphical tool for identifying non-linearity. For simple linear regression, a plot of residual vd predictor can be analyzed. In the case of multiple linear regression, as there are multiple predictors, a plot of residuals vs predicted values can be analyzed. Ideally, the residual plot will show no discernible pattern. The presence of a pattern may indicate a problem with some aspect of the linear model. If the residual plots indicate that there is a non-linear associations in the data, non-linear transformation of the predictors can be used in the model.


 - <b>Correlation of Error Terms:</b>

    An important assumption of linear regression model is that the error terms are uncorrelated. If there is a correlation between the error terms, the estimated standard errors will tend to underestimate the true standard errors. As a result, the confidence and prediction intervals will be narrower and the p-value associated with the model will be lower which results in an <b>unwarranted sense of confidence in the model.</b>d

    Correlation in error terms might occur in the context of <b>time series</b> data. This can be visualized by plotting the residuals against time and checking for a discernable pattern.


 - <b>Non-constant Variance of Error Terms:</b>

     Another assumption of the linear regression model is that the error terms have a constant variance. The non-constant variances in the errors can be identified by the presence of a <b>funnel shape</b> in the residual plot. When faced with this problem one possible approach is to transform the response $Y$ using a concave function $logY$ or $\sqrt Y$.


 - <b>Outliers:</b>

     An outlier may have a little effect on the least square fit but it can cause other problems like high value of RSE and lower R$^2$ values which can affect the interpretation of the model. <b>Residual plots</b> can be used to identify outliers.


 - <b>High Leverage Points:</b>

     High leverage points have an unsual values for $x_i$. Removing high leverage point has more substantial impact on the least square line compared to the outliers. Hence it is important to identify high leverage points. In a simple linear regression, it is easy to check on the range of the predictors and find the high levarage points. For a multiple linear regression, the predictors may lie in their individual ranges but can lie outside in terms of the full set of predictors. <b>Leverage statistic</b> is a way to identify the high leverage points. A large value of this statistic indicates high leverage.


 - <b>Collinearity:</b>

     When two or more predictor variables are closely related to each other, a situation of collinearity arises. Due to collinearity, it can be impossible to separate out the individual effects of collinear variables on the response. Collinearity also reduces the estimation accuracy of the regression coefficients. As <b>t-statistic</b> of a predictor is calculated by dividing $\beta_i$ by its standard error, and hence collinearity results in the decline of t-statistic and consequently we may <b>fail to reject the null hypothesis.</b>

     A simple approach is to detect collinearity is by analyzing the <b>correlation matrix</b>. This process has a drawback as it can not detect <b>multicollinearity</b> (collinearity between three or more variables). A better way to assess multicollinearity is by computing <b>variance inflation factor (VIF)</b>. VIF is the ratio of variance in a model with multiple predictors, divided by the variance of a model with one predictor alone. Smallest possible value of VIF is 1 and as a rule of thumb, a VIF value that exceeds 5 or 10 indicates a problematic amount of collinearity.

     There are two approaches to deal with the problem of collinearity. One is to simply drop one of the problematic variable. Alternatively we can combine the collinear variables together as a single predictor.

#### 3.5 Comparison of Linear Regression with K-Nearest Neighbors

Linear regression is an example of <b>parametric approach</b>, as it assumes a linear form for $f(X)$. It has the advantage of easy to fit as we only need to estimate a small number of coefficients. The one disadvantage of parametric method is that, they make a strong assumption about the shape of $f(X)$ and hence can affect the prediction accuracy if the shape deviates from the assumption.

<b>Non-parametric</b> methods does not assume a parametric form for $f(X)$ and hence provide a more flexible approach for regression. <b>KNN (K-nearest neighbors) regression</b> is an example of this.

KNN regression first identifies $K$ nearest training observations to $x_0$, represented as $N_0$. It then estimates $f(x_0)$ as the average of the training responses in $N_0$ as:

$$\widehat{f}(x_0) = \frac{1}{K} \sum _{x_i \in N_0} y_i$$

As the value of $K$ increases, the smothness of fit increases. The optimal vale of $K$ depends on the <b>bias-variance tradeoff</b>. A small value of $K$ provides the most flexible fit, which will have low bias but high variance. On contrast larger value of $K$ provides a smoother and less variable fit (as prediction depends on more points, changing one will have smaller effect on the overall prediction).

<b>The parametric approach will outperform the nonparametric approach if the parametric form that has been selected is close to the true form of $f$.</b> In this case, the non-parametric approach incurs a cost in variance that is not offset by a reduction in bias. As level of non-linearlity increases, for $p=1$, KNN regression outperforms linear regression. But as number of predictors increases, the performance of linear regression is better than KNN. This arises due to the phenomenon which can be termed as <b>curse of dimensionality</b>. As the number of predictors increase, number of dimensions increases and hence the given test observation $x_0$ may be very far away in the p-dimensional space when p is large and hence a poor KNN fit. <b>As a general rule, parametric methods will tend to outperform non-parametric approaches when there is a small number of observations per predictor.</b> Even when the dimension is small, linear regression is preferred due to better interpretability.
