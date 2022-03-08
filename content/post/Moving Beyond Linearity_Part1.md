+++
date = "2018-05-28T04:22:38+01:00"
description = "ISLR Moving Beyond Linearity"
draft = false
tags = ["ISLR", "Resampling", "Moving Beyond Linearity", "Polynomial Regression", "Step Functions", "Basis Functions"]
title = "ISLR Chapter 7: Moving Beyond Linearity (Part 1: Polynomial Regression, Step Functions, Basis Functions)"
topics = ["ISLR"]

+++


<h1><center>Moving Beyond Linearity</center></h1>

Lineaer models have its limitations in terms of predictive power. Linear models can be extended simply as:

 - <b>Polynomial regression</b> extends linear regression by adding extra higher order predictors (predictors rasied to higher order powers).


 - <b>Step functions</b> cut the range of a variable into $K$ distinct regions in order to produce a qualitative variable.


 - <b>Regression splines</b> is the extension of polynomial regression and step functions. It divides the range of predictor $X$ into $K$ distinct regions and within each region a polynomial function is fit to the data.


 - <b>Smoothing splines</b>


 - <b>Local regression</b>


 - <b>Generalized additive models</b>


### 7.1 Polynomial Regression

A standard linear regression model

$$y_i = \beta_0 + \beta_1 x_i + \epsilon_i$$

can be replaced by a more generic polynomial function

$$y_i = \beta_0 + \beta_1 x_i + \beta_2 x_i^2 + \beta_3 x_i^3 + ... + \beta_d x_i^d + \epsilon_i$$

This approach is known as <b>polynomial regression</b> and for large enough values of $d$, it can produce a highly non-linear curve. It is highly unusual to use $d$ greater than 3 or 4. The given model parameters can easily be estimated using linear least squares linear regression procedure. Similarly, polynomial functions can be modeled with the <b>logistic regression</b> as well.

### 7.2 Step Functions

Polynomial regression gives a fit that is more <b>global</b> in nature. In <b>step functions</b>, we divide the range of $X$ into <b>bins</b> and fit a different constant in each bin. We can create $K$ <b>cutpoints</b> $c_1, c_2, ..., c_K$ in the range of $X$, and then can construct $K+1$ new <b>categorical</b> variables as:

$$C _i(X) = I(c_i \leq X < c _{i+1})$$

where $I(.)$ is an <b>indicator function</b> which returns 1 if the condition is true and 0 oterwise. For any value of $X$, $C_0(X) + C_1(X) + ... + C_K(X) = 1$, as only one value will be 1 for each $X$. We can then fit a linear least squares model to fit $C_1(X), C_2(X),...,C_K(X)$ as predictors. We need to omit one predictor as there will be intarcept too. The linear model is given as:

$$y_i = \beta_0 + \beta_1 C_1(x_i) + \beta_2 C_2(x_i) + ... + \beta_K C_K(x_i) + \epsilon_i$$

$\beta_0$ is a response for $X<c_1$. The response for $c_j \leq X < c _{j+1}$ is $\beta_0 + \beta_j$. Hence, $\beta_j$ represents the average increase in the response for $X$ in $c_j \leq X < c _{j+1}$ relative to $X < c_1$. Logistic regression model can be fitted in the same way.

### 7.3 Basis Functions

Polynomial and piecewise-constant regression models are special cases of a <b>basis function</b> approach for regression. In basis function approach, we use a family of functions to transform $X$ and instead of fitting a linear model in $X$, we fit the transformed predictors as:

$$y_i = \beta_0 + \beta_1 b_1(x_i) + \beta_2 b_2(x_i) + ... + \beta_K b_K(x_i) + \epsilon_i$$

The basis functions are fixed and known. For polynomial regression, the basis functions are $b_j(x_i) = x_i^j$. For piecewise constant functions, they are $b_j(x_i) = I(c_j \leq x_i < c _{j+1})$. As in basis functions approach linear model is fitted on the transformed variables, all the inference tools for linear models can be used.
