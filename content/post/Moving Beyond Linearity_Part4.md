+++
date = "2018-06-04T16:12:19+01:00"
description = "ISLR Moving Beyond Linearity"
draft = false
tags = ["ISLR", "Resampling", "Moving Beyond Linearity", "Local Regression", "Generalized Additive Models"]
title = "ISLR Chapter 7: Moving Beyond Linearity (Part 4: Local Regression, Generalized Additive Models)"
topics = ["ISLR"]

+++

### 7.6 Local Regression

<b>Local regression</b> comutes the fit at a target point $x_0$ using only the nearby training observstions. The algorithm for local regression is as follows:

 - Gather the $k$ points closest to $x_0$.
 - Assign a weight $K_{i0} = K(x_i, x_0)$ to all the points in the neighborhood such that the points that are farthest have lower weights. All the points except from these $k$ nearest neighbors have weigth 0.
 - Fit a <b>weighted least squares regression</b> of the aformentioned points using weights, by finding $\beta_0, \beta_1$ that minimize

 $$\sum _{i=1}^{n}K _{i0}(y_i - \beta_0 - \beta_1 x_i)^2$$


 - The fitted value at $x_0$ is given as $\widehat{f}(x_0) = \widehat{\beta_0} + \widehat{\beta_1} x_0$.

The <b>span s</b> of a local regression is defined as $s = \frac{k}{n}$, where $n$ is total number of training samples. It plays the role of controling the flexibility of the non-linera fit. Smaller the value of $s$, the more local or wiggly is the fit. For larger values of $s$, we obtain a global fit. An appropriate value of $s$ can be chosen by cross-validation.

### 7.7 Generalized Additive Models

<b>Generalized additive models (GAMs)</b> predict $Y$ on the basis of $p$ predictors $X_1, X_2, ..., X_p$. This can be viewed as an extension of multiple linear regression.

#### 7.7.1 GAMs for Regression Problems

Multiple linear regression model can be given as:

$$y_i = \beta_0 + \beta_1 x _{i1} + \beta_2 x _{i2} + ... + \beta_p x _{ip} + \epsilon_i$$

In order to incorporte a non-linear relationship between each feature and the response, each linear component can be replaced with a smooth non-linear function. The model can be expressed as:

$$y_i = \beta_0 + \sum _{j=1}^{p} f _{j}(x _{ij}) + \epsilon_i = \beta_0 + f_1(x _{i1}) + f_2(x _{i2}) + ... + f_p(x _{ip}) + \epsilon_i$$

This is an example of GAM. GAM is <b>additive</b> as we fit a separate non-linear model for each predictor and then add together their contributions. We can use any regression method to fit these individual models.

##### Pros and Cons of GAMs

 - As GAM models a non-linear relationship for each individual predictor, it will automatically capture the non-linear behaviour of the response.


 - As the model is additive, we can analyze the effect of each predictor on response by keeping other predictors constant.


 - The smoothness of each individual function can be summarized by its degree of freedom.


 - The main limitation of GAM is its additive nature. Due to this, the interaction between individual parameters is missed. However, we can manually add interaction terms (of the form $X_j \times X_k$) in GAM.

#### 7.7.2 GAMs for Classification Problems

For a qualitative variable $Y$, which takes on two values 0 and 1, the logistic regression model can be given as:

$$log\bigg( \frac{p(X)}{1-p(X)} \bigg) = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_p X_p$$

where $p(X) = Pr(Y=1 | X)$ and the left hand side of the equation is called as <b>logit</b> or log of the odds. To accomodate non-linearity, above model can be modified as:

$$log\bigg( \frac{p(X)}{1-p(X)} \bigg) = \beta_0 + f_0(X_1) + f_2(X_2) + ... + f_p(X_p)$$
