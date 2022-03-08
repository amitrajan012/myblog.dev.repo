+++
date = "2018-05-28T11:12:18+01:00"
description = "ISLR Moving Beyond Linearity"
draft = false
tags = ["ISLR", "Resampling", "Moving Beyond Linearity", "Regression Splines"]
title = "ISLR Chapter 7: Moving Beyond Linearity (Part 2: Regression Splines)"
topics = ["ISLR"]

+++


### 7.4 Regression Splines

Regression splines are flixible class of basis functions that extend upon polynomial and piecewise constant regression approaches.

#### 7.4.1 Piecewise Polynomials

<b>Piecewise polynomial regression</b> fits separate low-degree polynomials over different regions of $X$. For example, a piecewise squared polynomial fits squared regression model of the form

$$y_i = \beta_0 + \beta_1 x_i + \beta_2 x_i^2 + \epsilon_i$$

where the coefficients $\beta_0, \beta_1, \beta_2$ differs in different parts of the range of $X$. The points where the coefficients change are called <b>knots</b>. Each of the polynomial functions can be fit using least square methods. Increasing the number of knots will give a more flexible piecewise polynomial.

#### 7.4.2 Constraints and Splines

By using piecewise polynomial regression, the fitted curve on the data may have a <b>discontinuity at the knots</b> or we can say that the fitted curve is too flexible. Instead, we can fit a piecewise polynomial under the constraint that the fitted curve must be continuous. We can further add more constraints, such as, both the first and second derivatives of the piecewise polynomials must be continuous. <b>Each added constraint frees up one degree of freedom. and hence reducing the complexity of the resulting piecewise polynomial fit</b>. Hence by imposing three constraints of continuity, continuity of the first and second derivative, we reduce the degree of freedom of model by 3.

A piecewise cubic polynomial function with three constraints(continuity, continuity of the first and second derivative) is called as <b>cubic spline</b>. The degree of freedom of cubic spline is $K+4$, where $K$ is the <b>number of knots</b>. It can be explained as: The left(or right) end of the polynomial has a degree of freedom 4(as we have to estimate 4 coefficients or parameters to fit a cubic spline). Each additional knot adds one parameter (as three imposed constraints leave one free parameter) and hence making a total of $K+4$ parameters for $K$ knots. In general, a <b>degree-d spline</b> is a piecewise degree-d polynomial with continuity in derivatives upto degree $d-1$ at each knot.

#### 7.4.3 The Spline Basis Representation

A cubic spline with $K$ knots can be modeled as:

$$y_i = \beta_0 + \beta_1 b_1(x_i) + \beta_2 b_2(x_i) + ... + \beta _{K+3} b _{K+3}(x_i) + \epsilon_i$$

First of all, the equation can be interpreted as: the degree of freedom of a cubic spline is $K+4$ and hence we have to estimate $K+4$ parameters. After composing the equation, we need to formulate the <b>basis functions</b> $b_1, b_2, ..., b _{K+3}$. As explained above, a cubic spline can be iterpreted as a polynomial function where left(or right) end has a degree of freedom 4 (as we need to fit a cubic polynomial without any constraint) giving the first three basis functions as $x, x^2$ and $x^3$. Then we have to add one degree of freedom (parameter) per knot, with the constraints of continuity and continuity of the first and second derivatives. This behaviour can be captured by adding one <b>truncated power basis function</b> per knot, which is given as:

$$
\begin{equation*}
  h(x, \xi) = (x - \xi)^3 _+ = \left\{
  \begin{array}{@{}ll@{}}
    y(x - \xi)^3, & \text{if}\ x > \xi \\
    0, & \text{otherwise}
  \end{array}\right.
\end{equation*}
$$

where $\xi$ is the knot. Adding $\beta_ih(x, \xi)$ will lead to discontinuity only in the third derivative at $\xi$. Hence to fit a cubic spline to a data set with $K$ knots, we need to perform least squares regression to estimate an intercept and $3+K$ parameters for $X, X^2, h(X, \xi_1), h(X, \xi_2), ..., h(X, \xi_K)$, where $\xi_1, \xi_2, ..., \xi_K$ are the knots.

Cubic splines have higher variance at the ends. A <b>natural spline</b> adds additional <b>boundary constraints</b>(requirement of being linear at boundaries, reducing 2 degree of freedom at each boundary) and hence reduce the variance, producing more stable estimates at boundaries.

#### 7.4.4 Choosing the Number and Locations of the Knots

The regression spline is most flexible in the regions which have highest number of knots. One approach is to place higher number of knots in the regions where we feel that the function might vary the most. In practice, it is common to place knots in a uniform fashion. The number of knots can be decided by analyzing the curve visually or by cross-validation.

#### 7.4.5 Comparison to Polynomial Regression

Regression splines give better results as compared to polynomial regression. Regression splines increase the fliexibility of the model by increasing the number of knots. As we increase the number of knots, we can place more knots in the regions where the function $f$ seems to change rapidly and fewer knots in the regions where it is stable. In polynomial regression, to increase the flexibility, we need to increase the degree of the polynomial. It may result in unstability and overfitting.
