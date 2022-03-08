+++
date = "2018-05-30T06:02:06+01:00"
description = "ISLR Moving Beyond Linearity"
draft = false
tags = ["ISLR", "Resampling", "Moving Beyond Linearity", "Smoothing Splines"]
title = "ISLR Chapter 7: Moving Beyond Linearity (Part 3: Smoothing Splines)"
topics = ["ISLR"]

+++

### 7.5 Smoothing Splines

#### 7.5.1 An Overview of Smoothing Splines

Regression splines are created by specifying a set of knots, producing a sequence of basis functions and then estimate spline coefficients using least squares.

To fit a smooth curve to a data set, we need to find a function $g(x)$ such that $RSS = \sum_{i=1}^{n}(y_i - g(x_i))^2$ is minimum. If we do not put any constraint on $g(x)$, we can always find a function $g(x)$, which will make RSS 0. This function will be too flexible and will overfit the data. Hence, we need to find a function $g$ which makes RSS small and which is <b>smooth</b> as well.

One way to find such a smooth function is to minimize:

$$\sum_{i=1}^{n}(y_i - g(x_i))^2 + \lambda \int g^{''}(t)^2 dt$$

where $\lambda$ is a nonnegative <b>tuning parameter</b>. The function that minimizes this is called as <b>smoothing spline</b>. The first part is a <b>loss function</b> and the second term is a <b>penalty</b> part that penalizes the variability of $g$. The second derivative of a function measures its smootheness as it corresponds to the amount by which the slope of a curve is changing. Hence, the second term encourages $g$ to be smooth. Larger the value of $\lambda$, smoother the $g$ as well. When $\lambda = 0$, the given model will be very flexible and will interpolate the training data. For $\lambda \to \infty$, the model corresponds to simple <b>least squares linear regression</b>. In a nut-shell, $\lambda$ <b>controls the bias-variance trade-off of the smoothing spline</b>.

The function $g$ that minimizes above quantity is the <b>natural cubic spline</b>. It is a piecewise cubic polynomial with knots having continuous first and second derivative at them. It should also be linear in the region outside the extreme knots. The obtained natural cubic spline is the <b>shrunken</b> version (due to tuning parameter $\lambda$) of the one which is obtaind by basis function approach.

#### 7.5.2 Choosing the Smoothing Parameter λ

The tuning parameter $\lambda$ controls the flexibility of the smoothing spline, and hence the <b>effective degree of freedom</b>. As $\lambda$ increases from 0 to $\infty$, the effective degree of freedom ($df_{\lambda}$) decreases from $n$ to 2.

Generally, degree of freedom refers to the number of free parameters(coefficients) in a model. A smoothing spline has $n$ parameters and hence $n$ nominal degree of freedom, but these $n$ parameters are heavily constrained. This phenomenon is measured by the effective degree of freedom.

In fitting a smoothing spline, we do not need to select the number of knots as there will be a knot at each training observation. Our main concern is the choice of $\lambda$. One possible approach is to choose $\lambda$ by croos-validation. LOOCV can be computed very efficiently for smoothing splines. The way RSS is calculated is slightly different though and is given as:

$$RSS_{cv}(\lambda) = \sum _{i=1}^{n} (y_i - \widehat{g _{\lambda}}^{(-i)}(x_i))^2 =
\sum _{i=1}^{n} \bigg[ \frac{y_i - \widehat{g _{\lambda}}(x_i)}{1- (S _{\lambda}) _{ii}} \bigg] ^2$$

Here $\widehat{g _{\lambda}}^{(-i)}(x_i)$ indicates the fitted value of smoothing spline evaluated at $x_i$, where the model uses all the training observation except $x_i$ (according to the definition of LOOCV). $\widehat{g _{\lambda}}(x_i)$ indicates the fit at $x_i$ using all the training observations. The matrix $S _{\lambda}$ can be computed as:

$$\widehat{g _{\lambda}} = S _{\lambda}y$$

where, $\widehat{g _{\lambda}}$ is the fitted values for a particular value of $\lambda$ and $y$ is the response vector. Hence, the <b>RSS of LOOCV can be computed by just using $\widehat{g _{\lambda}}$, which is the original fit using the entire data set</b>, and hence efficiently. The effective degree of freedom for the smoothing spline is given as:

$$df _{\lambda} = \sum _{i=1}^{n} (S _{\lambda}) _{ii}$$
