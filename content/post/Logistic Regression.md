+++
date = "2018-12-05T07:26:51+05:30"
description = "Logistic Regression: Derivation"
draft = false
tags = ["Logistic Regression", "Classification", "Newton-Raphson Method", "Taylor Series"]
title = "Logistic Regression"
topics = ["Logistic Regression"]
+++

</br>
In a classification setting, <b>logistic regression</b> models the probability of a response $Y$ belonging to a particaular category. A simple linear regression can not be used for classification as the output of a linear regression can have a range that goes from $-\infty$ to $\infty$ (we need to find the values in the range [0, 1]). Instead, we can transform the output of linear regression such that the output is confined in the range [0, 1]. For this, a <b>logistic function</b> which is given below can be used.

$$p(X) = \frac{e^{\beta_0 + \beta X}}{1 + e^{\beta_0 + \beta X}}$$

Manipulating the above function and taking the log, we get

$$log\bigg( \frac{p(X)}{1-p(X)} \bigg) = \beta_0 + \beta X$$

The left hand side expression is called as <b>log-odds</b> of <b>logit</b>. For a logistic regression, <b>logit is linear in $X$</b>.

A method of <b>maximum-likelihood</b> is used to fit the logistic regression model. In maximum-likelihood, we seek estimates of $\beta_0$ and $\beta$ such that the predicted probabilities corresponds as closely as possible to the observed individual probabilities. The <b>likelihood-function</b> is given as:

$$L(\beta _0, \beta) = \prod _{i:y_i=1}p(x_i) \prod _{i:y_i=0}(1 - p(x_i)) = \prod _{i=1}^{n}p(x_i)^{y_i} (1- p(x_i))^{1 - y_i}$$

In maximum-likelihood, our goal is to maximize the likelihood-function. As logarithm is an increasing function, we can transform the likelihood-function by taking logarithm and our objective remains unaltered. Hence, the <b>log-likelihood</b> (after taking the logarithm of the likelihood-function) is given as:

$$l(\beta_0, \beta) = log(L(\beta_0, \beta)) = \sum _{i=1}^{n} y_i log(p(x_i)) + (1- y_i)(1 - log(p(x_i)))$$

On further simplification, we get

$$l(\beta_0, \beta) = \sum _{i=1}^{n} log(1 - p(x_i)) + \sum _{i=1}^{n} y_i log \frac{p(x_i)}{1 - p(x_i)} =
-\sum _{i=1}^{n} log(1 + e^{\beta_0 + \beta x_i}) + \sum _{i=1}^{n} y_i (\beta_0 + \beta x_i)$$

To find the maximum likelihood estimates, we need to take the partial derivatives of log-likelihood function with respect to different parameters and set them to 0. Taking the derivative with respect to one component of $\beta$ (say it as $\beta_j$), we get

$$\frac{\partial l}{\partial \beta_j} = -\sum _{i=1}^{n} \frac{1}{e^{\beta_0 + \beta x_i}} e^{\beta_0 + \beta x_i} x _{ij}+ \sum _{i=1}^{n} y_i x _{ij} = \sum _{i=1}^{n} (y_i - p(x_i; \beta_0, \beta)) x _{ij}$$

As exponential function is a <b>transcendental function</b> (as it goes beyond the limit of algebra in that it can not be expressed in terms of a finite sequence of the algebraic operations (addition, multiplication,...)), the above equation is a <b>transcendental equation</b>. A transcendental equations do not have <b>closed-form solutions</b> (an expression that can be evaluated in a finite number of mathematical operations). However this can be solved approximately to get a numerical solution.

<b>Newton-Raphson</b> method can be used to find an approximate solution for the above mentioned optimization problem. Let us first assume a simpler case of minimizing a function of one scalar variable, denoted as $f(\beta)$. We have to find the location of the <b>global-minima</b> $\beta^{*}$. For $\beta^{ * }$ to be a global-minima, the first-derivative at $\beta^{*}$ should be 0 and the second-derivative should be positive. Doing a <b>Taylor Series</b> expansion near minima, we get

$$f(\beta) \approx f(\beta^{ * }) + (\beta - \beta^{ * })f^{'}(\beta^{ * }) + \frac{(\beta - \beta^{ * })^2}{2} f^{"}(\beta^{ * }) + ... = f(\beta^{ * }) + \frac{(\beta - \beta^{ * })^2}{2} f^{"}(\beta^{ * })$$

The last expression comes due to the fact that the first-derivative at the global-minima, i.e. $f^{'}(\beta^{*})$ is 0. Hence, instead of minimizing the original function, we can minimize the <b>quadratic approximation</b> of the original function.

In the Newton-Raphson method, we guess an initial point (let it be $\beta^{0}$). If this point is close to the minima, we can take a second order taylor expansion around $\beta^{0}$ and it will still be close to the original expression:

$$f(\beta) \approx f(\beta^{0}) + (\beta - \beta^{0})f^{'}(\beta^{0}) + \frac{(\beta - \beta^{*0})^2}{2} f^{"}(\beta^{0})$$

So, instead of minimizing the original expression, we minimize the right hand side, which is the second-order Taylor expansion of the original expression around $\beta^{0}$. Taking the derivative of the right hand side expression and setting it equal to 0, we get

$$f^{'}(\beta^{0}) + (\beta - \beta^{0}) f^{"}(\beta^{0}) = 0$$

This gives us a new guess $\beta^{1}$ for the global-minima $\beta^{*}$ as

$$\beta^{1} = \beta^{0} - \frac{f^{'}(\beta^{0})}{f^{"}(\beta^{0})}$$

After repeating the process till $n$ iterations, we get the $(n+1)^{th}$ guess as:

$$\beta^{n+1} = \beta^{n} - \frac{f^{'}(\beta^{n})}{f^{"}(\beta^{n})}$$

This gives us the approximation for global-minima.
