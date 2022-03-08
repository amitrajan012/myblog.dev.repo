+++
date = "2018-05-24T01:06:16+01:00"
description = "ISLR Linear Model Selection and Regularization"
draft = false
tags = ["ISLR", "Resampling", "Linear Model Selection", "Regularization", "Dimension Reduction"]
title = "ISLR Chapter 6: Linear Model Selection and Regularization (Part 3: Dimension Reduction Methods)"
topics = ["ISLR"]

+++

### 6.3 Dimension Reduction Methods

Instead of performing a least squares regression on all the $p$ predictors, we can <b>transform</b> the predictors and then fit a least squares model on the transformed variables. Let the transformed variables be $Z_1, Z_2, ..., Z_M$, where $M < p$, where each of $Z_m$s is a linear combination of predictors $X_1, X_2, ..., X_p$.i.e.

$$Z_m = \sum _{j=1}^{p} \phi _{jm}X_j$$

We can then fit the least squares regression model as $Z_m$s as the predictors:

$$Y = \theta_0 + \theta_1 Z_1 + ... + \theta_M Z_M + \epsilon$$

If the transformation constants are chosen wisely, then the dimensionality reduction approach can outperform plain least squares models. The term <b>dimensionality reduction</b> comes form the fact that instead of predicting $p+1$ coefficients, we only need to predict $M+1$ coefficients now. The least square coefficients can be derived from the transformation constants and the coefficients of the dimensionality reduction model as:

$$\beta_j = \sum _{m=1}^{M} \theta_m \phi _{jm} $$

Dimensionality reduction reduces the variance of the model. It works in two steps: first the transformed predictors $Z_1, Z_2, ... , Z_M$ are obtained and then the model is fitted using these $M$ predictors. Dimesionality reduction can be achieved by: <b>principal componenets</b> and <b>partial least squares</b>.

#### 6.3.1 Principal Components Regression

<b>Principal Component Analysis (PCA)</b> can be used to derive a low-dimensional set of features from a large set of variables. The <b>first principal component</b> is the direction of the data along which the observations <b>vary the most</b>. For a two variable case, the linear combination of the variables can be written as $\phi _{11} X_1 + \phi _{21} X_2$, where $\phi _{11}^2 + \phi _{21}^2 = 1$. The idea of PCA is to find the linear combination for which the variance is maximum out of all the possible ones. It is necessary to put a constraing of $\phi _{11}^2 + \phi _{21}^2 = 1$, as otherwise, variance can simply be increased by increasing the $\phi$s. <b>First principal componenet</b> can also be described as the line which is as close as possible to the data.

The <b>second principal component</b> can be described as the linear combinations of variables that is <b>uncorrelated to the first principal component $Z_1$</b> (this means that it should be <b>orthogonal</b> to the fisrt principal component) and have the largest variance.

##### The Principal Components Regression Approach

The <b>principal components regression (PCR)</b> approach constructs the first $M$ principal components first and then fit a least squares linear regression model on them. The basic idea behind the PCR is the fact that the <b>directions in which the predictors $X_1, X_2, ..., X_p$ shows the most variation are the directions that should be associated with the response $Y$</b>. Hence PCR can give better results over least square linear regression model by <b>mitigating overfitting</b>. As the number of principal components in a model is increased, the bias will decrease and the variance will increase. Performing PCR with appropriate number of principal components can improve the performance of the model compared to the least squares linear regression model.

PCR is a method of performing regression using $< p$ predictors but it is not a feature selection method as all the $p$ features are contained in $M$ principal components and hence <b>PCR is closely related to ridge regression</b>. One can even think of ridge regression as a continuous version of PCR.

In PCR, the desired number of principal components can be chosen by cross-validation. Prior to performing PCR, we need to standardize the individual variables as if we do not do so, the variables with higher variance will tend to affect the principal components obtained and hence will have an effect on the PCR model.

#### 6.3.2 Partial Least Squares

In PCR, the linear combinations or directions are identified in an <b>unsupervised way</b> (as the response $Y$ is not used to determine the principal components). Hence, it may be the case that the directions that best describe the predictors are the ones which best describe the response. In this case, the performance will PCR will suffer.

<b>Partial least squares (PLS)</b> is a <b>supervised</b> alternative to PCR. PLS identifies the new features (dimensionality reduction) in a supervised way (it makes the use of response $Y$ in the identification of the new features). In a nut shell, <b>PLS attempts to find directions that explain both the response and the predictors.</b>

In PLS, the <b>first principal component</b> is computed by setting the <b>transformation coefficients ($\phi _{j1}$)</b> as the <b>coefficient of the simple linear regression</b> of response $Y$ onto individual variables $X_j$. Hence, PLS gives the highest weights to the variables that are most strongly related to the response while doing the transformation.

Second PLS direction is identified by removing from each of the variables $X$s, the part which is already explained by $Z_1$. This can be done by <b>regressing the each variable on $Z_1$ and then taking the residuals.</b> These residuals are then used to determine the second principal component. This iterative approach is performed $M$ times to identify the PLS components $Z_1, Z_2, ..., Z_M$, which are then used to find the least squares fit of $Y$. $M$ is a tuning parameter, which can be found using cross-validation. We need to <b>standardize the predictors and response</b> before performing PLS. In practice it often performs no better than ridge regression or PCR.

### 6.4 Considerations in High Dimensions
#### 6.4.1 High-Dimensional Data

Data set which has considerably more features than observations are termed as <b>high-dimensional</b> data set. Classical approaches such as least squares linear regressions are not suitable for this setting.

#### 6.4.2 What Goes Wrong in High Dimensions?

In a high-dimensional setting, no matter whether there is a true relationship between the features and the response, classical approaches will lead to the models which will be a perfect fit to the data (residuals are 0). This is problematic as these perfect fit will lead to <b>overfitting</b>. For example, when $p \approx n$ or $p > n$, the least square line will is too <b>flexible</b> and hence overfits the data. Hence, we need to take extra care while analyzing high-dimensional data sets and always evaluate model performance on an independent test set. The various approaches to predict test MSE, such as $C_p$, BIC, AIC and adjusted $R^2$ does not perform well in high-dimensional settings and hence alternative approaches, best suited for high-dimensional settings are required.

#### 6.4.3 Regression in High Dimensions

Various approaches that are described for fitting a <b>less flexible</b> least square models (subset selection, stepwise selection, ridge regression, lasso) perorm well in high-dimensional setting. While using these models in high-dimensional setting we should always follow these points:

 - <b>Regularization</b> or <b>shrinkage</b> plays a key-role in high dimensional setting


 - Selection of appropriate <b>tuning parameter</b> is key for good prediction accuracy


 - Test error will increase as the dimensionality of the problem increases, until and unless the added features are truely associated with the response (also called as <b>curse of dimensionality</b>).

Hence, more extensive data collection can lead to overfitted model without much improvement in the prediction accuracy until and unless new features are associated with response.

#### 6.4.4 Interpreting Results in High Dimensions

In high-dimensional setting, the problem of <b>multi-collinearity</b> (variables in the regression might be correlated to each other) is a huge problem and hence we can not truly identify the variables which are predictive of the outcome. This may lead to <b>overstating</b> the results as the predictors included in the model will not be the only one which can predict the response efficiently. There may be a case that if we design a model based on a fresh training data, we may lead to get a different set of significant predictors but the model accuracy will remain unchanged.

Apart from this, it would be misleading to report traditional measures for the goodness of fit (p-value, $R^2$ statistic) on the <b>training data</b> as the statistic for model performance as it is easy to obtain 0 residuals. Instead, we can report the results obtained on an independent test set or may perform cross-validation.
