+++
date = "2018-07-06T01:19:52+01:00"
description = "ISLR Unsupervised Learning"
draft = false
tags = ["ISLR", "Unsupervised Learning", "Principal Components Analysis"]
title = "ISLR Chapter 10: Unsupervised Learning (Part 2: More on PCA)"
topics = ["ISLR"]

+++


#### 10.2.3 More on PCA

##### Scaling the Variables

The results of PCA also depend on the fact that whether <b>the variables are individually scaled or not</b>. If we perform PCA on the unscaled variables, the variables with higher variance will have very <b>large loading</b>. As it is undesirable for the principal components obtained to depend on the scale of the variables, we scale each variables to have the <b>standard deviation 1</b> before performing PCA. If the individual variables are measured in the same unit, the scaling need not to be done.

##### Uniqueness of the Principal Components

Each principal component loading vecotor is unique upto a <b>sign flip</b>. The sign of the loading vectors may differ as they specify directions in the $p$-dimensional space and hence flipping the sign has no effect. Similarly, the score vectors are unique up to a sign filp as the varince of $Z$ and $-Z$ are same.

##### The Proportion of Variance Explained

We may be iterested in the amount of variance that has been explained by projecting the data on to first $M$ principal components. This means that we are interested in knowing the <b>proportion of variance explained (PVE)</b> by each principal component. The <b>total variance</b> present in the data set can be given as:

$$\sum _{j=1}^{p} Var(X_j) = \sum _{j=1}^{p} \frac{1}{n} \sum _{i=1}^{n} x _{ij}^2$$

and the variance that is explained by the $m^{th}$ principal comonent is:

$$\frac{1}{n} \sum _{i=1}^{n} z _{im}^2 = \frac{1}{n} \sum _{i=1}^{n} \bigg( \sum _{j=1}^{p} \phi _{jm}x _{ij}\bigg)^2$$

Hence, <b>PVE</b> of the first principal component can be given as:

$$\frac{\sum _{i=1}^{n} \bigg( \sum _{j=1}^{p} \phi _{jm}x _{ij}\bigg)^2}{\sum _{j=1}^{p} \sum _{i=1}^{n} x _{ij}^2}$$

In order to obtain the PVE of first $M$ principal components, we can calculate each individual PVEs using the above equation and then sum them all.

##### Deciding How Many Principal Components to Use

The main goal of PCA is to find the smallest number of principal components that explains a good bit of variance in data. We can decide on the number of principal components to be used by examining the <b>scree plot</b>. It can be done by simply eyeballing the scree plot and finding a point at which the amount of variance explained by subsequent principal component drops off. This point is referred as the <b>elbow</b> in the scree plot.

This visual analysis is kind of <b>ad hoc</b> and there is no well-accepted objective way to decide on the number of principal components to be used. The number of principal components to be used depends on the area of application and the nature of the data set.

A simple approach is to find the first few principal components and then examine that whether there exists some interesting pattern in the data or not. If no pattern is found in the first few principal components, the subsequent principal components will be of lesser interest as well. This is a subjective approach and reflects on the fact that principal components are used for exploratory data analysis.

Principal components can be used for supervised analysis (as in principal component regression) as well. In this case, there is a simple and objective way to decide the number of principal components. The number of principal components to be used can be treated as a tuning parameter and can be decided by cross-validation or similar techniques.

#### 10.2.4 Other Uses for Principal Components

Like principal component regression, the principal component score vectors can be used as the features in several supervised learning techniques. This will lead to <b>less noisy</b> results as it is often the case that signal in the data set is concentrated in the first few principal components.
