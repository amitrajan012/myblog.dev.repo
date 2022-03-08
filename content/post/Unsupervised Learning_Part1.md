+++
date = "2018-07-04T03:09:02+01:00"
description = "ISLR Unsupervised Learning"
draft = false
tags = ["ISLR", "Unsupervised Learning", "Principal Components Analysis"]
title = "ISLR Chapter 10: Unsupervised Learning (Part 1: Principal Components Analysis)"
topics = ["ISLR"]

+++

<h1><center>Unsupervised Learning</center></h1>

### 10.1 The Challenge of Unsupervised Learning

<b>Unsupervised learning</b> is often performed as a a part of an <b>exploratory data analysis</b>. In unsupervised learning there is no practical way to assess the performance of the model as we do not have the true answer.

### 10.2 Principal Components Analysis

In the case of a large set of correlated variables, principal component analysis helps in summarizing the data set with a small number of representative variables that explains most of the variability in the data. The principal component directions are the directions in the feature space along which the original data is <b>highly variable</b>. <b>Principal component analysis (PCA)</b> refers to the process by which principal components are computed.

#### 10.2.1 What Are Principal Components?

To visualize a data set with $n$ observations with a $p$ sets of features $X_1, X_2, ..., X_p$, we can examine $p \choose 2$ two-dimensional scatter plots of the data. For a large value of $p$, the process is cumbersome and most likely none of them will informative as each contain only a small fraction of total information present in the data. Hence, we would be keen to find a low-dimensional representation of the data that captures as much of the information as possible in the data set. PCA finds this low-dimensional representation of the data set that contains as much as possible of the variation. Each of the dimensions found by PCA is a <b>linear combination</b> of the $p$ features.

The <b>first principal component</b> of a set of features $X_1, X_2, ..., X_p$ is the <b>normalized</b> linear combination of the featues

$$Z_1 = \phi _{11}X_1 + \phi _{21}X_2 + ... + \phi _{p1}X_p$$

that has the <b>largest variance</b>. By <b>normalized</b>, we mean that $\sum _{j=1}^{p} \phi _{j1}^2 = 1$. The elements $\phi _{11}, \phi _{21}, ... \phi _{p1}$ are referred to as the <b>loading</b> of the first principal component. Without normalization of the loadings, the  values can be set to arbitrarily large values and hence resulting is a large variance.

As for the computation of first principal componenet, we are mainly interested in the variance, we can assume that each of the features in the data set has been centered to have 0 mean. Then, we need to maximize the variance with a constraint that the loadings are normalized. Hence, the first principal component loading vector solves the optimization problem

$$maximize _{\phi _{11}, \phi _{21}, ..., \phi _{p1}} \bigg( \frac{1}{n} \sum _{i=1}^{n} \bigg( \sum _{j=1}^{p}  \phi _{j1}x _{ij}\bigg)^2\bigg) \ subject \ \ to \ \ \sum _{j=1}^{p}\phi _{j1}^2 = 1$$

The above mentioned objective can instead be written as $\frac{1}{n} \sum _{i=1}^{n} z _{i1}^2$. As the mean of $z_i$s will be 0 as well, the objective that we are maximizing is simply the sample varince of the $n$ values of z _{i1}. $z _{11}, z _{21}, ..., z _{n1}$ are called as the <b>scores</b> of the first principal component. This problem can be solved via an <b>eigen decomposition</b>.

Geometrically, if we project the $n$ data points $x_1, x_2, ..., x_n$ along the direction of the loading vector of the first principal component, the projected values are the principal componenet scores $z _{11}, z _{21}, ..., z _{n1}$.

Similarly, the <b>second principal component</b> is the linear combination of $X_1, X_2, ..., X_p$ that has maximal variance out of all linear combinations that are <b>uncorrelated</b> with $Z_1$. It takes the form

$$z _{i2} = \phi _{12}x _{i1} + \phi _{22}x _{i2} + ... + \phi _{p2}x _{ip}$$

where $\phi_2$ is the second principal component loading vector, with elements $\phi _{12}, \phi _{22}, ..., \phi _{p2}$. Constraining $Z_2$ to be uncorrelated with $Z_1$ results in the direction of $\phi_2$ to be <b>orthogonal</b> of $\phi_1$. The optimization problem for finding the second principal componenet can be formed in a similar way as the one for the first principal component with an additional constraint that the $\phi_2$ is orthogonal to $\phi_1$. It turns out that the principal component directions $\phi_1, \phi_2, \phi_3, ...$ are the ordered sequence of eigenvectors of the matrix $X^TX$, and the eigenvalues are the variance of the components.

Usually, PCA is performed after <b>standardizing each variable to have mean 0 and standard deviation 1</b>.

#### 10.2.2 Another Interpretation of Principal Components

Alternatively, principal components can be viewed as the low-dimensional linear surfaces that are <b>closest</b> to the observations. The first principal component loading vector can be viewed as the line in the $p$-dimensional space that is closest to the $n$ observations (closeness is defined in terms of Euclidean distance). This can be explained simply as we seek a single dimension of the data which is as close as possible to all the data points. It is more likely that this line will explain the variation in data more aptly.

This notion can be extended for second and higher principal components as well. The first two principal components can be viewed as the plane that is closest to the $n$ observations in terms of Euclidean distance. Using this interpretation, first $M$ principal components, with their scores and loading vectors, provides the best $M$-dimensional approximation of the data set in terms of the Euclidean distance. For a sufficiently large value of $M$, the $M$ principal components score and loading vectors provide a good approximation for the observations. When $M = min(n-1, p)$, the the representation is exact and hence $x _{ij} = \sum _{m=1}^{M} z _{im} \phi _{jm}$.
