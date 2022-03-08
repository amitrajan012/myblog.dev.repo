+++
date = "2018-06-19T12:14:23+01:00"
description = "ISLR Support Vector Machines"
draft = false
tags = ["ISLR", "Support Vector Machines", "Maximal Margin Classifier"]
title = "ISLR Chapter 9: Support Vector Machines (Part 1: Maximal Margin Classifier)"
topics = ["ISLR"]

+++

<h1><center>Support Vector Machines</center></h1>

<b>Support vector machine</b> is a generalization of a simple and intutive classifier called the <b>maximal margin classifier</b>. Maximal margin classifier has a limitation that it can be only applied to a data set whose classes are seperated by a linaer boundary.

### 9.1 Maximal Margin Classifier
#### 9.1.1 What Is a Hyperplane?

In a $p$-dimensional space, a <b>hyperplane</b> is a flat affine subspace of $p-1$ dimensions. In two dimensions, the hyperplane is a line and can be given as:

$$\beta_0 + \beta_1 X_1 + \beta_2 X_2 = 0$$

for parameters $\beta_0, \beta_1, \beta_2$. In a $p$ dimensional setting, the hyperplane can be given as:

$$\beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_p X_p = 0$$

If a point $X = (X_1, X_2, ..., X_p)^T$ in the space satisfies the above equation, the point $X$ lies on the hyperplane.

If $X$ satisfies $\beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_p X_p > 0$, $X$ lies to one side of the hyperplane. On the other hand, if $\beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_p X_p < 0$, $X$ lies on the other side of the hyperplane. Hence, a hyperplane divides a $p$-dimensional space into <b>two halves</b>.

#### 9.1.2 Classification Using a Separating Hyperplane

Classification can be done by using the concept of seperating hyperplanes. Suppose there exists a seperating hyperplane whihc divides the data set perfectly into two classes, labeled as $y_i=1, y_i=-1$. Separating hyperplane can be defined as:

$$\beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_p X_p > 0 \ \ if \ \ y_i= 1$$

$$\beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_p X_p < 0 \ \ if \ \ y_i= -1$$

and can be combined as:

$$y_i(\beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_p X_p) > 0; \forall \  i=1,2,..,n$$

If a separating hyperplane exists, it can be used to construct a classifier and an observation can be classified depending on which side of the hyperplane it is located. We can also use the <b>magnitude</b> of the distance of the observation from the hyperplane to decide the confidence of the classification. More the distance, the more is the confidence about the classification of the observation. A classifier based on a separating hyperplane leads to a <b>linear decision boundary</b>.

#### 9.1.3 The Maximal Margin Classifier

If the data set can be separated by a separating hyperplane, an infinity number of such hyperplanes exist. Hence, we need to devise a way to select which of the infinite possible hyperplanes to use.

A natural choice is the <b>maximal margin classifier</b> (also called as <b>optimal separating hyperplane</b>). It is the hyperplane which is farthest from the training observations. <b>Margin</b> is the smallest amongst the perpendicular distance of all the observations from the hyperplane. Maximal margin classifier is the hyperplane for which the <b>margin is maximum</b>. Maximal margin classifiers are often successful but they can lead to <b>overfitting</b> for large values of $p$.

<b>Support vectors</b> are the observations which are on the width of the classifier. If these points are moved, the maximal margin hyperplane will move as well, as these are the points which decide the end of the regions (margin) around the maximal margin classifier. Maximal margin hyperplane depends only on the support vectors. Displacement of other observations does not affect the maximal margin hyperplane, until and unless the observation crosses the boundary set by the margin.

#### 9.1.4 Construction of the Maximal Margin Classifier

Given a set of $n$ training observations $x_1, x_2, ..., x_n$ in a $p$-dimensional space having the class labels $y_1, y_2, ..., y_n \in \{-1, 1\}$, maximal margin hyperplane is the solution of the optimization problem:

$$maximize_{\beta_0, \beta_1, ..., \beta_p} M$$

$$subject \ to \ \sum_{j=1}^{p} \beta_j^2 = 1,$$
$$y_i(\beta_0 + \beta_1 X _{i1} + \beta_2 X _{i2} + ... + \beta_p X _{ip}) \geq M; \forall \  i=1,2,..,n$$

The second condition simply ensures that all the observations are on or beyond the margin $M$, given $M$ is positive, and on the correct side of the classification. Due to the first constraint, the perpendicular distance from the $i$th observation to the hyperplane is given by $y_i(\beta_0 + \beta_1 X _{i1} + \beta_2 X _{i2} + ... + \beta_p X _{ip})$. Hence, $M$ represents the <b>margin</b> of the hyperplane and the optimization problem chooses $\beta_0, \beta_1, ..., \beta_p$ that maximizes $M$.

#### 9.1.5 The Non-separable Case

Maximal margin classifier can be obtained if and only if the seperating hyperplane exists. The generalization of the maximal margin classifier to accomodate the <b>non-separable</b> classes is known as the <b>support vector classifier</b>.
