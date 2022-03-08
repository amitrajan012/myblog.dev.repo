+++
date = "2018-06-20T02:24:24+01:00"
description = "ISLR Support Vector Machines"
draft = false
tags = ["ISLR", "Support Vector Machines", "Support Vector Classifiers"]
title = "ISLR Chapter 9: Support Vector Machines (Part 2: Support Vector Classifiers)"
topics = ["ISLR"]

+++


### 9.2 Support Vector Classifiers

#### 9.2.1 Overview of the Support Vector Classifier

The maximal margin classifiers can be sensitive to individual observations. Sometimes, adding a single observation in the data set, can lead to dramatic change in the separating hyperplane. The sensitivity and the low margin for a maximal margin classifier may suggest that the maximal margin classifier has <b>overfit</b> the training data. So, sometimes we may be willing to consider a classifier based on a hyperplane that does not perfectly separate the two classes, and hence, will be <b>more robust (or less sensitive to individual observations)</b> and will give better results for the unseen data points.

<b>Support vector classifiers (soft margin classifiers)</b> misclassify a few observations for the sake of robustness and better results. In a soft margin classifier, some of the observations can lie on the wrong side of the margin or even on the wrong side of the hyperplane. Observations that are on the wrong side of the hyperplane are misclassified by the support vector classifier.

#### 9.2.2 Details of the Support Vector Classifier

Support vector classifier is the solution of the optimization problem which is given as:

$$maximize _{\beta_0, \beta_1, ..., \beta_p, \epsilon_1, \epsilon_2, ..., \epsilon_n} M$$

$$subject \ to \ \sum _{j=1}^{p} \beta_j^2 = 1,$$
$$y_i(\beta_0 + \beta_1 X _{i1} + \beta_2 X _{i2} + ... + \beta_p X _{ip}) \geq M(1 - \epsilon_i); \forall \  i=1,2,..,n,$$
$$\epsilon_i \geq 0, \ \sum _{i=1}^{n}\epsilon_i \leq C$$

where $C$ is a non-negative tuning parameter. Here $\epsilon_1, ..., \epsilon_n$ are <b>slack variables</b> and allow individual observations to be on the wrong side of the margin or the hyperplane. For a test observation $x^{\*}$, we simply classify it based on the sign of $f(x^\*) = \beta_0 + \beta_1 x_1^* + ... + \beta_p x_p^*$.

The slack variable $\epsilon_i$ depicts the location of $i$th observation relative to the margin and the hyperplane. If $\epsilon_i = 0$, the observation <b>lies on the correct side of the margin</b>. If $\epsilon_i > 0$, the observation is on the <b>wrong side of the margin</b> and if $\epsilon_i > 1$, the observation is on the <b>wrong side of the hyperplane</b>.

The hyperparamter $C$ amounts for the budget for the amount by which the margin can be violated by $n$ observations. If $C=0$, there is no budget for the violation of the margin and hence the support vector classifier turns into maximal margin classifier. For $C>0$, no more than $C$ observations can be on the wrong side of the hyperplane (as $\epsilon_i > 1$ for the observation to be on the wrong side of the hyperplane). If $C$ increases, the margin will widen and if it decreases, the margin will become narrow. $C$ is chosen via <b>cross-validation</b>. When $C$ is <b>small</b>, the margin narrows and hence the classifier fits more colsely to the data, giving a <b>high variance</b>. For <b>larger</b> $C$, the margin is wider and hence fitting the data less hard and obtaining a classifier whihc is <b>more biased</b> but have <b>low variance</b>.

The observations that lie on the correct side of the margin do not affect the hyperplane. The observations that either lie on the margin or that violate the margin affect the hyperplane and hence are known as <b>support vectors</b>. For larger value of $C$, we will get a more number of support vectors and hence the hyperplane will depend on large number of observations, making the model low in variance.
