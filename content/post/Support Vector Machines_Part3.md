+++
date = "2018-06-23T04:19:37+01:00"
description = "ISLR Support Vector Machines"
draft = false
tags = ["ISLR", "Support Vector Machines", "Support Vector Machines"]
title = "ISLR Chapter 9: Support Vector Machines (Part 3: Support Vector Machines)"
topics = ["ISLR"]

+++


### 9.3 Support Vector Machines

#### 9.3.1 Classification with Non-linear Decision Boundaries

Support vector classifiers, which are designed to work in the setting of linear decision boundary, can be extended to handle the case of non-linear decision boundary by enlarging the feature space using polynomial transformation of the predictors. For example, if we have a $p$-dimensional feature space given as: $X_1, X_2, ..., X_p$, we could instead fit a support vector classifier using $2p$ features: $X_1, X_1^2, X_2, X_2^2, ..., X_p, X_p^2$. In the transformed feature space, the decision boundary is still linear, but if we consider the original feature space, the decison boundary will be non-linear. There are many ways to enlarge the feature space, such as using higher order polynomials or using interaction terms.

#### 9.3.2 The Support Vector Machine

The <b>support vector machine(SVM)</b> is an extension of support vector classifier by enlarging the feature space using <b>kernels</b>. For obtaining the solution of the support vector classifier problem, we only need to have the <b>inner products</b> of the observations. The inner product of the observations $x_i$ and $x _{i^{'}}$ is given as:

$$\langle x_i, x _{i^{'}} \rangle = \sum _{j=1}^{p} x _{ij} x _{i^{'}j}$$

Linear support vector classifier can be represented as:

$$f(x) = \beta_0 + \sum _{i=1}^{n} \alpha_i \langle x, x_i \rangle$$

where $\alpha_i$s are a total of $n$ parameters, one per training observation. To estimate the parameters $\beta_0, \alpha_1, ..., \alpha_n$, we need to compute ${n}\choose{2}$ inner products for each combination of training observation. It turns out that if the training observation is <b>not a support vector</b>, the $\alpha_i$ is 0. Hence, $\alpha_i$s are non-zero only for support vectors. Hence, the solution can be rewritten in the form

$$f(x) = \beta_0 + \sum _{i \in S} \alpha_i \langle x, x_i \rangle$$

where $S$ is the set containing the indices of the support vectors.

A <b>kernel</b> is a function that <b>quantifies the similarity of two observations</b>. Inner product can be considered as a kernel and hence can be represented as:

$$K(x_i, x _{i^{'}}) = \sum _{j=1}^{p} x _{ij} x _{i^{'}j}$$

The above equation is a <b>linear kernel</b> as the support vector classifiers are linear in features. Linear kernel quantifies the similarity of a pair of observation using <b>Pearson (standard) correlation</b>. A kernel of the form:

$$K(x_i, x _{i^{'}}) = \bigg(1 + \sum _{j=1}^{p} x _{ij} x _{i^{'}j} \bigg)^d$$

is called as <b>polynomial kernel</b> of degree $d$. Using polynomial kernel in a support vector classifier leads to a much more flexible decision boundary. When a support vector classifier is combined with a non-linear kernal it is known as <b>support vector machine</b>. In this case, the function has the form:

$$f(x) = \beta_0 + \sum _{i \in S} \alpha_i K(x, x _{i})$$

Apart from polynomial kernel, another popular choice for non-linear kernel is <b>radial kernel</b>, which is given as:

$$K(x_i, x _{i^{'}}) = exp \bigg( - \gamma \sum _{j=1}^{p} (x _{ij} - x _{i^{'}j})^2 \bigg)$$

where $\gamma$ is a positive constant. The intution behind working of radial kernel is as follows:

<b><center>For the training observations which are far from the given test observation $x^\*$, the Euclidean distance will be large and hence the overall value of $K(x^\*, x_i)$ will be small and hence $x_i$ will play virtually no role in $f(x^\*)$. In other words, training observations that are far from $x^\*$ will play almost no role in the predicted class label for $x^*$ and hence the radial kernel shows a local behaviour.</center></b>

One advantage of using kernel instead of using enlarged feature space is the less computational complexity as we just need to compute ${n} \choose {2} $ values of $K(x_i, x _{i^{'}})$.

### 9.4 SVMs with More than Two Classes

SVM can be extended for more than two classes by the methods: <b>one-versus-one</b> and <b>one-versus-all</b>.

#### 9.4.1 One-Versus-One Classification

A <b>one-versus-one</b> or <b>all-pairs</b> classification process builds ${K} \choose {2}$ SVMs, each of which compares a pair of classes. A test observation is classified using each of the ${K} \choose {2}$ classifiers. The final classification is done by assigning the test observation to the class it was most frequently assigned by all of these ${K} \choose {2}$ classifiers.

#### 9.4.2 One-Versus-All Classification

In <b>one-versus-all</b> approach, $K$ SVMs are fit, each time comparing one of the $K$ classes to remaining $K-1$ classes. The test observation is assigned to the class for which the value of $f(x^*)$ is the largest.
