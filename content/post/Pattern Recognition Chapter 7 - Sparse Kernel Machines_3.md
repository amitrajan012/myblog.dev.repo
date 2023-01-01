+++
date = "2022-08-23T23:07:28+01:00"
description = "Pattern Recognition (Bishop): Chapter 7"
draft = false
tags = ["Bishop", "Pattern Recognition", "Kernel Methods", "Maximum Margin Classifiers", "Lagrange Multipliers", "Overlapping Class Distributions", "Slack Variables", "Support Vectors"]
title = "Sparse Kernel Methods - Maximum Margin Classifiers: Overlapping Class Distributions"
topics = ["Pattern Recognition"]

+++


### 7.1.1 Overlapping Class Distributions

In practice, the class-conditional distributions may overlap, in which case exact separation of the training data can lead to poor generalization. In maximum margin classifier for separable classes, we implicitly used an error function that gave infinite error if a data point was misclassified and zero error if it was classified correctly, and then optimized the model parameters to maximize the margin. We now modify this approach so that data points are allowed to be on the <b>wrong side</b> of the margin boundary, but with a penalty that increases with the distance from that boundary. It is convenient to make this penalty a linear function of this distance. To do this, we introduce <b>slack variables</b> $\xi_n \geq 0$ for all $n=1,2,...,N$, with one slack variable for each training data point. With a slack, some of the data points are allowed to be on the other side of margin and some are even allowed to be misclassified. The value of slack variables for different data points are as follows:
* Data points classified correctly and on the margin or correct side of the margin: $\xi_n = 0$
* Data points classified correctly and between decision surface and the margin: $0 < \xi_n < 1$
* Data points on the decision surface: $\xi_n = 1$
* Data points which are misclassified i.e. on the other side of decision surface: $\xi_n > 1$

The modified classification constraint is given as:

$$\begin{align}
t_n y(X_n) \geq 1 - \xi_n
\end{align}$$

Introduction of slack variables is sometimes described as relaxing the hard margin constraint to give a
soft margin and allows some of the training set data points to be misclassifie. While slack variables allow for overlapping class distributions, this framework is still sensitive to outliers because the penalty for misclassification increases linearly with $\xi$.

Our goal is now to maximize the margin while soft penalizing points that lie on the wrong side of the margin. We therefore minimize

$$\begin{align}
\frac{1}{2}||W||^2 + C\sum_{n=1}^{N}\xi_n
\end{align}$$

where the parameter $C>0$ controls the trade-off between the slack variable penalty and the margin. Any point which is misclassified has $\xi_n > 1$ and hence $\sum_n\xi_n$ is the upper bound on the number of misclassified points. In the limit $C \to \infty$, we will have the earlier support vector machine for separable data.

Our goal is to minimize the following expression

$$\begin{align}
\frac{1}{2}||W||^2 + C\sum_{n=1}^{N}\xi_n
\end{align}$$

with constraints

$$\begin{align}
t_n y(X_n) \geq 1 - \xi_n
\end{align}$$

$$\begin{align}
\xi_n \geq 0
\end{align}$$

The corresponding Lagrangian is given as

$$\begin{align}
L(W,b,a) = \frac{1}{2}||W||^2 + C\sum_{n=1}^{N}\xi_n - \sum_{n=1}^{N}a_n[t_n y(X_n) - 1 + \xi_n] - \sum_{n=1}^{N}\mu_n \xi_n
\end{align}$$

where $\{a_n \geq 0\}$ and $\{\mu_n \geq 0\}$ are Lagrange multipliers. The corresponding set of KKT conditions are given as

$$\begin{align}
a_n \geq 0
\end{align}$$

$$\begin{align}
t_n y(X_n) - 1 + \xi_n \geq 0 
\end{align}$$

$$\begin{align}
a_n(t_n y(X_n) - 1 + \xi_n) = 0 
\end{align}$$

$$\begin{align}
\mu_n \geq 0
\end{align}$$

$$\begin{align}
\xi_n \geq 0
\end{align}$$

$$\begin{align}
\mu_n\xi_n = 0
\end{align}$$

where $n=1,2,...,N$. Replacing $y(X_n) = W^T\phi(X_n) + b$ and differentiating with respect to $W,b,\xi_n$ and equating them to $0$, we have

$$\begin{align}
\frac{\partial L}{\partial W} = 0 \implies W = \sum_{n=1}^{N} a_nt_n\phi(X_n)
\end{align}$$

$$\begin{align}
\frac{\partial L}{\partial b} = 0 \implies \sum_{n=1}^{N} a_nt_n = 0
\end{align}$$

$$\begin{align}
\frac{\partial L}{\partial \xi_n} = 0 \implies a_n = C - \mu_n
\end{align}$$

Using these conditions, we obtain the dual Lagrangian in the form

$$\begin{align}
\tilde{L}(a) = \frac{1}{2}||W||^2 + C\sum_{n=1}^{N}\xi_n - \sum_{n=1}^{N}a_n[t_n y(X_n) - 1 + \xi_n] - \sum_{n=1}^{N}\mu_n \xi_n
\end{align}$$

$$\begin{align}
= \frac{1}{2}||W||^2 - \sum_{n=1}^{N}a_n t_n y(X_n)+ C\sum_{n=1}^{N}\xi_n + \sum_{n=1}^{N}a_n[1 - \xi_n] - \sum_{n=1}^{N}(C - a_n) \xi_n
\end{align}$$

$$\begin{align}
= \frac{1}{2}||W||^2 - \sum_{n=1}^{N}a_n t_n y(X_n)+ \sum_{n=1}^{N} a_n
\end{align}$$

$$\begin{align}
\tilde{L}(a) = \sum_{n=1}^{N} a_n - \frac{1}{2}\sum_{n=1}^{N} \sum_{m=1}^{N} a_na_m t_nt_m k(X_n,X_m)
\end{align}$$

which is identical to the separable case except that the constraints are different. From the constraint $\mu_n \geq 0$ and the expression $a_n = C - \mu_n$, we have $a_n \leq C$ which gives us a new constraint on $a_n$ as $0 \leq a_n \leq C$. Hence the above dual representation needs to be minimized with the constraints

$$\begin{align}
\sum_{n=1}^{N} a_nt_n = 0
\end{align}$$

$$\begin{align}
0 \leq a_n \leq C
\end{align}$$

for $n=1,2,...,N$ where the second constraint is known as <b>box constraints</b>. Based on value of $a_n$, the data points can be classified as follows:

* $a_n = 0$: For a subset of data points, we may have $a_n = 0$ and hence they do not contribute to the prediction. 
* $a_n > 0$: The remaining data points will constitute support vectors and will have $a_n > 0$ and hence must satisfy

$$\begin{align}
t_n y(X_n) = 1 - \xi_n
\end{align}$$

The case of $a_n > 0$ can be further classified as:

* $a_n < C$: This implies that $\mu_n > 0$ and hence $\xi_n = 0$ (from $\mu_n\xi_n = 0$). These points lie on the margin.
* $a_n = C$: This implies that $\mu_n > 0$ and hence $\xi_n > 0$. This can be further divided into two cases as $\xi_n \leq 1$ as the points that lie inside the margin and are correctly classified and $\xi_n > 1$ as the points that lie inside the margin and are incorrectly classified (or to the other side of the decision surface).

To determine $b$, we can take the support vectors for which $0 < a_n < C$ and have $\xi_n = 0$. This means that for these support vectors, $t_n y(X_n) = 1$ and hence will satisfy

$$\begin{align}
t_n \bigg(\sum_{m\in S}a_mt_mk(X_n,X_m) + b \bigg) = 1
\end{align}$$

The numerical stable solution can be found by multiplying the above equation by $t_n$ and taking the average for all the data points for which $0 < a_n < C$ as

$$\begin{align}
b = \frac{1}{N_M}\sum_{m \in M}\bigg( t_n - \sum_{m\in S}a_mt_mk(X_n,X_m) \bigg)
\end{align}$$

where $M$ denotes the set of indices of the data points for which $0 < a_n < C$.

It should be noted that althouh the predictions for new inputs are made using only the support vectors, the training phase makes use of the whole data set. Another thing to note is the fact that the support vector machine does not provide probabilistic outputs but instead makes classification decisions for new input vectors. However, if we wish to use the SVM as a module in a larger probabilistic system, then probabilistic predictions of the class label $t$ for new inputs $X$ are required. One approach to get probabilities is to fit a logistic sigmoid function to the outputs of a previously trained SVM. The data used to fit the sigmoid needs to be independent of that used to train the original SVM in order to avoid severe over-fitting. However, the SVM can give a poor approximation to the probabilities.

