+++
date = "2022-07-02T23:07:28+01:00"
description = "Pattern Recognition (Bishop): Chapter 4"
draft = false
tags = ["Bishop", "Pattern Recognition", "Linear Models", "Classification", "Perceptron Algorithm", "Least Squares", "Learning Rate"]
title = "Linear Models for Clasification - The Perceptron Algorithm"
topics = ["Pattern Recognition"]

+++

### 4.1.7 The Perceptron Algorithm

In a <b>perceptron algorithm</b>, input vector $X$ is first transformed using a fixed nonlinear transformation to get a feature vector $\phi(X)$, and this is then used to construct a generalized linear model of the form

$$\begin{align}
y(X) = f(W^T\phi(X))
\end{align}$$

where the nonlinear <b>activation function</b> is given as

$$\begin{align}
f(a) = 
\begin{cases}
    +1, & a \geq 0\\
    -1, & a < 0
\end{cases}
\end{align}$$

Here we will use a target coding scheme of $\{-1,+1\}$, where $t_n=+1$ for $C_1$ and $t_n=-1$ for $C_2$. A natural choice of error function would be the total number of misclassified patterns. But this error function is piecewise constant funtion of $W$ and hence the gradient with respect to $W$ will be $0$ almost everywhere. Insted, we can use an alternative error function known as <b>perceptron criterian</b>. Here, we are looking for a weight vector $W$ which, for class $C_1$ satisfies $W^T\phi(X_n) > 0$ and for class $C_2$ satisfies $W^T\phi(X_n) < 0$. Hence, when combined together, it should satisfy $W^T\phi(X_n)t_n < 0$. The perceptron criterian associates $0$ error with the inputs that are correctly classified. For misclassified inputs, it tries to minimize $-W^T\phi(X_n)t_n$. The perceptron criterian is hence given as

$$\begin{align}
E_P(W) = -\sum_{n \in M} W^T\phi(X_n)t_n
\end{align}$$

where $M$ is the set consisting misclassified patterns. The contribution to the error associated with a particular misclassified pattern is a linear function of w in regions of w space where the pattern is misclassified and zero in regions where it is correctly classified. The total error function is therefore piecewise linear. Change in the weight vector $W$ is given as

$$\begin{align}
W^{(\tau+1)} = W^{(\tau)} -\eta\nabla E_P(W) = W^{(\tau)} + \eta\phi(X_n)t_n
\end{align}$$

where $\eta$ is called as <b>learning rate</b>.

The perceptron learning algorithm has a simple interpretation, as follows. We cycle through the training patterns in turn, and for each pattern $X_n$ we evaluate the perceptron function $y(X_n) = f(W^T\phi(X_n))$. If the pattern is correctly classified, then the weight vector remains unchanged, whereas if it is incorrectly classified, then for class $C_1$ (positive class) we add the vector $\phi(X_n)$ onto the current estimate of weight vector $W$ while for class $C_2$ we subtract the vector $\phi(X_n)$ from $W$. 

It should be noted that the change in weight vector at any step may have caused some previously correctly classified patterns to become misclassified. Thus the perceptron learning rule is not guaranteed to reduce the total error function at each stage. However, the <b>perceptron convergence theorem</b> states that if there exists an exact solution (in other words, if the training data set is linearly separable), then the perceptron learning algorithm is guaranteed to find an exact solution in a finite number of steps.

Another thing to note is that even when the data set is linearly separable, there may be many solutions, and which one is found will depend on the initialization of the parameters and on the order of presentation of the data points. Furthermore, for data sets that are not linearly separable, the perceptron learning algorithm will never converge.