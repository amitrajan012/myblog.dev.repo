+++
date = "2022-07-01T23:07:28+01:00"
description = "Pattern Recognition (Bishop): Chapter 4"
draft = false
tags = ["Bishop", "Pattern Recognition", "Linear Models", "Classification", "Fisher’s Linear Discriminant", "Least Squares"]
title = "Linear Models for Clasification - Fisher’s Linear Discriminant"
topics = ["Pattern Recognition"]

+++

### 4.1.4 Fisher's Linear Discriminant

Linear classification model can be viewed as projecting the $D$-dimensional data onto a one-dimensional space. The equation $y=W^TX$ projects the $D$-dimensional input vector on a one dimensional space. Projection onto one dimension leads to a considerable loss of information and classes that are well-separated in the $D$-dimensional space may become overlapping in the one dimensional space. The goal of the classification problem is to <b>adjust the weight $W$ so that we can have the projection that maximizes the separation</b>.

Let us consider a two-class problem, where we have $N_1$ points for class $C_1$ and $N_2$ points for class $C_2$. The means for two classes are given as

$$\begin{align}
M_1 = \frac{1}{N_1}\sum_{n \in C_1}X_n ; M_2 = \frac{1}{N_2}\sum_{n \in C_2}X_n
\end{align}$$

One of the simplest measure of the separation of classes, when projected onto $W$, is the separation of projected class means. Hence, the choosen weight should maximize 

$$\begin{align}
m_2 - m_1 = W^TM_2 - W^TM_1 = W^T(M_2 - M_2)
\end{align}$$

where $m_k = W^TM_k$ is the projected class mean for $C_k$. This expression will keep on increasing as we increase the magnitude of $W$. Hence, we can add the constraint of $||W||^2 = 1$, i.e. $\sum_{i}W_i^2 = 1$. We can add this constraint and after solving the equation using Lagrange multiplier, we get $W \propto (M_2 - M_1)$. To have better separation, Fisher proposed an idea to <b>minimize within class variance</b>. Within-class variance of the transfomed data for class $C_k$ is given as 

$$\begin{align}
s_k^2 = \sum_{n \in C_k} (y_n - m_k)^2
\end{align}$$

where $m_k$ is the mean of the projected data and $y_n = W^TX_n$. The <b>Fisher criterian</b> is defined to be the ratio of the between-class variance to the within-class variance and is given as

$$\begin{align}
J(W) = \frac{(m_2 - m_1)^2}{s_1^2 + s_2^2}
\end{align}$$

As $m_2 - m_1 = W^T(M_2 - M_2)$, we have $(m_2 - m_1)^2 = (W^T(M_2 - M_2))(W^T(M_2 - M_2))^T = W^TS_BW$, where $S_B = (M_2 - M_2)(M_2 - M_2)^T$. The within-class variance can further be simplified as

$$\begin{align}
s_k^2 = \sum_{n \in C_k} (y_n - m_k)^2 = \sum_{n \in C_k} (W^TX_n - W^TM_k)^2
\end{align}$$

$$\begin{align}
= \sum_{n \in C_k} (W^TX_n - W^TM_k)(W^TX_n - W^TM_k)^T = W^T\bigg[\sum_{n \in C_k} (X_n - M_k)(X_n - M_k)^T\bigg]W
\end{align}$$

Hence, the Fisher criterian can be simplified as

$$\begin{align}
J(W) = \frac{(m_2 - m_1)^2}{s_1^2 + s_2^2} = \frac{W^TS_BW}{W^TS_WW}
\end{align}$$

where

$$\begin{align}
S_B = (M_2 - M_2)(M_2 - M_2)^T
\end{align}$$

$$\begin{align}
S_W = \sum_{n \in C_1} (X_n - M_1)(X_n - M_1)^T + \sum_{n \in C_2} (X_n - M_2)(X_n - M_2)^T
\end{align}$$

Differentiating $J(W)$ with respect to $W$ and computing it to $0$, we have

$$\begin{align}
W^TS_WW(S_BW) - W^TS_BW(S_WW) = 0
\end{align}$$

Solving this, we get

$$\begin{align}
W \propto S_W^{-1}(M_2 - M_1)
\end{align}$$

Fisher’s linear discriminant, although strictly it is not a discriminant but rather a specific choice of direction for projection of the data down to one dimension. However, the projected data can subsequently be used to construct a discriminant, by choosing a threshold $y_0$ so that we classify a
new point as belonging to $C_1$ if $y(X) \geq y_0$ and classify it as belonging to $C_2$ otherwise.

### 4.1.5 Relation to Least Squares

The least-squares approach to the determination of a linear discriminant was based on the goal of making the model predictions as close as possible to a set of target values. By contrast, the Fisher criterion was derived by requiring maximum class separation in the output space. For the two-class problem, the Fisher criterion can be obtained as a special case of least squares.

Insted of $1$-of-$K$ coding, if we use a different approach to encode the target variable, then the least squares solution for the weights become equivalent to to the Fisher solution. Let the new encoding be such that class $C_1$ as $N/N_1$ and $C_2$ as $-N/N_2$ where $N_1$ and $N_2$ are the number of patterns in class $C_1$ and $C_2$ respectively with $N$ being the total number of samples. The sum of square error function is

$$\begin{align}
E = \frac{1}{2}\sum_{n=1}^{N}(W^TX_n + W_0 - t_n)^2
\end{align}$$

Differentiating with respect to $W$ and $W_0$ and equating them to $0$, we have

$$\begin{align}
\sum_{n=1}^{N}(W^TX_n + W_0 - t_n)X_n = 0
\end{align}$$

$$\begin{align}
\sum_{n=1}^{N}(W^TX_n + W_0 - t_n) = 0
\end{align}$$

The sum of the target value in the new setting reduces to

$$\begin{align}
\sum_{n=1}^{N}t_n = N_1\frac{N}{N_1} - N_2\frac{N}{N_2} = 0
\end{align}$$

Using this, we have

$$\begin{align}
W_0 = -\frac{1}{N}\sum_{n=1}^{N}W^TX_n = -W^TM
\end{align}$$

where

$$\begin{align}
M = \frac{1}{N}\sum_{n=1}^{N}X_n = \frac{1}{N}(N_1M_1 + N_2M_2)
\end{align}$$

Solving for $W$, we get

$$\begin{align}
W \propto S_W^{-1}(M_2 - M_1)
\end{align}$$

Hence, the solution coincides with the Fisher criterian.