+++
date = "2022-08-18T23:07:28+01:00"
description = "Pattern Recognition (Bishop): Chapter 7"
draft = false
tags = ["Bishop", "Pattern Recognition", "Kernel Methods", "Maximum Margin Classifiers", "Lagrange Multipliers", "Dual Representation", "KKT Condition", "Support Vectors"]
title = "Sparse Kernel Methods - Maximum Margin Classifiers"
topics = ["Pattern Recognition"]

+++


## 7.1 Maximum Margin Classifiers

Let us consider a two-class classification problem using linear models of the form

$$\begin{align}
y(X) = W^T\phi(X) + b
\end{align}$$

where $\phi(X)$ denotes a fixed feature-space transformation. Let the training data set comprises $N$ input vectors $X_1,X_2,...,X_N$, with corresponding target values $t_1,t_2,...,t_N$ where $t_n \in \{-1,1\}$, and new data points $X$ are classified according to the sign of $y(X)$. 

We shall assume for the moment that the training data set is linearly separable in feature space, so that by definition there exists at least one choice of the parameters $W$ and $b$ such that the linear function satisfies $y(X_n) > 0$ for points having $t_n = 1$ and $y(X_n) < 0$ for points having $t_n = -1$, so that $t_ny(X_n) > 0$ for all training data points.

There may exist many such solutions that separate the classes exactly. The perceptron algorithm is guaranteed to find a solution in a finite number of steps. The solution that it finds, however, will be
dependent on the (arbitrary) initial values chosen for $W$ and $b$ and as well as on the order in which the data points are presented. If there are multiple solutions all of which classify the training data set exactly, then we should try to find the one that will give the smallest generalization error. The support vector machine approaches this problem through the concept of the margin, which is defined to be the smallest distance between the decision boundary and any of the samples. In support vector machines the decision boundary is chosen to be the one for which the margin is maximized.

The perpendicular distance of a point $X$ from the hyperplane defined by $y(X) = W^T\phi(X) + b = 0$ is given as $|y(X)|/||W||$. We are interested in the solution for which all data points are correctly classified, i.e. $t_ny(X_n) > 0$ for all $n$. Thus the distance of point $X_n$ from the decision surface is given as

$$\begin{align}
\frac{t_n y(X_n)}{||W||} = \frac{t_n (W^T\phi(X_n) + b)}{||W||} 
\end{align}$$

The margin is given by the perpendicular distance to the closest point $X_n$ from the data set, and we wish to optimize the parameters $W$ and $b$ in order to maximize this distance. Thus the maximum margin solution is found by solving

$$\begin{align}
\arg \max_{W,b}\bigg[\frac{1}{||W||}\min_{n}[t_n (W^T\phi(X_n) + b)]\bigg]
\end{align}$$

The factor of $\frac{1}{||W||}$ can be taken outside optimization over $n$ as $W$ does not depend on $n$. One thing to note is that if we take $W \to kW$ and $b \to kb$, the distance from any point $X_n$ to the decision surface given by $t_n y(X_n)/||W||$ remains unchnaged. We can use this to set 

$$\begin{align}
t_n (W^T\phi(X_n) + b) = 1
\end{align}$$

for the point that is closest to the surface. In this case, all data points will satisfy

$$\begin{align}
t_n (W^T\phi(X_n) + b) \geq 1
\end{align}$$

for all $n=1,2,...,N$. This is known as the canonical representation of the decision hyperplane. In the
case of data points for which the equality holds, the constraints are said to be <b>active</b>, whereas for the remainder they are said to be <b>inactive</b>. By definition, there will always be at least one active constraint, because there will always be a closest point, and once the margin has been maximized there will be at least two active constraints. The optimization problem will simply require that we maximize $1/||W||$, which is equivalent to minimizing $||W||^2$ given the constraint $t_n (W^T\phi(X_n) + b) \geq 1$, i.e. we have to solve

$$\begin{align}
\arg \min_{W,b}\frac{1}{2}||W||^2
\end{align}$$

subjected to the constraint $t_n (W^T\phi(X_n) + b) \geq 1$. This is an example of a <b>quadratic programming</b> problem in which we are trying to minimize a quadratic function subject to a set of linear inequality constraints. The bias parameter $b$ has disappeared from the optimization. It is determined implicitly via the constraints, because these require that changes to $W$ be compensated by changes to $b$.

In order to solve this constrained optimization problem, we introduce <b>Lagrange multipliers</b> $a_n \geq 0$, with one multiplier for each of the constraints, giving the Lagrangian function

$$\begin{align}
L(W,b,a) = \frac{1}{2}||W||^2 - \sum_{n=1}^{N} a_n[t_n (W^T\phi(X_n) + b) - 1]
\end{align}$$

where $a = (a_1,a_2,...,a_N)^T$. Note the minus sign in front of the Lagrange multiplier
term, because we are minimizing with respect to $W$ and $b$, and maximizing with respect to $a$. Setting the derivatives of $L(W,b,a)$ with respect to $W$ and $b$ equal to zero, we obtain the following two conditions

$$\begin{align}
W = \sum_{n=1}^{N} a_n t_n \phi(X_n)
\end{align}$$

$$\begin{align}
0 = \sum_{n=1}^{N} a_n t_n
\end{align}$$

Eliminating $W,b$ from $L(W,b,a)$ using these conditions gives the <b>dual representation</b> of the maximum margin problem in which we maximize

$$\begin{align}
\tilde{L}(a) = \sum_{n=1}^{N} a_n -\frac{1}{2}\sum_{n=1}^{N}\sum_{m=1}^{N}a_n a_m t_n t_m k(X_n,X_m)
\end{align}$$

subject to the constraints

$$\begin{align}
a_n \geq 0
\end{align}$$

$$\begin{align}
\sum_{n=1}^{N} a_n t_n = 0
\end{align}$$

and the kernel function is defined as $k(X,X^{'}) = \phi(X)^T\phi(X^{'})$. In going to the dual formulation, we have turned the original optimization problem, which involved minimizing over $M$ varaiables, into the dual problem, which has $N$ variables. For a fixed set of basis functions whose number $M$ is smaller than the number $N$ of data points, the move to the dual problem appears disadvantageous. However, it allows the model to be reformulated using kernels, and so the maximum margin classifier can be applied efficiently to feature spaces whose dimensionality exceeds the number of data points, including infinite feature spaces.

In order to classify new data points using the trained model, we evaluate the sign of $y(X)$. This can be expressed in terms of the parameters $\{a_n\}$ and the kernel function by substituting for $W$ to give

$$\begin{align}
y(X) = \sum_{n=1}^{N} a_n t_n k(X,X_n) + b
\end{align}$$

The constraint optimization of the above form satisfies the <b>KKT condition</b> which in this case require that the following three properties hold

$$\begin{align}
a_n \geq 0
\end{align}$$

$$\begin{align}
t_n y(X_n) - 1 \geq 0
\end{align}$$

$$\begin{align}
a_n(t_n y(X_n) - 1) \geq 0
\end{align}$$

Hence for every data point, either $a_n = 0$ or $t_n y(X_n) = 1$. The data points for which $a_n = 0$ will not contribute to the sum $y(X) = \sum_{n=1}^{N} a_n t_n k(X,X_n) + b$ and hence play no role in making predictions for new data points. The remaining data points are called <b>support vectors</b>, and as they satisfy $t_n y(X_n) = 1$, they corresponds to the points that lie on the maximum margin hyperplanes in feature space. Once the model is trained, a significant proportion of the data points can be discarded and only the support vectors are retained.

After determining the value of $a = (a_1,a_2,...,a_N)^T$, the value of the threshold paramenter $b$ can be determined by noting that any support vector $X_n$ satisfies $t_n y(X_n) = 1$. This gives

$$\begin{align}
t_n y(X_n) = t_n \bigg(\sum_{m \in S} a_m t_m k(X_n,X_m) + b \bigg) = 1
\end{align}$$

where $S$ denotes the set of indices of support vectors. We can hence find the value of $b$ using an arbitrarily chosen support vector $X_n$. A more robust way to get the value of $b$ would be to find its value for all the support vectors and average it. Multiplying the above equation by $t_n$ and using $t_n^2 = 1$, we have

$$\begin{align}
b = t_n - \sum_{m \in S} a_m t_m k(X_n,X_m)
\end{align}$$

The above expression for $b$ is evaluated at the support vector $X_n$. Evaluating it for all the support vectors and taking the average, we have

$$\begin{align}
b = \frac{1}{N_S}\sum_{n \in S} \bigg[t_n - \sum_{m \in S} a_m t_m k(X_n,X_m)\bigg]
\end{align}$$

where $N_S$ is the total number of support vectors.

The maximum margin classifier can be expressed in terms of the minimization of an error function with a simple quadratic regularizer in the form

$$\begin{align}
\sum_{n=1}^{N}E_{\infty}(t_n y(X_n) - 1) + \lambda||W||^2
\end{align}$$

where $E_{\infty}(z)$ is a function which is $0$ when $z \geq 0$ and $\infty$ otherwise and hence ensures that the constraint $t_n y(X_n) \geq 1$ is satisfied.

From geometrical prespective, we can notice that the maximum margin hyperplane is defined by the location of the support vectors. Other data points can be moved around freely (so long as they remain outside the margin region) without changing the decision boundary, and so the solution will be independent of such data points which leads to the sparsity.

