+++
date = "2022-06-28T23:07:28+01:00"
description = "Pattern Recognition (Bishop): Chapter 4"
draft = false
tags = ["Bishop", "Pattern Recognition", "Linear Models", "Classification"]
title = "Linear Models for Clasification - Discriminant Functions"
topics = ["Pattern Recognition"]

+++


The goal of a calssification problem is to take the input vector $X$ and assign it to $K$ discrete classes $C_k$ where $k=1,2,3,...,K$. The input space is divided into <b>decision regions</b> whose boundaries are called as <b>decision boundaries</b> or <b>decision surfaces</b>. For linear models for classification, the decision surfaces are linear functions of the input vector $X$. Hence, for a $D$ -dimensional input space, decision surface will be a $(D-1)$ -dimensional hyperplane. Data sets whose classes can be separated exactly by linear decision surfaces are said to be <b>linearlly separable</b>.

For a two class classification problem, the target classes can be represented as $t=\{0,1\}$, where $t=1$ represents class $C_1$ and $t=0$ represents $C_2$. For $K>2$ classes it is convenient to use $1-of-K$ coding scheme in which $t$ is a vector of lenght $K$ where for class $C_j$ all the elements of $t_k$ of $t$ are zero except the element $t_j$, which is $1$. The value of the element $t_k$ can be interpreted as the probability the class is $C_k$.

There are multiple ways to achieve the classification task. One way to do is to construct a <b>discriminant function</b> which directlt assigns the input vector $X$ to one of the classes. Another approach is to model the conditional probability $p(C_k|X)$ in an <b>inference stage</b> and then use this distribution to make decisions. There are two different ways to model the conditional distribution $p(C_K|X)$ as well. One way to do it is model them directly using a parametric approach where the parameters are optimized using training set. Anothre way to do it is to adopt a <b>generative approach</b>, where we model the <b>class-conditional densities</b> $p(X|C_k)$ and the prior probabilities $p(C_k)$ and then use Bayes' theorem to obtain $p(C_k|X)$.

In linear regression models, the model prediction $y(X,W)$ was given as the linear function of the parameters $W$. In the simplest case, the model is also a linear function of input $X$ and hence the prediction takes the form $y(X,W) = W^TX + W_0$, where $y(X,W)$ is a real number. But in the classification problem, we need the probabilities as the output and hence the linear output $y(X,W)$ is transformed using a non-linear function $f(.)$ so that

$$\begin{align}
y(X,W) = f(W^TX+W_0)
\end{align}$$

The function $f$ is called as the <b>activation function</b> and its inverse is called as the <b>link function</b>. The decision surfaces correspond to $y(X,W) = const$ and hence they are linear in $X$ and $W$. However, the model is no longer linear in the parameters as we have a non-linear activation function. One thing to note is that even if we use a non-linear basis $\phi(X)$, all the properties of the model will remain intact with respect to $\phi(X)$ instead of $X$.