+++
date = "2022-05-06T14:07:28+01:00"
description = "Linear Algebra (Gilbert Strang): Chapter 25"
draft = false
tags = ["Linear Algebra", "Determinant", "Gilbert Strang", "Eigenvalues", "Eigenvectors", "Similar Matrices", "Jordan Block"]
title = "Similar Matrices"
topics = ["Linear Algebra"]

+++

## 25.1 Similar Matrices

One of the major generator of a <b>positive definite</b> matrix is when we square one. In a nutsheel, the matrix $A^TA$ is always positive definite. To prove this, let $A$ be a $m \times n$ rectangular matrix. Then $A^TA$ is a square symmetric matrix. If we evaluate the expression $x^T(A^TA)x$, we get $x^T(A^TA)x = (x^TA^T)(Ax) = (Ax)^T(Ax) = |Ax|^2 > 0$ if $Ax$ is non-zero. Another important thing to note is if matrix $A, B$ are positive definite, $A+B$ is positive definite.

Matrices $A$ and $B$ are <b>similar</b> if for some invertible matrix $M$, $B=M^{-1}AM$. One of the examples is the eigen value matrix $\Lambda$. As we know that for any matrix $A$, $S^{-1}AS = \Lambda$. This means <b>$A$ is similar to $\Lambda$</b> where $M=S$. Let's take an example, where

$$\begin{align}
A = \begin{bmatrix}
2 & 1 \\\\
1 & 2
\end{bmatrix};\Lambda = \begin{bmatrix}
3 & 0 \\\\
0 & 1
\end{bmatrix}
\end{align}$$

We can find a lot of values of $M$ and hence $B$ and one of it is as follows:

$$\begin{align}
\begin{bmatrix}
1 & -4 \\\\
0 & 1
\end{bmatrix}\begin{bmatrix}
2 & 1 \\\\
1 & 2
\end{bmatrix}\begin{bmatrix}
1 & 4 \\\\
0 & 1
\end{bmatrix}=\begin{bmatrix}
-2 & -15 \\\\
1 & 6
\end{bmatrix} = B
\end{align}$$

<b>The most important fact about similar matrices is that they have the same eigenvalues</b>. 

## 25.2 Why Similar Matrices have same Eigenvalues?

Let $A$ and $B$ are similar matrices and $\lambda$ is one of the eigen values of $A$. Then, $Ax = \lambda x$ and $B = M^{-1}AM$. We can write $Ax$ as $Ax = AIx = AMM^{-1}x = \lambda x$. Multiplying by $M^{-1}$ on the both side, we get $M^{-1}AMM^{-1}x = \lambda M^{-1}x \implies (M^{-1}AM)M^{-1}x = \lambda M^{-1}x \implies BM^{-1}x = \lambda M^{-1}x \implies B(M^{-1}x) = \lambda (M^{-1}x)$. Hence $\lambda$ is alse an eigenvalue of $B$ with $M^{-1}x$ being it's eigenvector. Hence, <b>Similar Matrices have the same eigenvalues with the eigenvectors moved around</b>.

## 25.3 Non-diagonalizable Matrix and Jordan Form

Let the eigenvalues of $A$ are same. For example, let us say $\lambda_1 = \lambda_2 = 4$. In this case, <b>one family of similar matrices</b> is just the matrix $\begin{bmatrix}
4 & 0 \\\\
0 & 4
\end{bmatrix}$ as for all $M$, $M^{-1}\begin{bmatrix}
4 & 0 \\\\
0 & 4
\end{bmatrix}M = \begin{bmatrix}
4 & 0 \\\\
0 & 4
\end{bmatrix}$.

</b>Another family of similar matrices</b> will have matrices in the form of $\begin{bmatrix}
4 & 1 \\\\
0 & 4
\end{bmatrix}, \begin{bmatrix}
4 & 0 \\\\
17 & 4
\end{bmatrix}, \begin{bmatrix}
5 & 1 \\\\
-1 & 3
\end{bmatrix}$, where the first matrix representing the simplest (the best we can get to a diagonal matrix) matrix in the family. This matrix is called in the <b>Jordan Form</b>. A matrix in the Jordan Form can be divided into <b>Jordan Blocks</b> and each Jordan Block has one eigenvalue in it. In a nutshell, <b>every square matrix $A$ is similar to a Jordan Matrix $J$ with $n$ Jordan Blocks where $n$ is the number of eigenvalues of $A$</b>, as we have one eigenvalue per Jordan Block. When the matrix $A$ is diagonalizable, i.e. it has distinct eigenvalues, then $J = \Lambda$.


