+++
date = "2022-04-17T14:07:28+01:00"
description = "Linear Algebra (Gilbert Strang): Chapter 19"
draft = false
tags = ["Linear Algebra", "Determinant", "Gilbert Strang", "Eigenvalues", "Eigenvectors", "Eigenvalue Decomposition", "Power", "Difference Equation"]
title = "Diagonalization and Powers of a Matrix"
topics = ["Linear Algebra"]

+++


## 19.1 Diagonalization of a Matrix

Suppose for a given matrix $A$, we have $n$ <b>linearlly independent eigenvectros</b> and we put them in a matrix $S$. Then $AS = A \begin{bmatrix}
x_1 & x_2 & ... & x_n
\end{bmatrix}$, where each $x_i$ is the eigenvector. But for each of the eigenvectors, $Ax_i = \lambda_ix_i$. Hence, $AS = A \begin{bmatrix}
x_1 & x_2 & ... & x_n
\end{bmatrix} = \begin{bmatrix}
\lambda_1x_1 & \lambda_2x_2 & ... & \lambda_nx_n
\end{bmatrix} = \begin{bmatrix}
x_1 & x_2 & ... & x_n
\end{bmatrix}diag(\lambda_i) = S\Lambda$. Hence, $AS = S\Lambda$, where $\Lambda$ is a <b>diagonal matrix containing eigenvalues</b> and $S$ is the matrix of <b>eigenvectors as the columns</b>. If $S$ is invertible, i.e. the eigenvectors are independent, the equation can be rewritten as $S^{-1}AS = \Lambda$ or $A = S \Lambda S^{-1}$. 

## 19.2 Powers of a Matrix

Suppose the eigenvalues and eigenvectors of the matrix $A$ are $\lambda$ and $x$, i.e. $Ax = \lambda x$. Then, $A^2x = A\lambda x = \lambda Ax = \lambda\lambda x = \lambda^2 x$. Using the equation $A = S\Lambda S^{-1}$, we derive at the same conclusion, $A^2 = (S\Lambda S^{-1})^2 = S\Lambda S^{-1}S\Lambda S^{-1} = S \Lambda^2 S^{-1}$. Hence, $A^k = S \Lambda^k S^{-1}$. 

<b>Theorem:</b> $A^k \to 0$ as $k \to \infty$ if all $|\lambda_i| < 1$. The proof follows from the equation $A^k = S \Lambda^k S^{-1}$. As $K \to \infty$, $|\lambda_i|^k \to 0$. Hence, $\Lambda^k \to 0 \implies A^k \to 0$. This holds true only if the eigenvectors of $A$ are independent.

One thing to note is: <b>$A$ is sure to have $n$ independent eigenvectors if all of it's eigenvalues are different. For repeated eigenvalues, we may or may not have $n$ independent eigenvectors.</b> 

## 19.3 Solution of Difference Equation ($u_{k+1} = Au_k$)

Let us start with a vector $u_0$, then $u_1 = Au_0, u_2 = A^2u_1, ... ,u_k = A^ku_0$. Let us represent $u_0$ as the combinations of eigenvectors, i.e. $u_0 = c_1x_1 + c_2x_2 + ... + c_nx_n = Sc$. Then $Au_0 = c_1Ax_1 + c_2Ax_2 + ... + c_nAx_n = c_1\lambda_1x_1 + c_2\lambda_2x_2 + ... + c_n\lambda_nx_n = \Lambda Sc$. Similarly, $A^ku_0 = c_1\lambda_1^kx_1 + c_2\lambda_2^kx_2 + ... + c_n\lambda_n^kx_n = \Lambda^kSc$.

<b>Example:</b> Let us take an example of <b>Fibonacci Number</b>. They follow a second order difference equation $F_{k+2} = F_{k+1} + F_{k}$, where $F_0=0;F_1=1$. Let $u_k = \begin{bmatrix}
F_{k+1} \\\\
F_{k}
\end{bmatrix}$. If we consider two equations: $F_{k+2} = F_{k+1} + F_{k}$ and $F_{k+1} = F_{k+1}$, these two equations can be transformed as $u_{k+1} = \begin{bmatrix}
F_{k+2} \\\\
F_{k+1}
\end{bmatrix} = \begin{bmatrix}
1 & 1 \\\\
1 & 0
\end{bmatrix}\begin{bmatrix}
F_{k+1} \\\\
F_{k}
\end{bmatrix} = \begin{bmatrix}
1 & 1 \\\\
1 & 0
\end{bmatrix}u_k$. This transforms the problem into a first order difference equation. 

For $A = \begin{bmatrix}
1 & 1 \\\\
1 & 0
\end{bmatrix}$, $|A - \lambda I| = \begin{vmatrix}
1-\lambda & 1 \\\\
1 & -\lambda
\end{vmatrix} = \lambda^2 - \lambda - 1$. Hence, $\lambda = \frac{1 \pm \sqrt{5}}{2}$, i.e. $\lambda_1 = \frac{1 + \sqrt{5}}{2} \approx 1.618; \lambda_2 = \frac{1 - \sqrt{5}}{2} \approx -0.618$. The sequence of Fibonacci number is: 0,1,1,2,3,5,8,13,..., it should be noted that they are growing by a factor of the larger eigenvalue. The eigenvectors are: $x_1 = \begin{bmatrix}
\lambda_1 \\\\
1
\end{bmatrix}; x_2 = \begin{bmatrix}
\lambda_2 \\\\
1
\end{bmatrix}$. The starting vector $u_0$ can be given as $u_0 = \begin{bmatrix}
1 \\\\
0
\end{bmatrix}$. The vector $c$ can be found by solving $u_0 = c_1x_1 + c_2x_2$.
