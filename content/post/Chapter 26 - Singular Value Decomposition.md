+++
date = "2022-05-09T14:07:28+01:00"
description = "Linear Algebra (Gilbert Strang): Chapter 26"
draft = false
tags = ["Linear Algebra", "Determinant", "Gilbert Strang", "Eigenvalues", "Eigenvectors", "Singular Value Decomposition"]
title = "Singular Value Decomposition"
topics = ["Linear Algebra"]

+++

## 26.1 Singular Value Decomposition

If any matrix $A$ can be decomposed as $A = U\Sigma V^T$, where $\Sigma$ is a <b>diagonal matrix</b> and $U,V$ are <b>orthogonal matrices</b>, this is called as <b>Singular Value Decomposition (SVD)</b>. One of the examples of SVD is for a symmetric positive definite matrix $A$, we know that $A = Q\Lambda Q^T$, where $\Lambda$ is diagonal and $Q$ is orthogonal.

For a matrix $m \times n$ matrix $A$, let the row-space be entire $\mathbb{R}^n$ and the column-space be entire $\mathbb{R}^m$. Any vector $v_1$ in the row-space can be transformed to a vector $u_1$ in the column-space as $u_1 = Av_1$. The goal of SVD is to find an <b>orthogonal basis in row-space which can be transformed to an orthogonal basis in column-space</b>. Let for a rank $r$ matrix $A$, the orthonormal basis (unit vectors) in the row-space consists of $v_1, v_2, ..., v_r$. Their transformations in the column-space are $\sigma_1u_1 = Av_1, \sigma_2u_2 = Av_2, ..., \sigma_ru_r = Av_r$ where $u_1, u_2, ..., u_r$ are also orthonormal (unit vectors and hence a factor of corresponding $\sigma$ for each $u$). In the matrix form, this can be represented as $AV=U\Sigma$ where $V$ and $U$ are the matrices whose columns are $v_1, v_2, ..., v_r$ and $u_1, u_2, ..., u_r$, and $\Sigma$ is a diagonal matrix of $\sigma_i$. Hence, $A = U\Sigma V^{-1} = U\Sigma V^{T}$ as $V$ has orthonormal columns. On further exploration, $A^TA = V\Sigma^{T}U^TU\Sigma V^T = V\Sigma^{T}(U^TU)\Sigma V^T = V\Sigma^{T}I\Sigma V^T = V\Sigma^{T}\Sigma V^T = V\Sigma^2V^T$, as $\Sigma$ is a daigonal matrix and hence $\Sigma^T \Sigma$ will have $\sigma_i^2$ at it's diagonal. <b>$A^TA = V\Sigma^2 V^T$ can be interpreted as symmetric positive definite matrix $A^TA$ decomposed into multiple of eigenvectors and eigenvalues, where $V$ is the eigenvector matrix of $A^TA$ and $\Sigma^2$ is the diagonal matrix having eigenvalues of $A^TA$</b>. Similarly, $AA^T = U\Sigma^2U^T$ and above mentioned facts hold for it as well. We can find $U,V$ and $\Sigma$ using this method.

<b>Example:</b> Let $A = \begin{bmatrix}
4 & 4 \\\\
-3 & 3
\end{bmatrix}$. Then $A^TA = \begin{bmatrix}
25 & 7 \\\\
7 & 25
\end{bmatrix}$ whose eigenvectors and eigenvalues are $\frac{1}{\sqrt{2}}\begin{bmatrix}
1 \\\\
1
\end{bmatrix},\frac{1}{\sqrt{2}}\begin{bmatrix}
1 \\\\
-1
\end{bmatrix}$ and $32,18$. Similarly, $AA^T = \begin{bmatrix}
32 & 0 \\\\
0 & 18
\end{bmatrix}$ whose eigenvectors and eigenvalues are $\begin{bmatrix}
1 \\\\
0
\end{bmatrix},\begin{bmatrix}
0 \\\\
1
\end{bmatrix}$ and $32,18$. Hence $A = \begin{bmatrix}
4 & 4 \\\\
-3 & 3
\end{bmatrix} = \begin{bmatrix}
1 & 0 \\\\
0 & 1
\end{bmatrix}\begin{bmatrix}
\sqrt{32} & 0 \\\\
0 & \sqrt{18}
\end{bmatrix}\begin{bmatrix}
1/\sqrt{2} & 1/\sqrt{2} \\\\
1/\sqrt{2} & -1/\sqrt{2}
\end{bmatrix}$.

<b>Example:</b> Let $A = \begin{bmatrix}
4 & 3 \\\\
8 & 6
\end{bmatrix}$ be a <b>singular matrix</b>. Row-space and column-space will be one dimensional as the rank of tha matrix is $1$. Hence, row-space just have multiple of $\begin{bmatrix}
4 \\\\
3
\end{bmatrix}$ and column-space just have multiple of $\begin{bmatrix}
4 \\\\
8
\end{bmatrix}=4\begin{bmatrix}
1 \\\\
2
\end{bmatrix}$. Hence $A = \begin{bmatrix}
4 & 3 \\\\
8 & 6
\end{bmatrix} = \frac{1}{\sqrt{5}}\begin{bmatrix}
1 & 2 \\\\
2 & -1
\end{bmatrix}\begin{bmatrix}
\sqrt{125} & 0 \\\\
0 & 0
\end{bmatrix}\begin{bmatrix}
0.8 & 0.6 \\\\
0.6 & -0.8
\end{bmatrix}$, where $\Sigma$ is calculated by finding the eigenvalue of $A^TA$.
