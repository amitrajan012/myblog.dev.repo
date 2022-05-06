+++
date = "2022-04-14T14:07:28+01:00"
description = "Linear Algebra (Gilbert Strang): Chapter 18"
draft = false
tags = ["Linear Algebra", "Determinant", "Gilbert Strang", "Eigenvalues", "Eigenvectors"]
title = "Eigenvalues and Eigenvectors"
topics = ["Linear Algebra"]

+++


## 18.1 Eigenvalues and Eigenvectors

The usual function of a matrix is to act on a vector like a <b>function</b>, i.e. in goes a vector $x$, out comes a vector $Ax$. For a specific matrix $A$, if $Ax$ <b>is parallel to</b> $x$ i.e. $Ax = \lambda x$, the vectors $x$ are called as <b>eigenvectors</b>. The constant $\lambda$ is called as <b>eigenvalue</b>.

The vector $x$ and constant $\lambda$ satisfying the equation $Ax = \lambda x$ are <b>eigenvectors</b> and <b>eigenvalues</b>. The eigenvalues and eigenvectors for some of the commonly used matrices are as follows:

* <b>Projection Matrix:</b> A projection matrix $P$ takes a vector $b$ and projects it as $Pb$, where $Pb$ is in the plane of $P$. If the vector $x$ is in the plane of $P$, then it's projection $Px (=x)$ will be the same. This means that all the vectors in the plane of projection matrix $P$ are eigenvectors with eigenvalues equal to $\lambda = 1$. Apart from this, any $x \perp P$ will have a $0$ projection and hence an eigenvesctor with $\lambda = 0$ eigenvalue.


* <b>Permutation Matrix</b>: Let us take an example of a permutation matrix $A = \begin{bmatrix}
0 & 1 \\\\
1 & 0
\end{bmatrix}$, the eigenvalues and eigenvectors are $\lambda_1 = 1, x_1 = \begin{bmatrix}
1 \\\\
1
\end{bmatrix}; \lambda_2 = -1, x_2 = \begin{bmatrix}
-1 \\\\
1
\end{bmatrix}$.


* <b>Rotation Matrix</b>: Let $Q = \begin{bmatrix}
0 & -1 \\\\
1 & 0
\end{bmatrix}$, then $\begin{vmatrix}
-\lambda & -1 \\\\
1 & -\lambda
\end{vmatrix} = 0 \implies \lambda^2 + 1 = 0 \implies \lambda = \pm i$. Here eigenvalues are not real. It should be noted that the eigenvalues are complex conjugate of each other. Intutively, finding the eigenvectors of a rotation matrix is like finding vectors which when rotated by $90^\circ$ comes out to be the same. Only vector having this property is the $0$ vector.


* <b>Traiangular Matrix</b>: Let $Q = \begin{bmatrix}
3 & 1 \\\\
0 & 3
\end{bmatrix}$, then $\begin{vmatrix}
3-\lambda & 1 \\\\
0 & 3-\lambda
\end{vmatrix} = 0 \implies (3-\lambda)^2 = 0 \implies \lambda_1 = 3,\lambda_2 = 3$. We get $x_1 = x_2 = \begin{bmatrix}
1\\\\
0
\end{bmatrix}$. Hence, if we get repeated eigenvalues, we can have shortage of eigenvectors.


One of the most important fact about eigenvalue is: <b>Sum of the eigenvalues of a matrix is the sum of the elements across it's diagonal (called as trace) and the product of the eigenvalues is the value of the determinant</b>.

## 18.2 How to solve $Ax = \lambda x$

We can rewrite the equation $Ax = \lambda x$ as $(A - \lambda I)x = 0$. For this equation to have any other solution apart from $x=0$, the matrix $A - \lambda I$ has to be <b>singular</b>, i.e. $|A - \lambda I| = 0$. Now to find $x$ is like finding the <b>null space</b> of $(A - \lambda I)$.

<b>Example:</b> Take $A = \begin{bmatrix}
3 & 1 \\\\
1 & 3
\end{bmatrix}$. Then, $|A - \lambda I| = \begin{vmatrix}
3-\lambda & 1 \\\\
1 & 3-\lambda
\end{vmatrix} = (3-\lambda)^2 - 1 = \lambda^2 - 6\lambda + 8$. In the equation $\lambda^2 - 6\lambda + 8$, $6$ is the <b>trace (sum of diagonal elements)</b> and $8$ is the <b>determinant</b> of matrix $A$. On solving this equation, we get the eigenvalues as $\lambda_1 = 4,\lambda_2 = 2$. The eigenvector $x_1$ can be obtained by solving the equation $(A - \lambda_1I)x_1 = 0 \implies (A - 4I)x_1 = 0 \implies \begin{bmatrix}
-1 & 1 \\\\
1 & -1
\end{bmatrix}x_1 = 0 \implies x_1 = \begin{bmatrix}
1 \\\\
1 
\end{bmatrix}$. Similarly, for $\lambda_2$, $x_2 = \begin{bmatrix}
-1 \\\\
1 
\end{bmatrix}$.

If we look at the matrix in the example, it is $3I$ away from the permutation matrix discussed above. It should be noted that the eigenvalues of the new matrix $A$ can be obtained by adding $3$ in the eigenvalues of the permutation matrix with eigenvectors remaining the same. This fact can be demonstrated mathematically as well.

If $Ax = \lambda x$, then $(A + 3I)x = \lambda x + 3x = (\lambda + 3)x$. Hence for the new matrix $A + 3I$, eigenvalues are $\lambda + 3$ with eigenvectors remaining unchanged.
