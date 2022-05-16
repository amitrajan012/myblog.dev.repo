+++
date = "2022-04-29T14:07:28+01:00"
description = "Linear Algebra (Gilbert Strang): Chapter 23"
draft = false
tags = ["Linear Algebra", "Determinant", "Gilbert Strang", "Eigenvalues", "Eigenvectors", "Complex Matrices", "Fourier Transform", "Hermitian Matrices"]
title = "Complex Matrices and Fourier Transform"
topics = ["Linear Algebra"]

+++

## 23.1 Complex Vectors/Matrices

Let $z = \begin{bmatrix}
z_1 \\\\
z_2 \\\\
... \\\\
z_n
\end{bmatrix}$ be a complex vector in $C^n$. The expresson $z^Tz$ doesn't represent the length of the vector $z$. Instead it's length is represented by $\overline{z}^Tz$. $\overline{z}^T$ is also called as $z^H$ and is called as <b>Hermitian of a matrix</b>. Hence, the length of a complex vector is $z^Hz = |z_1|^2 + |z_2|^2 + ... + |z_n|^2$. Similarly, for a complex matrix $A$ to be symmetric, $A^H = A$ with diagonal elements being real. In a complex domain, we call symmetric matrices as <b>Hermitian Matrices</b>. For two vectors $x$ and $y$ in a complex plane are perpendicular to each other if and only if $y^Hx = 0$. Hence, for an orrhogonal matrix (matrix with orthonormal columns) $Q$ in complex plane, $Q^HQ = I$. These orthogonal matrices in the complex plane are called as <b>Unitary Matrices</b>. 

## 23.2 Fast Fourier Transform

A $n \times n$ <b>Fourier Matrix</b> is shown below.

$$\begin{align}
F_n = \begin{bmatrix}
1 & 1 & 1 & ... & 1 \\\\
1 & w & w^2 & ... & w^{n-1} \\\\
1 & w^2 & w^4 & ... & w^{2(n-1)} \\\\
.. & .. & .. & .. & .. \\\\
1 & w^{n-1} & w^{2(n-1)} & ... & w^{(n-1)(n-1)}
\end{bmatrix}
\end{align}$$

The entry in the $i^{th}$ and $j^{th}$ cell where $i,j = 0,1,2,...,n-1$ is given as $(F_n)_{ij} = w^{ij}$. $w$ is a complex number such that $w^n=1$. Or to be more precise, $w = e^{i\frac{2\pi}{n}} = \cos \frac{2\pi}{n} + i \sin \frac{2\pi}{n}$. All the $w^k$ lies on a unit circle. For example, if $n=6$, $w = e^{i\frac{2\pi}{6}}$ lies at an angle of $60^{\circ}$ from the real-axis. Similary the other powers (for $n=6$) $w^2, w^3, w^4, w^5, w^6$ lie at an angle of $120^{\circ}, 180^{\circ}, 240^{\circ}, 300^{\circ}, 360^{\circ}$ from the real-axis. 

For $n=4$, $w=e^{i\frac{2\pi}{4}} = i$, $w^2 = -1$, $w^3 = -i$ and $w^4 = 1$. The fourier matrix is given as

$$\begin{align}
F_4 = \begin{bmatrix}
1 & 1 & 1 & 1 \\\\
1 & i & i^2 & i^3 \\\\
1 & i^2 & i^4 & i^6 \\\\
1 & i^3 & i^6 & i^9
\end{bmatrix} = \begin{bmatrix}
1 & 1 & 1 & 1 \\\\
1 & i & -1 & -i \\\\
1 & -1 & 1 & -1 \\\\
1 & -i & -1 & i
\end{bmatrix}
\end{align}$$

It should be noted that the columns of this matrix is orthogonal. All the columns have a length $2$ and hence to make the columns orthonormal, we can divide the matrix by $2$. As the columns being orthonormal, $F_4^H$ will be the inverse of $F_4$, i.e. $F_4^HF_4 = I$. For $n=k$ and $n=2k$, we have $w_k =  e^{i\frac{2\pi}{k}}$ and $w_{2k} =  e^{i\frac{2\pi}{2k}}$, i.e. $w_{2k}^2 = w_{k}$. Using this property any fourier matrix of size $2k$ can be written as

$$\begin{align}
F_{2k} = \begin{bmatrix}
I & D \\\\
I & -D
\end{bmatrix}\begin{bmatrix}
F_k & 0 \\\\
0 & F_k
\end{bmatrix}P
\end{align}$$

where $P$ is a <b>permutation matrix</b> and $D$ is a <b>diagonal matrix</b>. $F_k$ can be further reduced using recurrsion. Using this property multiplication of a $n \times n$ matrix will need $\frac{1}{2}n \log_2 n$ calculations instead of $n^2$.
