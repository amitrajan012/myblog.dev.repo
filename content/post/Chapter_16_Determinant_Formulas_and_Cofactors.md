+++
date = "2022-04-07T14:07:28+01:00"
description = "Linear Algebra (Gilbert Strang): Chapter 16"
draft = false
tags = ["Linear Algebra", "Determinant", "Gilbert Strang", "Cofactors"]
title = "Determinant and Cofactors"
topics = ["Linear Algebra"]

+++


## 16.1 Formula for $|A|$

For a $2 \times 2$ matrix $A$, the formula for $|A|$ can be derived as follows:

$$\begin{align}
|A| = \begin{vmatrix}
    a & b \\\\
    c & d
\end{vmatrix}=
\begin{vmatrix}
    a & 0 \\\\
    c & d
\end{vmatrix}+\begin{vmatrix}
    0 & b \\\\
    c & d
\end{vmatrix}
\end{align}$$

$$\begin{align}
=\begin{vmatrix}
    a & 0 \\\\
    c & 0
\end{vmatrix}+\begin{vmatrix}
    a & 0 \\\\
    0 & d
\end{vmatrix}+\begin{vmatrix}
    0 & b \\\\
    c & 0
\end{vmatrix}+\begin{vmatrix}
    0 & b \\\\
    0 & d
\end{vmatrix}
\end{align}$$

$$\begin{align}
=0+ad-bc+0=ad-bc
\end{align}$$

For a $3 \times 3$ matrix, first row can be seperated into $3$ pieces as demonstrated above. For each of the individual separated matrices, the second row will be separated into $3$ pieces giving a total of $9$ matrices. Finally, for each of these $9$ matrices, the third row will be separated into $3$ pieces, giving a total of $27$ matrices. Out of these $27$ matrices, a lot will have $0$ determinant. <b>Matrices with non-zero determinant will have one entry from each row and column</b>. The splitted matrices with non-zero determinant for a $3 \times 3$ matrix $A$ is shown below. The sign of individual determinants is derived based on number of row exchanges needed to get a diagonal matrix.

$$\begin{align}
|A| = \begin{vmatrix}
    a_{11} & a_{12} & a_{13}\\\\
    a_{21} & a_{22} & a_{23}\\\\
    a_{31} & a_{32} & a_{33}
\end{vmatrix}
\end{align}$$

$$\begin{align}
=\begin{vmatrix}
    a_{11} & 0 & 0\\\\
    0 & a_{22} & 0\\\\
    0 & 0 & a_{33}
\end{vmatrix}+\begin{vmatrix}
    a_{11} & 0 & 0\\\\
    0 & 0 & a_{23}\\\\
    0 & a_{32} & 0
\end{vmatrix}+\begin{vmatrix}
    0 & a_{12} & 0\\\\
    a_{21} & 0 & 0\\\\
    0 & 0 & a_{33}
\end{vmatrix}
\end{align}$$

$$\begin{align}
+\begin{vmatrix}
    0 & a_{12} & 0\\\\
    0 & 0 & a_{23}\\\\
    a_{31} & 0 & 0
\end{vmatrix}+\begin{vmatrix}
    0 & 0 & a_{13}\\\\
    a_{21} & 0 & 0\\\\
    0 & a_{32} & 0
\end{vmatrix}+\begin{vmatrix}
    0 & 0 & a_{13}\\\\
    0 & a_{22} & 0\\\\
    a_{31} & 0 & 0
\end{vmatrix}
\end{align}$$

$$\begin{align}
=a_{11}a_{22}a_{33} - a_{11}a_{23}a_{32} - a_{12}a_{21}a_{33}+ \\\\
a_{12}a_{23}a_{31} + a_{13}a_{21}a_{32} - a_{13}a_{22}a_{31}
\end{align}$$

The generic formula for a $n \times n$ matrix $A$ is as follows:

$$\begin{align}
|A| = \sum_{\text{n! terms}} \pm a_{1\alpha}a_{2\beta}...a_{n\omega}
\end{align}$$

$$\begin{align}
\text{such that}: (\alpha, \beta, ..., \omega) = \text{Permutation of}(1,2,3,...,n)
\end{align}$$

## 16.2 Cofactors

<b>Cofactor Formula</b> connects determinant of $n \times n$ matrix to the determinant of smaller matrix of size $n-1 \times n-1$. The determinant of the $3 \times 3$ matrix $A$ shown in cofactor format is as follows:

$$\begin{align}
|A| = a_{11}(a_{22}a_{33} - a_{23}a_{32}) + a_{12}(-a_{21}a_{33} + 
a_{23}a_{31}) + a_{13}(a_{21}a_{32} - a_{22}a_{31})
\end{align}$$

The above determinant formula can be seen as the combination of following permutations.

$$\begin{align}
\begin{vmatrix}
    a_{11} & 0 & 0\\\\
    0 & a_{22} & a_{23}\\\\
    0 & a_{32} & a_{33}
\end{vmatrix};\begin{vmatrix}
    0 & a_{12} & 0\\\\
    a_{21} & 0 & a_{23}\\\\
    a_{31} & 0 & a_{33}
\end{vmatrix};\begin{vmatrix}
    0 & 0 & a_{13}\\\\
    a_{21} & a_{22} & 0\\\\
    a_{31} & a_{32} & 0
\end{vmatrix}
\end{align}$$

$$\begin{align}
\text{Cofactor of } a_{ij} = C_{ij} = (-1)^{i+j}|n-1 \text{ matrix with row } i \text{ col } j \text{ erased}|
\end{align}$$

<b>Cofactors without the sign are called Minors</b>. Hence,

$$\begin{align}
|A| = a_{11}C_{11} + a_{12}C_{12} + ... + a_{1n}C_{1n}
\end{align}$$
