+++
date = "2022-03-17T14:07:28+01:00"
description = "Linear Algebra (Gilbert Strang): Chapter 9"
draft = false
tags = ["Linear Algebra", "Matrix Space", "Gilbert Strang",
"Vector Space", "Subspace", "Basis", "Dimension"]
title = "Matrix Spaces"
topics = ["Linear Algebra"]

+++

## 9.1 Matrix Spaces

The idea of vector spaces can be extended to matrices as well as far as they follow the following properties

* If $A \in S$ then $cA \in S$
* If $A \in S; B \in S$ then $A+B \in S$

It should be noted that the matrix multiplication doesn't need to belong to the same matrix space.

For a matrix $M$, some of the examples of matrix spaces are: <b>Upper Triangular Mtrices, Symmetric Matrices, Diagonal Matrices</b> etc. For a $3 \times 3$ matrix $M$, the basis can be given as ($dim(M) = 9)$:

$$\begin{align}
\begin{bmatrix}
    1 & 0 & 0 \\\\
    0 & 0 & 0 \\\\
    0 & 0 & 0
\end{bmatrix},
\begin{bmatrix}
    0 & 1 & 0 \\\\
    0 & 0 & 0 \\\\
    0 & 0 & 0
\end{bmatrix},
\begin{bmatrix}
    0 & 0 & 1 \\\\
    0 & 0 & 0 \\\\
    0 & 0 & 0
\end{bmatrix},...,
\begin{bmatrix}
    0 & 0 & 0 \\\\
    0 & 0 & 0 \\\\
    0 & 0 & 1
\end{bmatrix}
\end{align}$$

The dimension of the matrix space formed by <b>Symmetric Matrices</b> is 6 ($dim(S) = 6$) as we are free to choose all the 3 diagonal elements and the $3$ elements above it beacuse the three elements below the diagonal will be the same as the one above, making it's dimension as 6. 

Similary, the dimension of the matrix space formed by <b>Upper Triangular Matrices</b> is 6 ($dim(U) = 6$) as we are free to choose all the 3 diagonal elements and the $3$ elements above it beacuse the three elements below the diagonal are $0$, making it's dimension as 6. 

The intersection of $S$ and $U$ gives us the <b>Diagonal Matrix</b> and it's dimension will be $3$ as there are $3$ diagonal elements which can be chosen freely while all the other elements are 0.

## 9.2 Rank of Matrix Spaces:

One of the notable fact is that when we add two matrices, the rank of the matrix space formed by the resultant matrix will not exceed the sum of the ranks of the matrix spaces formed by indivdual matrices: $rank(A+B) \leq rank(A) + rank(B)$.
