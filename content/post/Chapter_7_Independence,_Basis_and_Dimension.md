+++
date = "2022-01-30T14:07:28+01:00"
description = "Linear Algebra (Gilbert Strang): Chapter 7"
draft = false
tags = ["Linear Algebra", "Matrix Multiplications", "Gilbert Strang",
"Vector Space", "Matrix Factorization", "Subspace", "Column Space", "Null Space", "Independence", "Span", "Basis", "Dimension"]
title = "Matrix Independence, Span, Basis & Dimension"
topics = ["Linear Algebra"]

+++

## 7.1 Independence
Vectors $x_1,x_2, ..., x_n$ are <b>linearly independent</b> if no combinations of these vectors gives zero vector, except the zero combination. i.e.



$$\begin{align}
c_1x_1 + c_2x_2 + ... + c_nx_n \neq 0; except \ \forall c_i=0
\end{align}$$

As we know that, for a $m \times n$ matrix $A$, if $m < n$, then we will have at least one non-zero solution to $Ax=0$. It implies that we will always find some non-zero combinations of $c_i$ that will make satisfy the above condition. This means that whenever we are in a m-dimensional space with $n$ vectors in it such that $m < n$, these $n$ vectors will always be <b>linearly dependent</b>.

In general, given the vectors $v_1, v_2, ..., v_n$, to check for their indepedence we can form a matrix $A$ with these vectors as the columns. Once the matrix is formed, we have to find the null-space of it (i.e. solution for $Ax=0$). If the <b>null-space of $A$ is just a zero vector, the columns of $A$ are independent</b>. We will not have any non-zero vector in the null-space of $A$ if and only there are not any <b>free columns</b> after the substitution steps. This means that all the columns are <b>pivot columns</b> and hence the <b>rank of matrix $A$ is $n$</b>. The columns in $A$ are dependent if there exist any <b>non-zero</b> $c$ such that $Ac=0$, which implies that there are some <b>free columns</b> in $A$, i.e. <b>rank of $A < n$</b>


## 7.2 Span, Basis and Dimension

Vectors $v_1, v_2, ..., v_l$ <b>spans a space</b> if the vector space consists of all combinations of these vectors. For example, columns of a matrix spans it's column space. The vectors which span a space can be dependent as well. For example, for a one dimnesional space $S_1$, we need at least one vector to span the entire space. But even if we have more that one vectors in that one-dimensional space, they will span the same $S_1$. This gives rise to basis.

<b>Basis</b> for a space is a sequence of vectors $v_1, v_2, ..., v_d$ which have following properties:
* They are independent.
* They span the space.

Example: For the space be $R^3$, one basis is $\begin{bmatrix}
    1 \\\\
    0 \\\\
    0
\end{bmatrix}, \begin{bmatrix}
    0 \\\\
    1 \\\\
    0
\end{bmatrix}, \begin{bmatrix}
    0 \\\\
    0 \\\\
    1
\end{bmatrix}$.

For a space $R^n$, $n$ vectors is a basis for it if the $n \times n$ matrix is <b>invertible</b>. Other fact about basis is: Given a space $S$, <b>every basis for the space has the same number of vectors and this number is the dimension of the space</b>.

Let us take an example. For the below matrix $A$, we have to find the basis for it's <b>column space</b> $C(A)$. 

$$\begin{align}
A = \begin{bmatrix}
    1 & 2 & 3 & 1 \\\\
    1 & 1 & 2 & 1 \\\\
    1 & 2 & 3 & 1
\end{bmatrix}
\end{align}$$

The first thing to notice is the fact that the columns of the matrix $A$ are not independent. This means that all the four columns do not form the basis for $C(A)$. If we start looking at the columns of $A$ from left to right the first two columns are linearly indepndent. Third column is the sum of the first and second and the fourth column is same as the first one. This makes $col_1, col_2$ as one of the basis of $C(A)$. It should also be noted that the rank of matrix $A$ is $2$. In a matrix $A$, <b>rank is equal to the number of pivot columns and the dimensions of $C(A)$</b>. i.e.

$$\begin{align}
dim(C(A)) = rank(A)
\end{align}$$

Similary, <b>the dimension of the null-space of $A$ is the number of free variables.</b> Hence, for a $m \times n$ matrix $A$, the dimension of null-space can be given as

$$\begin{align}
dim(N(A)) = n - rank(A)
\end{align}$$
