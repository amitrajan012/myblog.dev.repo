+++
date = "2022-01-12T14:07:28+01:00"
description = "Linear Algebra (Gilbert Strang): Chapter 3"
draft = false
tags = ["Linear Algebra", "Linear Equations", "Matrix Multiplications", "Gilbert Strang",
"Inverse Matrix", "Matrix Factorization"]
title = "Inverse of a Matrix & Factorization into $A=LU$"
topics = ["Linear Algebra"]

+++

## 3.1 Inverse of a Matrix & Factorization into $A=LU$

Given a <b>square matrix</b> $A$, $A^{-1}$ is it's inverse if $AA^{-1} = A^{-1}A = I$, where $I$ is the <b>Identity Matrix</b> and we say that $A$ is <b>invertible</b> or <b>nonsingular</b>.

For a <b>singular</b> matrix $A$, the inverse does not exist and it's <b>determinant</b> is $0$. Also, we can find a vector $x$ such that $Ax=0$. This means that one or more than one columns of $A$ is linear combination of other columns. For example, below $2 \times 2$ matrix is not invertible.

$$\begin{align}
\begin{bmatrix}
    1 & 2\\\\
    2 & 4\\\\
\end{bmatrix}
\begin{bmatrix}
    2 \\\\
    -1 
\end{bmatrix}=
\begin{bmatrix}
    0 \\\\
    0 
\end{bmatrix}
\end{align}$$

The inverse of the product of two matrices given as $AB$ is $B^{-1}A^{-1}$. This can be verified as follows. Let $M$ be the inverse of $AB$.
$$\begin{align}
M(AB) = I \\\\
M(AB)B^{-1} = IB^{-1} \\\\
MA = B^{-1} \\\\
MAA^{-1} = B^{-1}A^{-1} \\\\
M = B^{-1}A^{-1}
\end{align}$$

Multiplication of two matrices can be viewed as performing an operation on second matrix based on the first. For example, for the row multiplication and subtraction elimination step, the operation is encoded as an elimination matrix. Elimination matrix $E_{21}$ for value at $row \ 2, column \ 1$ with the corresponding origianl matrix $A$ and the result $E_{21}A$ is shown below.

$$\begin{align}
A = \begin{bmatrix}
    1 & 2\\\\
    3 & 8\\
\end{bmatrix},
E_{21} = \begin{bmatrix}
    1 & 0\\\\
    -3 & 1\\
\end{bmatrix},
E_{21}A = \begin{bmatrix}
    1 & 2\\\\
    0 & 2\\
\end{bmatrix}
\end{align}$$

Let us say we want to reverse this operation. This means that we want to get matrix $A$ from $E_{21}A$. As the resultant matrix is obtained by the row operation $row_2 \leftarrow row_2 - 3row_1$, the operation can be reversed if $3$ times $row_1$ is added back to $row_2$, i.e, $row_2 \leftarrow row_2 + 3row_1$. This operation can be achieved by the below multiplication matrix.

$$\begin{align}
E_{21}^{-1} = \begin{bmatrix}
    1 & 0\\\\
    3 & 1\\
\end{bmatrix}
\end{align}$$

It should be noted that $E_{21}^{-1}E_{21}=I$ and $E_{21}^{-1}$ is called the <b>inverse</b> of $E_{21}$.

$$\begin{align}
E_{21}^{-1}E_{21} = \begin{bmatrix}
    1 & 0\\\\
    3 & 1\\
\end{bmatrix}
\begin{bmatrix}
    1 & 0\\\\
    -3 & 1\\
\end{bmatrix}=
\begin{bmatrix}
    1 & 0\\\\
    0 & 1\\
\end{bmatrix} = I
\end{align}$$

The resultant matrix $E_{21}A = U$ is an <b>upper triangular matrix</b>. If we multiply $U$ with $E_{21}^{-1}$, we will get back $A$, i.e. $E_{21}^{-1}(E_{21}A) = E_{21}^{-1}U = A$. Here, the matrix $E_{21}^{-1}=L$ is a <b>lower triangular matrix</b>. This process of factoring matrix $A$ into $LU$ is shown below.

$$\begin{align}
A = LU \implies
\begin{bmatrix}
    1 & 2\\\\
    3 & 8\\
\end{bmatrix} = 
\begin{bmatrix}
    1 & 0\\\\
    3 & 1\\
\end{bmatrix}
\begin{bmatrix}
    1 & 2\\\\
    0 & 2\\
\end{bmatrix}
\end{align}$$

Another thing to notice is that all the diagonal elements of $L$ is $1$ but that of $U$, they are not. To have all the diagonal entries in $U$ as $1$, the matrix can be further factorized as follows. Matrix $D$ is a <b>diagonal matrix</b> where diagonals (<b>also called as pivots</b>) are the ones from $U$.

$$\begin{align}
A = LU = LDU^{'}\implies
\begin{bmatrix}
    1 & 2\\\\
    3 & 8\\
\end{bmatrix} = 
\begin{bmatrix}
    1 & 0\\\\
    3 & 1\\
\end{bmatrix}
\begin{bmatrix}
    1 & 0\\\\
    0 & 2\\
\end{bmatrix}
\begin{bmatrix}
    1 & 2\\\\
    0 & 1\\
\end{bmatrix}
\end{align}$$

## 3.2 Permutation Matrix: Inverse and Factorization 

<b>Permutation Matrices</b> are the class of matrices which are needed for the <b>row exchange operation</b>. In some of the cases, to get a <b>upper triangular matrix</b> from a matrix, some row exchange operations can also be needed. Let us look at the following transformation step for matrix $A$.

$$\begin{align}
A =
\begin{bmatrix}
    0 & 2 & 1\\\\
    1 & 2 & 3\\\\
    1 & 6 & 3
\end{bmatrix}
\xrightarrow{\text{P}}
\begin{bmatrix}
    1 & 2 & 3\\\\
    0 & 2 & 1\\\\
    1 & 6 & 3
\end{bmatrix}
\xrightarrow{\text{E}}
\begin{bmatrix}
    1 & 2 & 3\\\\
    0 & 2 & 1\\\\
    0 & 0 & -2
\end{bmatrix} = U
\end{align}$$
 
The steps are: $P \equiv row_2 \leftrightarrow row_1$, $E \equiv row_3 \leftarrow row_3 - row_1 - 2row_2$. The <b>permutation matrix</b> $P$ which is nothing but <b>the identity matrix with reordered rows</b> for the above case is shown below.

$$\begin{align}
P =
\begin{bmatrix}
    0 & 1 & 0\\\\
    1 & 0 & 0\\\\
    0 & 0 & 1
\end{bmatrix}
\end{align}$$

<b>All the permutation matrices are invertible</b> with $P^{-1} = P^T$. With row exchange as one of the elimination steps, the factorization $A = LU$ is transformed as $PA = LU$.

## 3.3 Symmetric Matrices
There is one more special class of matrices called as <b>symmetric matrices</b>. For a symmetric matirx $A$, $A^T = A$. This means that the entry in cell $(i,j)$ is the same as entry in cell $(j,i)$ for $A$ as $A^T_{i,j} = A_{j,i}$. Apart from this, <b>for any matrix $R$, $R^TR$ is symmetric</b>. This can be easily seen as:$(R^TR)^T = R^T(R^T)^T = R^TR$, which implies that the <b>transpose of $R^TR$ is same as $R^TR$ and hence it is symmetric</b>. Another way to look at it is by matrix multiplication. First of all, <b>the transpose of a matrix exchanges it's rows and columns</b>, i.e. $i^{th}$ row of original matrix is the $j^{th}$ column of the transpose matrix. For example,

$$\begin{align}
R =
\begin{bmatrix}
    1 & 3\\\\
    2 & 3\\\\
    4 & 1
\end{bmatrix},
R^T = 
\begin{bmatrix}
    1 & 2 & 4\\\\
    3 & 3 & 1
\end{bmatrix}
\end{align}$$

Let us now evaluate $R^TR$ using <b>row times column method</b> of mtrix multiplication. The entries in the resultant matrix $S$ can be viewed as $S_{12} = R^T_{row_1} \times R_{col_2}$ and $S_{21} = R^T_{row_2} \times R_{col_1}$. But as $R^T_{row_1} = R_{col_1}$ and $R^T_{row_2} = R_{col_2}$ which gives us $S_{12} = S_{21}$.

$$\begin{align}
R^TR =
\begin{bmatrix}
    1 & 2 & 4\\\\
    3 & 3 & 1
\end{bmatrix}
\begin{bmatrix}
    1 & 3\\\\
    2 & 3\\\\
    4 & 1
\end{bmatrix}=
\begin{bmatrix}
    21 & 13\\\\
    13 & 19
\end{bmatrix} = S
\end{align}$$
