+++
date = "2022-01-08T14:07:28+01:00"
description = "Linear Algebra (Gilbert Strang): Chapter 2"
draft = false
tags = ["Linear Algebra", "Linear Equations", "Matrix Multiplications", "Gilbert Strang"]
title = "Elimination & Permutation with Matrices"
topics = ["Linear Algebra"]

+++

## 2.1 Elimination & Permutation with Matrices

Elimination is the method which is used to solve a system of linear equations. Let $Ax=b$ is a system of linear equation where 


$$\begin{align}
A = \begin{bmatrix}
    1 & 2 & 1\\\\
    3 & 8 & 1\\\\
    0 & 4 & 1
\end{bmatrix},
b = \begin{bmatrix}
    2 \\\\
    12 \\\\
    2
\end{bmatrix}
\end{align}$$

The idea of elimination is to transform $A$ into a matrix where all the entries in the cell below the diagonal is 0. The transformed matrix is called as <b>upper triangular matrix</b> (denoted by $U$). Once we have the upper triangular matrix, the system of equation can be easly solved using back substitution. 

To get $U$, certain elimination steps are followed. Usually the right hand side $b$ is appended to $A$ such that all the elimination steps are performed on them simultaneously. The appended matrix is shown below.

$$\begin{align}
A^{'} = \left[
\begin{array}{ccc|c}
1 & 2 & 1 & 2 \\\\
3 & 8 & 1 & 12 \\\\
0 & 4 & 1 & 12 \\\\
\end{array}
\right]
\end{align}$$

The usual elimination step which is used to make the entry of <b>row m column n</b> is denoted as $E_{mn}$. The two primaraly used elimination steps are:
* <b>row multiplication and subtraction:</b> In this step, a row is multiplied by a number and subtracted from another row
* <b>row exchange:</b> In this step, two rows are exchanged

The elimination steps for the matrix $A$ are as follows.
* <b>Step 1 - $E_{21}$</b>: $row_2 \leftarrow row_2 - 3row_1$

$$\begin{align}
A = \begin{bmatrix}
    1 & 2 & 1\\\\
    3 & 8 & 1\\\\
    0 & 4 & 1
\end{bmatrix} \rightarrow
\begin{bmatrix}
    1 & 2 & 1\\\\
    0 & 2 & -2\\\\
    0 & 4 & 1
\end{bmatrix}
\end{align}$$

* <b>Step 2 - $E_{32}$</b>: $row_3 \leftarrow row_3 - 2row_2$

$$\begin{align}
A = \begin{bmatrix}
    1 & 2 & 1\\\\
    3 & 8 & 1\\\\
    0 & 4 & 1
\end{bmatrix} \rightarrow
\begin{bmatrix}
    1 & 2 & 1\\\\
    0 & 2 & -2\\\\
    0 & 4 & 1
\end{bmatrix} \rightarrow
U = \begin{bmatrix}
    1 & 2 & 1\\\\
    0 & 2 & -2\\\\
    0 & 0 & 5
\end{bmatrix}
\end{align}$$

Let us look at these elimination steps individually. The step $E_{21}$ leaves $row_{1}, row_{3}$ unchanged. The new $row_{2}$ is obtained by adding $-3$ times $row_1$ to existing $row_2$. If we look at the matrix multiplication as row operations, the new $row_2$ can be achieved by multiplying $A$ with $\begin{bmatrix}-3 & 1 & 0 \end{bmatrix}$ as

$$\begin{align}
\begin{bmatrix}
-3 & 1 & 0
\end{bmatrix} A = 
\begin{bmatrix}
-3 & 1 & 0
\end{bmatrix}
\begin{bmatrix}
    1 & 2 & 1\\\\
    3 & 8 & 1\\\\
    0 & 4 & 1
\end{bmatrix} \end{align}$$
$$\begin{align}= -3\begin{bmatrix}
1 & 2 & 1
\end{bmatrix}
+1\begin{bmatrix}
3 & 8 & 1
\end{bmatrix}
+0\begin{bmatrix}
0 & 4 & 1
\end{bmatrix}
=\begin{bmatrix}
0 & 2 & -2
\end{bmatrix}
\end{align}$$

$row_1$ and $row_2$ can remain unchanged by multiplying $A$ with $\begin{bmatrix}1 & 0 & 0 \end{bmatrix}$ and $\begin{bmatrix}0 & 0 & 1 \end{bmatrix}$ respectively. Hence, elimination step $E_{21}$ can be written as matrix multiplication in the form of $A \rightarrow E_{21}A$, where

$$\begin{align}
E_{21} = 
\begin{bmatrix}
    1 & 0 & 0\\\\
    -3 & 1 & 0\\\\
    0 & 0 & 1
\end{bmatrix}
\end{align}$$

Similarly, $E_{32}$ in which $row_1, row_2$ are unchanged and new $row_3$ is obtained by subtracting twice $row_2$ from $row_3$ can be written as $E_{21}A \rightarrow E_{32}E_{21}A$, where

$$\begin{align}
E_{32} = 
\begin{bmatrix}
    1 & 0 & 0\\\\
    0 & 1 & 0\\\\
    0 & -2 & 1
\end{bmatrix}
\end{align}$$

The entire transformation can be represented as where $E = E_{32}E_{21}$ is the transformation matrix.

$$\begin{align}
A \rightarrow E_{32}(E_{21}A) = (E_{32}E_{21})A = EA = U 
\end{align}$$

The above explained transformation steps come under the category of <b>row multiplication and subtraction</b>. There may arise a situation when to achieve an <b>upper triangular matirx</b>, we have to <b>substitute the rows</b>. For example, let us look at the following matrix.

$$\begin{align}
A = \begin{bmatrix}
    0 & 2\\\\
    1 & 8\\\\
\end{bmatrix}
\end{align}$$

To get the upper triangular matrix from $A$, no possible row multiplication and subtraction steps can be thought of. The only way to do the transformation is to replace $row_{1}$ and $row_{2}$. Hence the elimination step for the matrix $A$ is as follows.

* <b>Step 1 -</b> $P: row_{1} \leftrightarrow row_{2}$

$$\begin{align}
A = \begin{bmatrix}
    0 & 2\\\\
    1 & 8\\\\
\end{bmatrix} \rightarrow
U = \begin{bmatrix}
    1 & 8\\\\
    0 & 2\\\\
\end{bmatrix}
\end{align}$$



The above <b>row substitution</b> operation can be achieved by using a special class of matrices called <b>permutation matrices</b> for multiplication. By using matrix multiplication as a row operation, the above transformation can be achieved using multiplying $A$ by below matrix.

$$\begin{align}
P = \begin{bmatrix}
    0 & 1\\\\
    1 & 0\\\\
\end{bmatrix}
\end{align}$$

If we look at the row wise multiplication further, the new $row_1$ is:

$$\begin{align}
\begin{bmatrix}
    0 & 1
\end{bmatrix}
\begin{bmatrix}
    0 & 2\\\\
    1 & 8\\\\
\end{bmatrix} = 
0\begin{bmatrix}
    0 & 2
\end{bmatrix} +
1\begin{bmatrix}
    1 & 8
\end{bmatrix} =
\begin{bmatrix}
    1 & 8
\end{bmatrix}
\end{align}$$

For a matrix which has $n$ rows, there can be a total of $n!$ <b>permutation matrices</b> as $n$ rows can be arranged in a total of $n!$ ways. For example, for a $2 \times 2$ matrix the set of permutation matrices are as follows. The first matrix belongs to the case when no row substitution takes place and the second matrix belongs to the case when $row_1$ and $row_2$ are interchanged. It should be noted that different matrices in the set of permutation matrices are constructed by <b>exchanging the rows of Identity Matrix</b>.

$$\begin{align}
\bigg\\{\begin{bmatrix}
    1 & 0\\\\
    0 & 1\\\\
\end{bmatrix},
\begin{bmatrix}
    0 & 1\\\\
    1 & 0\\\\
\end{bmatrix} \bigg\\}
\end{align}$$
