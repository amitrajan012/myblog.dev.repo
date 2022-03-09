+++
date = "2022-01-20T14:07:28+01:00"
description = "Linear Algebra (Gilbert Strang): Chapter 5"
draft = false
tags = ["Linear Algebra", "Matrix Multiplications", "Gilbert Strang",
"Vector Space", "Matrix Factorization", "Subspace", "Column Space", "Null Space", "Reduced Row-Echelon Form"]
title = "Algorithm for solving $Ax=0$"
topics = ["Linear Algebra"]

+++

### 5.1 Algorithm for solving $Ax=0$

The idea behind this exercise is to come up with an algorithm to find the nullspace of a matrix $A$. Let us start by taking a matrix $A$.

$$\begin{align}
A = \begin{bmatrix}
    1 & 2 & 2 & 2 \\\\
    2 & 4 & 6 & 8 \\\\
    3 & 6 & 8 & 10
\end{bmatrix}
\end{align}$$

One of the first thing to notice in $A$ is that $row_3$ is a linear combination of $row_1$ and $row_2$ ($row_3 = row_1 + row_2$). The way to solve a system of linear equations is by elimination. Different <b>row elimination steps</b> can be used to transform $A$ into an <b>upper triangular matrix</b> with the same steps repeated on the right hand side. As the right hand side is $0$ here, the elimination steps can be skipped for it. 

$$\begin{align}
A = \begin{bmatrix}
    1 & 2 & 2 & 2 \\\\
    2 & 4 & 6 & 8 \\\\
    3 & 6 & 8 & 10
\end{bmatrix}
\xrightarrow{\text{S1}}
\begin{bmatrix}
    1 & 2 & 2 & 2 \\\\
    0 & 0 & 2 & 4 \\\\
    0 & 0 & 2 & 4
\end{bmatrix}
\xrightarrow{\text{S3}}
\begin{bmatrix}
    1 & 2 & 2 & 2 \\\\
    0 & 0 & 2 & 4 \\\\
    0 & 0 & 0 & 0
\end{bmatrix} = U
\end{align}$$

The resultant matrix after the elimination steps is $U$ which is in the <b>echelon form</b>. Now, instead of solving $Ax=0$, $Ux=0$ can be solved. There are <b>two pivots</b>: $U_{11} = 1,U_{23} = 2$. The number of pivots is also called as the <b>rank of the matrix</b>. Here, the rank is $2$. The columns in which pivots are there ($col_1, col_3$) are called as <b>pivot columns</b>. Remaining columns ($col_2, col_4$) are called as <b>free columns</b>. The equation $Ux=0$ can be written as follows.

$$\begin{align}
x_1\begin{bmatrix}
    1  \\\\
    0  \\\\
    0
\end{bmatrix} +
x_2\begin{bmatrix}
    2  \\\\
    0  \\\\
    0
\end{bmatrix} +
x_3\begin{bmatrix}
    2  \\\\
    2  \\\\
    0
\end{bmatrix} +
x_4\begin{bmatrix}
    2  \\\\
    4  \\\\
    0
\end{bmatrix} = 
\begin{bmatrix}
    0  \\\\
    0  \\\\
    0
\end{bmatrix}
\end{align}$$

The values in $x$ corresponding to the free columns (here they are $x_2,x_4$) can be assigned any values and then we can find the values of $x_1,x_3$ to complete the solution. We can systematically assign values to the multipliers corresponding to the free columns. Let $x_2=1,x_4=0$, this gives $x_1=-2,x_3=0$ and hence $\begin{bmatrix}
    -2 \\\\
    1 \\\\
    0 \\\\
    0
\end{bmatrix}$ is a solution (or any multiple of this vector), a vector in the <b>null space</b>. Another choice for free variables can be $x_2=0,x_4=1$, which gives $x_1=2,x_3=-2$ and hence $\begin{bmatrix}
    2 \\\\
    0 \\\\
    -2 \\\\
    1
\end{bmatrix}$ is another solution (or any multiple of this vector), another vector in the <b>null space</b>. Hence, the general solution can be given as:

$$\begin{align}
x = c\begin{bmatrix}
    -2  \\\\
    1  \\\\
    0 \\\\
    0
\end{bmatrix} +
d\begin{bmatrix}
    2  \\\\
    0  \\\\
    -2 \\\\
    1
\end{bmatrix}
\end{align}$$

Another thing to note is the number of individual solutions is equal to the number of <b>free variables</b> or <b>free columns</b>. For a $m \times n$ matrix whose <b>rank</b> is $r$, the number of <b>free columns/variables</b>( individual solutions) will be $n-r$.

## 5.2 Reduced Row-Echelon Form
 The row-echelon matrix $U$ can be further simplified by elimination steps. The idea is to get a matrix which have $0$ in the cells above and below <b>pivot</b>. Furthermore, the pivot values can be reduced to $1$. The reduced matrix $R$ is called as the matrix in <b>reduced row-echelon form</b>.

$$\begin{align}
U =
\begin{bmatrix}
    1 & 2 & 2 & 2 \\\\
    0 & 0 & 2 & 4 \\\\
    0 & 0 & 0 & 0
\end{bmatrix} \rightarrow
\begin{bmatrix}
    1 & 2 & 0 & -2 \\\\
    0 & 0 & 1 & 2 \\\\
    0 & 0 & 0 & 0
\end{bmatrix} = R
\end{align}$$

If we further examine the matrix $R$, it can be noticed that an <b>identity matrix</b> $\begin{bmatrix}
    1 & 0  \\\\
    0 & 1 
\end{bmatrix} = I$ sits in the <b>pivot rows and columns</b> ($row_1, row_2, col_1, col_2$). Corresponding entries in the <b>free columns</b> are $\begin{bmatrix}
    2 & -2  \\\\
    0 & 2 
\end{bmatrix} = F$. If we look at $I$ and $F$, they are nothing but the rearranged individual solutions with sign switched for $F$. Let us just rearrnage the reduced row-echelon matrix $R$ as follows, where the last matrix is nothing but matrix represented in the <b>block form</b>. We have to find $x$ such that $Rx=0$.

$$\begin{align}
R = 
\begin{bmatrix}
    1 & 2 & 0 & -2 \\\\
    0 & 0 & 1 & 2 \\\\
    0 & 0 & 0 & 0
\end{bmatrix} \rightarrow
\begin{bmatrix}
    1 & 0 & 2 & -2 \\\\
    0 & 1 & 0 & 2 \\\\
    0 & 0 & 0 & 0
\end{bmatrix} = 
\begin{bmatrix}
    I & F \\\\
    0 & 0 
\end{bmatrix}
\end{align}$$

For the given R in the block form, the solution can be given as:

$$\begin{align}
 \begin{bmatrix}
    I & F \\\\
    0 & 0 
\end{bmatrix}\begin{bmatrix}
    -F \\\\
    I
\end{bmatrix} = 0
\end{align}$$

Hence, the final solution in the rearranged form is:

$$\begin{align}
 \begin{bmatrix}
    -F \\\\
    I
\end{bmatrix} =
\begin{bmatrix}
    -2 & 0 \\\\
    2 & -2 \\\\
    1 & 0 \\\\
    0 & 1
\end{bmatrix}
\end{align}$$

## 5.3 Example

Let us look at one more example. The system of equations which we have to solve is $Bx=0$ where $B=A^T$ for the above matrix $A$.

$$\begin{align}
B = \begin{bmatrix}
    1 & 2 & 3 \\\\
    2 & 4 & 6 \\\\
    2 & 6 & 8 \\\\
    2 & 8 & 10
\end{bmatrix} 
\end{align}$$

The transformation of $B$ into <b>row-reduced echelon form</b> can be given as follows:

$$\begin{align}
B = \begin{bmatrix}
    1 & 2 & 3 \\\\
    2 & 4 & 6 \\\\
    2 & 6 & 8 \\\\
    2 & 8 & 10
\end{bmatrix} \rightarrow
\begin{bmatrix}
    1 & 2 & 3 \\\\
    0 & 0 & 0 \\\\
    0 & 2 & 2 \\\\
    0 & 4 & 4
\end{bmatrix} \rightarrow
\begin{bmatrix}
    1 & 2 & 3 \\\\
    0 & 2 & 2 \\\\
    0 & 0 & 0 \\\\
    0 & 0 & 0 \\\\
\end{bmatrix} \rightarrow
\begin{bmatrix}
    1 & 0 & 1 \\\\
    0 & 2 & 2 \\\\
    0 & 0 & 0 \\\\
    0 & 0 & 0 \\\\
\end{bmatrix} \rightarrow
\begin{bmatrix}
    1 & 0 & 1 \\\\
    0 & 1 & 1 \\\\
    0 & 0 & 0 \\\\
    0 & 0 & 0 \\\\
\end{bmatrix}
\end{align}$$

$$\begin{align}
I = 
\begin{bmatrix}
    1 & 0\\\\
    0 & 1
\end{bmatrix};
F = 
\begin{bmatrix}
    1 \\\\
    1 
\end{bmatrix}
\end{align}$$

Hence the <b>null space</b> can be given as
$$\begin{align}
c\begin{bmatrix}
    -F \\\\
    I
\end{bmatrix} =
c\begin{bmatrix}
    -1 \\\\
    -1 \\\\
    1
\end{bmatrix} 
\end{align}$$
