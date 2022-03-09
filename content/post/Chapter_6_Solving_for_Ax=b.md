+++
date = "2022-01-25T14:07:28+01:00"
description = "Linear Algebra (Gilbert Strang): Chapter 6"
draft = false
tags = ["Linear Algebra", "Matrix Multiplications", "Gilbert Strang",
"Vector Space", "Matrix Factorization", "Subspace", "Column Space", "Null Space", "Reduced Row-Echelon Form"]
title = "Algorithm for solving $Ax=b$"
topics = ["Linear Algebra"]

+++

### 6.1 Algorithm for solving $Ax=b$

The idea behind this exercise is to come up with the solution for $Ax=b$. Let us start by taking a matrix $A$ and vector $b$.

$$\begin{align}
A = \begin{bmatrix}
    1 & 2 & 2 & 2 \\\\
    2 & 4 & 6 & 8 \\\\
    3 & 6 & 8 & 10
\end{bmatrix};
b = \begin{bmatrix}
    b_1 \\\\
    b_2 \\\\
    b_3
\end{bmatrix}
\end{align}$$

The <b>augumented matrix</b> $\begin{bmatrix} A & b \end{bmatrix}$ can be formed and elimination steps can be performed on it as follows.

$$\begin{align}
A = \begin{bmatrix}
    1 & 2 & 2 & 2 & b_1\\\\
    2 & 4 & 6 & 8 & b_2\\\\
    3 & 6 & 8 & 10 & b_3
\end{bmatrix} \rightarrow
\begin{bmatrix}
    1 & 2 & 2 & 2 & b_1\\\\
    0 & 0 & 2 & 4 & b_2-2b_1\\\\
    0 & 0 & 2 & 4 & b_3-3b_1
\end{bmatrix} \rightarrow
\begin{bmatrix}
    1 & 2 & 2 & 2 & b_1\\\\
    0 & 0 & 2 & 4 & b_2-2b_1\\\\
    0 & 0 & 0 & 0 & b_3-b_2-b_1
\end{bmatrix} = U
\end{align}$$

The last row of the augumented matrix after elimination implies that $b_3-b_2-b_1 = 0$ and this is the <b>condition for solvability</b>. The condition of solvability can be listed as:
* $Ax=b$ is solvable when $b \in C(A)$
* If a combination of rows of $A$ gives a zero row, the same combination of entries of $b$ must be $0$. 

One of the valid values of $b$ for which we will have the solution can be $\begin{bmatrix}
    1 \\\\
    5 \\\\
    6
\end{bmatrix}$. Putting this value for $b$, we get the following augumented  row-echelon matrix.

\begin{align}
U = \begin{bmatrix}
    1 & 2 & 2 & 2 & 1\\\\
    0 & 0 & 2 & 4 & 3\\\\
    0 & 0 & 0 & 0 & 0
\end{bmatrix} \rightarrow
\begin{bmatrix}
    1 & 2 & 0 & -2 & -2\\\\
    0 & 0 & 1 & 2 & 3/2\\\\
    0 & 0 & 0 & 0 & 0
\end{bmatrix} = R
\end{align}

To find the complete solution of $Ax=b$, we can start with finding a particular solution (x_p) and then proceed from there. The steps are as follows:
* Set all <b>free varibales</b> in $x$, variables corresponding to <b>free columns</b> to $0$ and solve for the <b>pivot variables</b>. This gives us the set of equations as: $x_1 + 2x_3 =0;2x_3=3;x_2=0;x_4=0$ (as $x_2,x_4$ are free variables). Hence, $x_p = \begin{bmatrix}-2 \\\\
0 \\\\
3/2 \\\\
0
\end{bmatrix}$.
* The <b>complete solution</b> can be given as:
\begin{align}
x = x_p + x_n
\end{align}
where $x_n$ is any vector in the <b>null space</b>. This is valid as the particular solution gives $Ax_p = b$ and any vector in null space gives $Ax_n = 0$. Adding these two, we get $x=x_p+x_n$. For this particular $A$, the null space is $\begin{bmatrix}
    -2 & 0 \\\\
    2 & -2 \\\\
    1 & 0 \\\\
    0 & 1
\end{bmatrix}$. Hence the complete solution is:
\begin{align}
x = \begin{bmatrix}
    -2 \\\\
    0 \\\\
    3/2 \\\\
    0
\end{bmatrix}+
c_1\begin{bmatrix}
    -2 \\\\
    2  \\\\
    1  \\\\
    0 
\end{bmatrix} +
c_2\begin{bmatrix}
    0 \\\\
    -2  \\\\
    0 \\\\
    1 
\end{bmatrix}
\end{align}

The above solution forms a two-dimensional plain in $R^4$ which goes through $x_p$. It should be noted that this is not a subsapce as it does not goes through origin.

Let us further explore about the nature of the solutions of $Ax=b$. For a $m \times n$ matrix $A$, the <b>rank $r$</b> satisfies: $r \leq m$ and $r \leq n$, as there can't be more pivots than number of columns or number of rows in the matrix. 

* A <b>full column rank</b> means $r=n < m$. A full columnn rank means we have $n$ pivots, no <b>free variables</b> and hence the nullspace will only have the zero vector. It's reduced row-echelon matrix will look like $\begin{bmatrix}
    I \\\\
    0 
\end{bmatrix}$. This means that the solution $x$ for $Ax=b$ will be just $x_p$ i.e. <b>a unique solution if it exists</b> i.e. $0$ or $1$ solution.

* A <b>full row rank</b> means $r=m < n$. This means we will have $m$ pivots, i.e. every row will have pivot and there will be $n-r=n-m$ <b>free columns/variables</b>. It's typical reduced row-echelon matrix will look like $\begin{bmatrix}
    I & F
\end{bmatrix}$In this case, there isn't any restriction on $b$ as well as the equation will be solvable for each $b$ and will have <b>infinitely many solutions</b>.

* For a case, when $r=m=n$, the matrix is a square matrix and we call the matrix a <b>full rank</b> matrix. A full rank rank matrix is <b>always invertible</b>. It should also be noted that it's reduced row-echelon form is $I$. It's nullspace is a zero vector and for the equation $Ax=b$ to be solvable, there is't any restriction on $b$.

* For a case when, $r < m;r < n$, the typical reduced row-echelon matrix will look like $\begin{bmatrix}
    I & F \\\\
    0 & F
\end{bmatrix}$ and $Ax=b$ can have $0$ or infinitely many solutions.

One more thing to consider is the fact that for a $m \times n$ matrix $A$, if $m < n$, then we will have at least one non-zero solution to $Ax=0$.
