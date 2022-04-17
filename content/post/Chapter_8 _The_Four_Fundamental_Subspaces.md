+++
date = "2022-03-15T14:07:28+01:00"
description = "Linear Algebra (Gilbert Strang): Chapter 8"
draft = false
tags = ["Linear Algebra", "Vector Space", "Gilbert Strang",
"Vector Space", "Subspace", "Column Space", "Null Space", "Row Space", "Left Null Space", "Basis", "Dimension"]
title = "Four Fundamental Subspaces"
topics = ["Linear Algebra"]

+++


## 8.1 Four Fundamental Subspaces

The four fundamental subspaces for a $m \times n$ matrix $A$ are as follows:

* <b>Column Space $C(A)$ in $\mathbb{R}^m$</b>
* <b>Null Space $N(A)$ in $\mathbb{R}^n$</b>: Solution to $Ax=0$
* <b>Row Space $C(A^T)$ in $\mathbb{R}^n$</b>: All combinations of the rows of $A$ or we can say that <b>all combinations of the columns of $A^T$</b>
* <b>Left Null Space of $A^T$ $N(A^T)$ in $\mathbb{R}^m$</b>: Solution to $A^Ty=0$ and is also called as <b>Left Null Spcae of $A$</b>

The pictorial representation of these spaces with their dimension and basis is as follows:

{{% fluid_img "/img/Linear_Algebra/Four_Subspaces.png" %}}

Let's take an example matrix $A$ to understand these four subspaces in a better way. 

$$\begin{align}
A = \begin{bmatrix}
    1 & 2 & 3 & 1 \\\\
    1 & 1 & 2 & 1 \\\\
    1 & 2 & 3 & 1
\end{bmatrix} 
\xrightarrow{\text{S1}}
\begin{bmatrix}
    1 & 2 & 3 & 1 \\\\
    0 & 1 & 1 & 0 \\\\
    0 & 0 & 0 & 0
\end{bmatrix}
\xrightarrow{\text{S2}}
\begin{bmatrix}
    1 & 0 & 1 & 1 \\\\
    0 & 1 & 1 & 0 \\\\
    0 & 0 & 0 & 0
\end{bmatrix} = R
\end{align}$$

where, the performed <b>row operations</b> on the matrix $A$ are: $S1: row_2 = -1(row_2 - row_1);row_3 = row_3 - row_1$ and $S2:row_1 = (row_1 - 2 \times row_2)$. The resultant matrix $R$ can be represented as:

$$\begin{align}
R = \begin{bmatrix}
    I & F \\\\
    0 & 0 
\end{bmatrix};
I = \begin{bmatrix}
    1 & 0 \\\\
    0 & 1 
\end{bmatrix}
F = \begin{bmatrix}
    1 & 1 \\\\
    1 & 0 
\end{bmatrix}
\end{align}$$

### 8.1.1 Column Space & Null Space

<b>Column Space</b> $C(A)$ is formed by the pivot columns of the reduced-row echelon matrix $R$. It's dimension is $r$ where $r$ is the <b>rank</b> of the matrix or number of pivot columns in it. One important thing to note is: <b>Row operations DO NOT preserve the Column Space.</b> (as the column space of $A$ and $R$ are different). i.e. $C(A) \neq C(R)$.

<b>Null Space</b> is the solution of the equation $Ax=0$. As discussed in the previous post, the <b>dimension of null space</b> is the number of free variables which is $n-r$.

### 8.1.2 Row Space

The matrix $R$ is obtained by <b>row operations</b> on $A$. <b>Row operations preserve the Row Space</b>. The <b>basis for the row space</b> of matrix $A$ is the <b>first $r$ rows of $R$</b>. The logic of first $r$ columns is not true for $A$ as row exchange can be one of the steps in the row operations. The reason for row operation preserving the row space is the fact that it only consists of linear operations on each of the row. Every row operation transforms the row linearlly by taking linear combinations of different rows in the matrix. As all linear combinations of the vectors in a subspace lie in the subspace itself, row operations do not preserve the row space. The <b>dimension of row space</b> is $r$ as there will be $r$ pivots and hence $r$ non-zero rows in $R$. 

### 8.1.3 Left Null Space

<b>Left Null Space</b>, also called as null-space of $A^T N(A^T)$ is the vector spcae spanned by the solution of the equation $A^Ty=0$. The <b>row reduction steps</b> which transform $A$ to $R$ can be encoded in a matrix $E$ such that $EA=R$. $E$ can be obtained by performing the same steps on an <b>Identity Matrix</b>. 

$$\begin{align}
I = \begin{bmatrix}
    1 & 0 & 0 \\\\
    0 & 1 & 0 \\\\
    0 & 0 & 1
\end{bmatrix} 
\xrightarrow{\text{S1}}
\begin{bmatrix}
    1 & 0 & 0 \\\\
    1 & -1 & 0 \\\\
    -1 & 0 & 1
\end{bmatrix}
\xrightarrow{\text{S2}}
\begin{bmatrix}
    -1 & 2 & 0 \\\\
    1 & -1 & 0 \\\\
    -1 & 0 & 1
\end{bmatrix} = E
\end{align}$$

Hence, the above row reduction procedure can be written as:

$$\begin{align}
\begin{bmatrix}
    -1 & 2 & 0 \\\\
    1 & -1 & 0 \\\\
    -1 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
    1 & 2 & 3 & 1 \\\\
    1 & 1 & 2 & 1 \\\\
    1 & 2 & 3 & 1
\end{bmatrix}
= \begin{bmatrix}
    1 & 0 & 1 & 1 \\\\
    0 & 1 & 1 & 0 \\\\
    0 & 0 & 0 & 0
\end{bmatrix}
\end{align}$$

Now, finding <b>Left Null Space</b> means finding the solution of $A^Ty=0$. This means, we have to find the combinations of rows of $A$ (or columns of $A^T$) which gives us the $0$ vector. If we look at the above equation, the last row of $R$ is a zero vector which we get by multiplying the matrix $A$ by $\begin{bmatrix}-1 & 0 & 1\end{bmatrix}$.

$$\begin{align}
\begin{bmatrix}
    -1 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
    1 & 2 & 3 & 1 \\\\
    1 & 1 & 2 & 1 \\\\
    1 & 2 & 3 & 1
\end{bmatrix}
= \begin{bmatrix}
    0 & 0 & 0 & 0
\end{bmatrix}
\end{align}$$

Hence, we can say that the <b>basis of Left Null Space</b> is formed by the rows in E corresponding to the zero vector in $R$. It's dimension will be $m-r$ as there will be $m-r$ (number of rows - pivot rows) zero vector rows in $R$.

