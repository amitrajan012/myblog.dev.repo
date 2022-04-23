+++
date = "2022-03-23T14:07:28+01:00"
description = "Linear Algebra (Gilbert Strang): Chapter 11"
draft = false
tags = ["Linear Algebra", "Matrix Space", "Gilbert Strang",
"Vector Space", "Orthogonal Vector", "Orthogonal Subspace"]
title = "Orthogonal Vectors and Orthogonal Subspaces"
topics = ["Linear Algebra"]

+++

## 11.1 Orthogonal Vectors

Orthogonal means <b>perpendicular</b>. For two vectors $x,y$, they are orthogonal if and only if $x^Ty=0$. As per <b>Pythagoras Theorem</b>, the test for orthogonality is: $\lVert x \rVert^2 + \lVert y \rVert^2 = \lVert x+y \rVert^2$.

We can conclude the dot product condition from Pythoagoras Theorem as follows:

$$\begin{align}
\lVert x \rVert^2 + \lVert y \rVert^2 = \lVert x+y \rVert^2 \\\\
x^Tx + y^Ty = (x+y)^T(x+y) \\\\
x^Tx + y^Ty = x^Tx + y^Ty + x^Ty + y^Tx \\\\
x^Ty + y^Tx = 0 \\\\
2x^Ty = 0 \\\\
x^Ty = 0
\end{align}$$

One importnt thing to note is: <b>zero vector is orthogonal to all the other vectors</b>.

## 11.2 Orthogonal Subspaces

Subspace $S$ is orthogonal to Subspace $T$ if and only if: <b>Every vector in subspace $S$ is orthogonal to every vector in $T$</b>. Some of the examples of orthogonal subspaces are: <b>Row Space</b> $\perp$ <b>Null Space</b>, <b>Column Space</b> $\perp$ <b>Null Space($A^T$)</b>.

#### 11.1.1 <b>Row Space</b> $\perp$ <b>Null Space</b>:

Null Space consists of the solution vectors $x$ of the equation $Ax=0$. If we expand this equation, we get the following:

$$\begin{align}
\begin{bmatrix}
    row1 \\\\
    row2 \\\\
    ... \\\\
    rowN
\end{bmatrix} 
\begin{bmatrix}
    x
\end{bmatrix} =
\begin{bmatrix}
    0 \\\\
    0 \\\\
    ... \\\\
    0
\end{bmatrix} 
\end{align}$$

Expanding this equation using <b>row times column multiplication method</b>, we get: $[row1]^T[x]=0; [row2]^T[x]=0; ... ;[rowN]^T[x]=0$. <b>Row Space</b> of the matrix will have these row vectors or linear combination of these row vectors ($c_1[row1] + c_2[row2] + ... + c_N[rowN]$) in it. Combining these equations together we get: $(c_1[row1] + c_2[row2] + ... + c_N[rowN])^T[x] = 0$. This means any vector in row space is perpendicular to $x$ as it's dot product with $x$ is $0$.

Using similar arguments, we can prove that <b>Column Space</b> $\perp$ <b>Null Space($A^T$)</b>. It should be noted that Row Space with Null Space carves the entire $n-dimensional$ space. Similarly, Column Space with Null Space of $A^T$ carves the entire $m-dimensional$ space.

Hence, to conclude:
* <b>Null Space and Row Space are Orthogonal Complements in $\mathbb{R}^n$
* <b>Null Space of $A^T$ and Column Space are Orthogonal Complements in $\mathbb{R}^m$

## 11.3 Solve $Ax=b$ when we have No Solution

Let $A$ be a $m \times n$ matrix such that $m > n$. Our goal is to solve $Ax=b$ when it has no solution. By solving this equation, we need to find the best solution in the given constraint. 

Consider the matrix $A^TA$. One of the important property of $A^TA$ is that this matrix is a $n \times n$ <b>symmetric matrix</b>. If we nultiply the above equation by $A^T$, we get $A^TA \hat{x} = A^Tb$. 

One of the examples of unsolvalbe $Ax=b$ is as follows:

$$\begin{align}
\begin{bmatrix}
    1 & 1 \\\\
    1 & 2 \\\\
    1 & 5
\end{bmatrix} 
\begin{bmatrix}
    x_1 \\\\
    x_2
\end{bmatrix}=
\begin{bmatrix}
    b_1 \\\\
    b_2 \\\\
    b_3
\end{bmatrix}
\end{align}$$

Transforming this equation into the form $A^TA\hat{x} = A^Tb$, we get

$$\begin{align}
\begin{bmatrix}
    3 & 8 \\\\
    8 & 30
\end{bmatrix} 
\begin{bmatrix}
    \hat{x_1} \\\\
    \hat{x_2}
\end{bmatrix}=
\begin{bmatrix}
    1 & 1 & 1 \\\\
    1 & 2 & 5
\end{bmatrix}
\begin{bmatrix}
    b_1 \\\\
    b_2 \\\\
    b_3
\end{bmatrix}
\end{align}$$

If $A^TA$ is <b>invertible</b>, we can get the solution for $A^TA\hat{x} = A^Tb$. Two important properties of $A^TA$ are as follows:

* $N(A^TA) = N(A)$
* $Rank(A^TA) = Rank(A)$
* Using these properties, we can conclude that <b>$A^TA$ is invertible if and only if $N(A)$ only has zero vectors in it</b>, i.e. $A$ has independent columns.
