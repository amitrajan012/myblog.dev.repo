+++
date = "2022-05-15T14:07:28+01:00"
description = "Linear Algebra (Gilbert Strang): Chapter 28"
draft = false
tags = ["Linear Algebra", "Determinant", "Gilbert Strang", "Eigenvalues", "Eigenvectors", "Left Inverse", "Right Inverse", "Pseudo Inverse", "Singular Value Decomposition"]
title = "Left, Right  and Pseudo Inverses"
topics = ["Linear Algebra"]

+++

## 28.1 Left and Right Inverses

For any matrix $A$, there are four fundamental subspaces: <b>Row Space, Null Space, Column Space, Null Space of $A^T$</b>. Row Space and Null Space combined span entire $\mathbb{R}^n$ for a $m \times n$ matrix $A$. Column Space and Null Space of $A^T$ combined span entire $\mathbb{R}^m$ for a $m \times n$ matrix $A$. For the inverse of the matrix to exist, it's rank should be $r=m=n$ whtere the dimension of the matrix is $m \times n$.

For the <b>Left Inverse</b> to exist, the matrix should have a <b>full column rank</b>, i.e. $r=n < m$ and all the columns are independent. For this case, the null space will just have a zero vector in it, i.e. $N(A) = \{0\}$. Lastly, the count of solutions for $Ax=b$ is either $0$ or $1$. For this case, the matrix $A^TA$ will be a $n \times n$ invertible matrix and the <b>left inverse</b> of the matrix $A$ is given as $A_{left}^{-1} = (A^TA)^{-1}A^T$. Then, $A_{left}^{-1}A = (A^TA)^{-1}A^TA = I$, but $AA_{left}^{-1} = A(A^TA)^{-1}A^T = P$, which is a projection on the column space.

Similarly, the <b>Right Inverse</b> to exist, the matrix should have a <b>full row rank</b>, i.e. $r=m < n$ and all the rows are independent. For this case, the null space of $A^T$ will just have a zero vector in it, i.e. $N(A^T) = \{0\}$. Lastly, $Ax=b$ will have infinitely many solutions. For this case, the matrix $AA^T$ will be a $m \times m$ invertible matrix and the <b>right inverse</b> of the matrix $A$ is given as $A_{right}^{-1} = A^T(AA^T)^{-1}$. Then, $AA_{right}^{-1} = AA^T(AA^T)^{-1} = I$, but $A_{right}^{-1}A = A^T(AA^T)^{-1}A = P$, which is a projection on the row space.

## 28.2 PseudoInverse

For a matrix $A$, when any vector $x$ in the row-space is multiplied by $A$, we get $Ax$ which is going to be a vector in the column space as $Ax$ is a combination of columns. This connection between row and column space is <b>one-to-one</b>. To prove this, let us assume that there exist two disticnt vectors $x,y$ in the row-space, i.e. $x \neq y$. Then we have to prove that $Ax \neq Ay$. Let $Ax = Ay$, then $A(x-y) = 0$. But as $x,y$ are in the row-space, $x-y$ will also be in the row-space but from the equation $A(x-y) = 0$, we can infer that $x-y$ is in the null-space of $A$ and hence a contradiction. This means that our assumption $Ax=Ay$ is false and hence $Ax \neq Ay$. This whole thing can be interpreted as: <b>matrix $A$ is a nice one-to-one mapping from row-space to column-space. The reverse mapping from column to row-space is called as pseudo inverse of $A$ and is denoted as $A^{+}$</b>.

To calculate the pseudo inverse of a matrix $A$, the first step is to do SVD on $A$ as $A = U\Sigma V^T$. The matrices $U,V^T$ are orthogonal and invertible. To calculate the pseudo inverse of diagonal matrix $\Sigma$, we have to replace all non-zero diagonal entries with it's inverse, i.e. each $\sigma_i$ in the diagonal will be replaced with $\frac{1}{\sigma_{i}}$ and we will leave the $0$ as it is. Finally $A^{+} =  (U\Sigma V^T)^{+} = V\Sigma^{+}U^T $.
