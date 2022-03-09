+++
date = "2022-01-17T14:07:28+01:00"
description = "Linear Algebra (Gilbert Strang): Chapter 4"
draft = false
tags = ["Linear Algebra", "Matrix Multiplications", "Gilbert Strang",
"Vector Space", "Matrix Factorization", "Subspace", "Column Space", "Null Space"]
title = "Vector Space and Subspace"
topics = ["Linear Algebra"]

+++

## 4.1 Vector Space and Subspace

<b>Vector Space</b> is a set of vectors whose linear combinations belong to the same set. This means that whenever we pick $2$ vectors from a vector space and find thier linear combination, the resultant vector will be in the vector space.
A vector space in two and three dimensions is $R^2$ and $R^3$.

We can make some common observations about vector spaces. Let $u,v$ be two vectors, then their linear combination can be described as $w=au+bv$ where $a,b$ are scalars. If $a=b=0$, $w=au+bv=0$. This means that <b>for any vector space (which contains vectors $u,v$), the zero vector will always be in it</b>. This observation gives us a powerful tool to deduce whether a set is a vector space or not. If zero vector is not in the set, it's not a vector space.

A vector space that is contained within another vector space is called a <b>subspace</b> of that space. A space defined in $R^2$ contianing just zero vector as $S=\bigg\\{\begin{bmatrix}
    0 \\\\
    0 
\end{bmatrix}\bigg\\}$ is a <b>subspace</b> as any linear combination of zero vector is still a zero vector. Another example of subspace in $R^2$ is <b>a line passing through origin</b> as any multiple of the vector will also be on the line and hence fulfilling the criteria of a vector space. Any plane passing through origin is a subspace in $R^3$ as linear combination of the vectors in the plane will still reside in the plane.

## 4.2 Union and Intersection of Vector Space

Let $S_1,S_2$ be two vector spaces, their union $S_1 \cup S_2$ is not a subsapce. For example, let $S_1$ and $S_2$ be two lines, the sum of two non-zero vectors in $S_1$ and $S_2$ respectively will not be on any of the lines and hence the union is not a subspace.

The <b>intersection of the subspaces denoted as $S_1 \cap S_2$ is a subspace</b>. Let there are two vectors $u,v$ such that both $u,v \in S_1 \cap S_2$. This means that $u,v \in S_1$ and $u,v \in S_2$. For $S_1 \cap S_2$ to be a <b>subsapce</b>, $au + bv \in S_1 \cap S_2$ where $a$ and $b$ are scalars. <b>As $u,v \in S_1$ and $S_1$ is a subspace, it's linear combination $au + bv \in S_1$</b>. Similarly, $au + bv \in S_2$ and hence $au + bv \in S_1 \cap S_2$.

## 4.3 Column Space

Given a matrix $A$, it's columns and all their linear combinations form a vector space called as the <b>column space</b> $C(A)$ of $A$. For example, for the matrix $A = \begin{bmatrix}
    1, 3 \\\\
    2, 3 \\\\
    4, 1 
\end{bmatrix}$ it's column space $C(A)$ is a plane passing through $\begin{bmatrix}
    0 \\\\
    0 \\\\
    0
\end{bmatrix}$ (origin), $\begin{bmatrix}
    1 \\\\
    2 \\\\
    4
\end{bmatrix}$, $\begin{bmatrix}
    3 \\\\
    3 \\\\
    1
\end{bmatrix}$ in $R^3$. 

Let us look at another example. Following matrix $A$ is in $R^4$. The <b>column space $C(A)$</b> of $A$ is formed by <b>all linear combinations of columns of $A$</b>. Let there be a system of linear equations $Ax = b$ where $x,b$ are column vectors. <b>Does this system of linear equation will have a solution $x$ for all values of $b$?</b>

$$\begin{align}
A = \begin{bmatrix}
    1 & 1 & 2 \\\\
    2 & 1 & 3 \\\\
    3 & 1 & 4 \\\\
    4 & 1 & 5
\end{bmatrix},
x = \begin{bmatrix}
    x_1 \\\\
    x_2 \\\\
    x_3
\end{bmatrix},
b = \begin{bmatrix}
    b_1 \\\\
    b_2 \\\\
    b_3 \\\\
    b_4
 \end{bmatrix}
\end{align}$$

The system of linear equation $Ax = b$ can be further reduced to:

$$\begin{align}
Ax =b \implies\begin{bmatrix}
    1 & 1 & 2 \\\\
    2 & 1 & 3 \\\\
    3 & 1 & 4 \\\\
    4 & 1 & 5
\end{bmatrix} \begin{bmatrix}
    x_1 \\\\
    x_2 \\\\
    x_3
\end{bmatrix} = b
 \implies x_1\begin{bmatrix}
    1 \\\\
    2 \\\\
    3 \\\\
    4 
\end{bmatrix} +
x_2\begin{bmatrix}
    1 \\\\
    1 \\\\
    1 \\\\
    1 
\end{bmatrix} +
x_3\begin{bmatrix}
    2 \\\\
    3 \\\\
    4 \\\\
    5 
\end{bmatrix} = b
\end{align}$$

This means that for $Ax=b$ to have a solution, $b$ should be the linear combination of the columns of $A$. Hence, <b>the system of linear equation $Ax=b$ will have a solution only if $b \in C(A)$</b>, where $C(A)$ is the column space of $A$.

If we further look at the columns of $A$, $col_3$ is a linear combination of $col_1$ and $col_2$ as $col_3 = col_1 + col_2$. This means that $col_3$ does not contribnute or add any dimension while forming the column space of $A$. $C(A)$ can be defined just using $col_1$ and $col_2$. Hence <b>column space</b> $C(A)$ of $A$ is a two dimensional plane in $R^4$.

## 4.4 Null Space

<b>Null space of marix $A$ contains all solutions of $Ax=0$</b>, i.e null space is the solution of following equation for above $A$. Here the null space will be in $R^3$ as $x$ is a three dimensional vector.

$$\begin{align}
Ax=0 \implies \begin{bmatrix}
    1 & 1 & 2 \\\\
    2 & 1 & 3 \\\\
    3 & 1 & 4 \\\\
    4 & 1 & 5
\end{bmatrix} \begin{bmatrix}
    x_1 \\\\
    x_2 \\\\
    x_3
\end{bmatrix} = 
 \begin{bmatrix}
    0 \\\\
    0 \\\\
    0 \\\\
    0 
\end{bmatrix}
\end{align}$$

Some the vectors which will be in the null space of $A$ are: $\begin{bmatrix}
    0 \\\\
    0 \\\\
    0
\end{bmatrix},
\begin{bmatrix}
    1 \\\\
    1 \\\\
    -1
\end{bmatrix},
\begin{bmatrix}
    2 \\\\
    2 \\\\
    -2
\end{bmatrix},...$ In general, all vectors of the form $\begin{bmatrix}
    c \\\\
    c \\\\
    -c
\end{bmatrix}$ forms the <b>null space of $A$</b>.

For these set of vectors to be a subspace, for two solutions $u,v$ of $Ax=0$, their linear combination $au+bv$ should also be a solution of $Ax=0$. As $u, v$ are solutions of $Ax=0$, $Au=0;Av=0$. This means:

$$\begin{align}
aAu=0;bAv=0 \implies Aau=0; Abv=0 \implies Aau+Abv=0 \implies A(au+bv)=0 
\end{align}$$

Hence, $au+bv$ is also a solution of $Ax=0$ which makes nullspace a vector space. One of the major intution about null space is the fact that <b>for any matrix $A$, null space will contains non-zero vectors only if all of it's columns are not lineraly independent</b>, i.e. at least one of the columns of $A$ should be the linear combination of all the other columns.

Another important question which can be asked that whether for any non-zero $b$, whether the solutions of the equation $Ax=b$ forms a subspace. <b>No, they don't</b>. As for non-zero $b$, the zero vector is not a solution and hence the set of solutions don't form a subspace as every subspace should contain the zero vector.
