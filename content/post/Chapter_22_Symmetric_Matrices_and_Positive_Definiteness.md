+++
date = "2022-04-26T14:07:28+01:00"
description = "Linear Algebra (Gilbert Strang): Chapter 22"
draft = false
tags = ["Linear Algebra", "Determinant", "Gilbert Strang", "Eigenvalues", "Eigenvectors", "Eigenvalue Decomposition", "Symmetric Matrices", "Positive Definiteness"]
title = "Symmetric Matrices and Positive Definiteness"
topics = ["Linear Algebra"]

+++

## 22.1 Symmetric Matrices

For a <b>Symmetric Matrix</b> $A$, $A = A^T$. <b>Eigenvalues of real Symmetric Matrices are real and eigenvectors are perpendicular</b>. Any matrix $A$ can be written as $A = S\Lambda S^{-1}$. For a symmetric matrix $A$, this relationship reduces to $A = Q \Lambda Q^{-1}$ as $S$, which is the eigenvector matrix has <b>orthonormal eigenvectors</b>. For an orthonormal matrix, $Q^{-1} = Q^T$ and hence $A = Q\Lambda Q^T$. This is called as the <b>Spectral Theorem</b> in mathematics.

## 22.2 Why eigenvalues of a Symmetric Matrix are Real?

Let's start from the basic equation for the eigenvalue and eigenvector: $Ax = \lambda x$. Taking the conjugate of this equation, we get $\overline{A}\overline{x} = \overline{\lambda}\overline{x}$. As $A$ is a real matrix, $\overline{A} = A$ and hence  $Ax = \lambda x \implies A\overline{x} = \overline{\lambda}\overline{x}$. Taking transpose of the equation, we get $Ax = \lambda x \implies A\overline{x} = \overline{\lambda}\overline{x} \implies \overline{x}^TA^T = \overline{x}^T\overline{\lambda} \implies \overline{x}^TA = \overline{x}^T\overline{\lambda}$, as $A^T = A$ because $A$ is symmetric. Taking the dot product of $\overline{x}^T$ with the original equation, we get $\overline{x}^TAx = \lambda \overline{x}^Tx$. Multiplying the last equation by $x$ we get $\overline{x}^TAx = \overline{\lambda}\overline{x}^Tx$. Comapring these two equations, we have $\overline{x}^TAx = \lambda \overline{x}^Tx = \overline{\lambda}\overline{x}^Tx$, i.e. $ \lambda \overline{x}^Tx = \overline{\lambda}\overline{x}^Tx$. As $\overline{x}^Tx \neq 0$, we get $\overline{\lambda} = \lambda$ and hence eigenvalues are real.

For a <b>complex matrix</b> $A$, we will have <b>real eigenvalues and perpendicular eigenvectors</b> if $\overline{A}^T = A$.

For a symmetric matrix $A$, we have $A = Q\Lambda Q^T$. This further reduces to $A = \lambda_1q_1q_1^T + \lambda_2q_2q_2^T + ...$. The matrix $q_iq_i^T$ is a <b>projection matrix</b> and are multually perpendicular to each other. Hence <b>any symmetric matrix can be decomposed into a combination of perpendicular projection matrices</b>.

## 22.3 Positive Definite Symmetric Matrix and Sign of Eigenvalues

For a symmetric matrix <b>number of positive pivots is same as the number of positive eigenvalues</b>. <b>Positive Definite</b> matrices are symmetric matrices whose all the <b>eigenvalues are positives</b>. This means that all of their <b>pivots are positive</b> as well. Lastly, <b>the determinant of a positive definite matrix is positive</b>, to be more precise <b>all the sub-determinants of a positive definite matrix are positive</b>. This means that for a $n \times n$ positive definite matrix, the determinant of $1 \times 1$ sub-matrix, $2 \times 2$ sub-matrix,... are all positive.
