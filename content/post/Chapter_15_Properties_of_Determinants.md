+++
date = "2022-04-05T14:07:28+01:00"
description = "Linear Algebra (Gilbert Strang): Chapter 15"
draft = false
tags = ["Linear Algebra", "Determinant", "Gilbert Strang"]
title = "Determinant"
topics = ["Linear Algebra"]

+++


## 15.1 Determinant

Every <b>square matrix</b> has a number assosiated with it can we call this number as <b>determinant</b>, often denoted as $det(A) = |A|$ for matrix $A$.

Prperties of determinant is as follows:

1. $|I|=1$


2. <b>Row exchange</b> reverses the sign of the determinant:

<b>Permutation Matrices</b> are derived by row exchange of <b>Identity Matrix</b>. Hence, $|P|= \pm1$.


3. (a)  For any square matrix $A$, $\begin{vmatrix}
ta & tb \\\\
c & d
\end{vmatrix}=t\begin{vmatrix}
a & b \\\\
c & d
\end{vmatrix}$


3. (b)  For any square matrix $A$, $|A|$ behaves like a linear function of a row if all the other rows are keep fixed. $\begin{vmatrix}
a+a^{'} & b+b^{'} \\\\
c & d
\end{vmatrix}=\begin{vmatrix}
a & b \\\\
c & d
\end{vmatrix}+\begin{vmatrix}
a^{'} & b^{'} \\\\
c & d
\end{vmatrix}$


4. <b>If two rows of a square matrix $A$ are equal, $|A| = 0$</b>: This can be proved using <b>property 2</b>. Exchanging the rows changes the sign of the determinant, but for a matrix $A$ which has two equal rows, the matrix obtained by exchanging these equal rows is same as $A$, i.e. $|A| = -|A| \implies |A|=0$.


5. <b>Subtracting a multiple of one row from another,  doesn't change the determinant</b>: 

$$\begin{align}
|A| = \begin{vmatrix}
    a & b \\\\
    c-la & d-lb
\end{vmatrix}=
\begin{vmatrix}
    a & b \\\\
    c & d
\end{vmatrix}+\begin{vmatrix}
    a & b \\\\
    -la & -lb
\end{vmatrix} [\text{From 3(b)}]
\end{align}$$

$$\begin{align}
=\begin{vmatrix}
    a & b \\\\
    c & d
\end{vmatrix}+(-l)\begin{vmatrix}
    a & b \\\\
    a & b
\end{vmatrix} [\text{From 3(a)}]=\begin{vmatrix}
    a & b \\\\
    c & d
\end{vmatrix}+(-l)0 [\text{From 4}] = |A|
\end{align}$$


6. <b>For a square matrix $A$, row of zeroes lead to $|A|=0$</b>: Let $t$ be a multiple other than one for one of the rows of $A$, then

$$\begin{align}
t|A| = t\begin{vmatrix}
    0 & 0 \\\\
    c & d
\end{vmatrix}=
\begin{vmatrix}
    0 & 0 \\\\
    c & d
\end{vmatrix}=|A| [\text{From 3(b)}] 
\implies t|A| = |A| \implies |A| = 0
\end{align}$$


7. <b>For an upper traingular matrix $U$ with $d_is$ are the diagonal elements, $|U| = \prod_{i=1}^{n}d_i$</b>: Any upper triangular matrix $U$ can be converted to a diagonal matrix $D$ just by row elimination steps. Hence, $|U| = |D|$. For a $3 \times 3$ matrix, we can say that

$$\begin{align}
|U|= |D| = \begin{vmatrix}
    d_1 & 0 & 0 \\\\
    0 & d_2 & 0 \\\\
    0 & 0 & d_3
\end{vmatrix}=d_1d_2d_3\begin{vmatrix}
    1 & 0 & 0 \\\\
    0 & 1 & 0 \\\\
    0 & 0 & 1
\end{vmatrix}\text{[From 3(a)]}=d_1d_2d_3|I|=d_1d_2d_3
\end{align}$$

8. <b>$|A|=0$ exactly when $A$ is singular and $|A| \neq 0$ when $A$ is invertible</b>: If the matrix is singular, by elimination steps we can get a row of all zeroes and hence $0$ determinant.


9. <b>$|AB| = |A||B|$</b>: Using this property, $A^{-1}A = I \implies |A^{-1}A| = |I| \implies |A^{-1}||A| = |I| \implies |A^{-1}| = \frac{1}{|A|}$. Similarly, $|A^2| = |A|^2$ and $|2A| = 2^n|A|$ (if multiplying a matrix by 2 means doubling all the entries of $A$).


10. <b>$|A^T| = |A|$</b>: This means that all the properties about the rows hold good for columns as well. <b>Proof:</b> Any matrix $A$ can be factored as $LU$. Using this,

$$\begin{align}
|A^T| = |A| \implies |U^TL^T| = |LU|
\end{align}$$

Now, $L$ is a matrix where all the diagonal elements are $1$ with the elements in upper half $0$, i.e. $|L| = 1$. $L^T$ is an upper triangular matrix where all the diagonal elements are $1$ and hence $|L^T| = 1$. Similarly, $|U^T| = |U|$. Hence the equality holds.
