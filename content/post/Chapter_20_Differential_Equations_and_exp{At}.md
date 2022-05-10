+++
date = "2022-04-20T14:07:28+01:00"
description = "Linear Algebra (Gilbert Strang): Chapter 20"
draft = false
tags = ["Linear Algebra", "Determinant", "Gilbert Strang", "Eigenvalues", "Eigenvectors", "Eigenvalue Decomposition", "Differential Equations", "Matrix Exponentials"]
title = "Differential Equations and Matrix Exponentials"
topics = ["Linear Algebra"]

+++


## 20.1 Differential Equations $\frac{du}{dt} = Au$

<b>Example:</b> Let the system of differential equation to be solved is: $\frac{du_1}{dt} = -u_1 + 2u_2; \frac{du_2}{dt} = u_1 - 2u_2$ with initial condition of $u(0) = \begin{bmatrix}
1 \\\\
0
\end{bmatrix}$. The matrix $A$ representing the coefficients of the equation is $A = \begin{bmatrix}
-1 & 2 \\\\
1 & -2
\end{bmatrix}$. The eigenvalues of the matrix $A$ satisfies the equation $\lambda_1 + \lambda_2 = -3; \lambda_1 \times \lambda_2 = 0$, i.e. $\lambda_1 = 0, \lambda_2 = -3$ with the eigenvectors $x_1 = \begin{bmatrix}
2 \\\\
1
\end{bmatrix}; x_2 = \begin{bmatrix}
1 \\\\
-1
\end{bmatrix}$. The general solution of the set of differential equation is given as: $u(t) = c_1e^{\lambda_1t}x_1 + c_2e^{\lambda_2t}x_2$. Individual pure solutions can be checked by plugging in $e^{\lambda_1t}x_1$ and $e^{\lambda_2t}x_2$ to the equation $\frac{du}{dt} = Au$ and verifying the outcome.

To compute the value of constants, we can plug in the values of $\lambda_1,\lambda_2,x_1,x_2$ in the equation. The updated equation is $u(t) = c_1 \begin{bmatrix}
2 \\\\
1
\end{bmatrix}+ c_2e^{-3t}\begin{bmatrix}
1 \\\\
-1
\end{bmatrix}$. Using $u(0) = \begin{bmatrix}
1 \\\\
0
\end{bmatrix}$, we get $\begin{bmatrix}
1 \\\\
0
\end{bmatrix} = c_1 \begin{bmatrix}
2 \\\\
1
\end{bmatrix}+ c_2\begin{bmatrix}
1 \\\\
-1
\end{bmatrix}$, which gives $c_1=c_2=\frac{1}{3}$. Hence the final solution is $u(t) = \frac{1}{3} \begin{bmatrix}
2 \\\\
1
\end{bmatrix}+ \frac{1}{3}e^{-3t}\begin{bmatrix}
1 \\\\
-1
\end{bmatrix}$.

The <b>condition of stability</b> means at some point in time $u(t) \rightarrow 0$. This means that $e^{\lambda t} \rightarrow 0$, and hence $\lambda < 0$ or to be more precise $Re(\lambda) < 0$ as the absolute value of the imaginary part is $1$. 

The <b>condition of steady state</b> is insured when one of the eigenvalues is $0$ and for other eigenvalues $Re(\lambda) < 0$. 

The <b>value diverges</b> if for any of the eigenvalues $Re(\lambda) > 0$. One important thing to note is: <b>reversing the sign of a matrix reverses the sign of eigenvalues</b>. 

The stability condition for a $2 \times 2$ matrix $A = \begin{bmatrix}
a & b \\\\
c & d
\end{bmatrix}$ is $Re(\lambda_1) < 0$ and $Re(\lambda_2) < 0$. This means that <b>trace</b>, $\lambda_1 + \lambda_2 < 0$ and the <b>determinant</b> $|A| = ad - bc > 0$. It should be noted that a negative trace isn't enough to make the matrix stable.

Another way to look at the equation $\frac{du}{dt} = Au$ is to set $u = Sv$, where $S$ is the <b>eigenvector matrix</b>. Differentiating this equation w.r.t. $t$ gives $\frac{du}{dt} = S\frac{dv}{dt}= ASv$. This implies $\frac{dv}{dt} = S^{-1}ASv = \Lambda v$. $\Lambda$ is a diagonal matrix of eigenvalues. This means that we don't have any coupled terms in the equation and we can separate them as: $\frac{dv_1}{dt} = \lambda_1v_1; \frac{dv_2}{dt} = \lambda_2v_2..., \frac{dv_n}{dt} = \lambda_nv_n$. The general solution for this equation is: $v(t) = e^{\Lambda t}v(0)$. This gives $u(t) = Sv(t) = Se^{\Lambda t} v(0) = Se^{\Lambda t} S^{-1} u(0) = e^{At} u(0)$. <b>The relation</b> $e^{At} = Se^{\Lambda t} S^{-1}$ <b>is proved in the next section</b>.

## 20.2 Matrix Exponentials

Exponential expressions can be expanded using <b>power series expansion</b>. For the expression $e^x$, the power series expansion is given as:

$$\begin{align}
e^x = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + ... + \frac{x^n}{n!} + ...
\end{align}$$ 

For the expression involving matrix, the expansion of $e^{At}$ can be given as:

$$\begin{align}
e^{At} = I + At + \frac{(At)^2}{2!} + \frac{(At)^3}{3!} + ... + \frac{(At)^n}{n!} + ...
\end{align}$$ 

The expression $(I-At)^{-1}$ can be expanded using expansion of geometric series as follows:

$$\begin{align}
(I-At)^{-1} = I + At + (At)^2 + (At)^3 + ... + (At)^n + ...
\end{align}$$ 

$e^{At}$ will always converge but for $(I-At)^{-1}$ to converge, the eigenvalues should be negative. The expression $e^{At}$ can be further transformed as:

$$\begin{align}
e^{At} = I + At + \frac{(At)^2}{2!} + \frac{(At)^3}{3!} + ... + \frac{(At)^n}{n!} + ...
\end{align}$$ 

$$\begin{align}
=I + S\Lambda S^{-1}t + \frac{(S\Lambda S^{-1}t)^2}{2!} + \frac{(S\Lambda S^{-1}t)^3}{3!} + ... + \frac{(S\Lambda S^{-1}t)^n}{n!} + ...
\end{align}$$ 

$$\begin{align}
=SS^{-1} + S\Lambda S^{-1}t + \frac{(S\Lambda^2 S^{-1}t^2)}{2!} + \frac{(S\Lambda^3 S^{-1}t^3)}{3!} + ... + \frac{(S\Lambda^n S^{-1}t^n)}{n!} + ...
\end{align}$$ 

$$\begin{align}
=S\bigg[I + \Lambda t + \frac{(\Lambda t)^2}{2!} + \frac{(\Lambda t)^3}{3!} + ... + \frac{(\Lambda t)^n}{n!} + ...\bigg]S^{-1} = Se^{\Lambda t}S^{-1}
\end{align}$$ 

Another thing to note is the fact that a <b>diagonal matrix</b> is always decoupled. i.e.

$$\begin{align}
e^{\Lambda t} = \begin{bmatrix}
e^{\lambda_1t} & ... & ... \\\\
... & ... & ... \\\\
... & ... & e^{\lambda_nt}
\end{bmatrix}
\end{align}$$

## 20.3 Second Order Differential Equation

Let $y^{''} + by^{'} + k = 0$ be a second order differential equation. If we take $u = \begin{bmatrix}
y^{'} \\\\
y
\end{bmatrix}$, then $u^{'} = \begin{bmatrix}
y^{''} \\\\
y^{'}
\end{bmatrix}$. Apart from this, we can add one more trivial equation $y^{'} = y^{'}$. Combining these things together, we get

$$\begin{align}
u^{'} = \begin{bmatrix}
y^{''} \\\\
y^{'}
\end{bmatrix} = \begin{bmatrix}
-b & -k \\\\
1 & 0
\end{bmatrix}\begin{bmatrix}
y^{'} \\\\
y
\end{bmatrix}
\end{align}$$

which is a first order differential equation.
