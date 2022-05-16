+++
date = "2022-05-12T14:07:28+01:00"
description = "Linear Algebra (Gilbert Strang): Chapter 27"
draft = false
tags = ["Linear Algebra", "Determinant", "Gilbert Strang", "Eigenvalues", "Eigenvectors", "Linear Transformations", "Change of Basis", "Image Compression", "Fourier Basis", "Wavelet Basis"]
title = "Linear Transformations, Change of Basis and Image Compression"
topics = ["Linear Algebra"]

+++

## 27.1 Linear Transformations

When we are dealing with coordinates, every linear transformation leads us to a matrix. For any vector $v$ and $w$, linear transformation $T$ follows following properties:

* $T(v+w) = T(v) + T(w)$
* $T(cv) = cT(v)$
* $T(0) = T(0)$, derived from tha above two properties

Below are some examples and non-examples of linear transformation:

<b>Example 1: Projection</b> In a two-dimensional space $\mathbb{R}^2$, a projection of a vector $v$ on a line is linear transformation and can be denoted as $T:\mathbb{R}^2 \to \mathbb{R}^2$.

<b>Example 2:</b> Shifting a vector by $v_0$ is <b>not a linear transform</b>, as $T(v) = v+v_0; T(cv) = cv + v_0 \neq cT(v)$.

<b>Example 3: Rotation by $45^{\circ}$</b> or any degree is a linear transormation. 

<b>Example 4:</b> For any matrix $A$ and vector $v$, $T(v) = Av$ is a linear transformation.

The goal of linear algebra in linear transformation is to find the matrix behind it.

## 27.2 Matrix behind Linear Transformation

Let a linear transformation $T$ is $T:\mathbb{R}^3 \to \mathbb{R}^2$. In matrix notation, it can be represented as $T(v) = Av$ where $A$ is a $2 \times 3$ matrix. One of the fundamental question is: <b>how much information do we need to know such that $\forall v$, we can derive $T(v)$</b>. Let us say that we know the linear transformation for vector $v_1$ and $v_2$ are $T(v_1)$ and $T(v_2)$. From this information, we can derive the linear transformation for all the linear combinations of $v_1,v_2$. Hence, <b>if the basis vectors for the space are $v_1,v_2, ..., v_n$ and their corresponding linear transformations are $T(v_1),T(v_2), ..., T(v_n)$, then we can get the linear transformation for any vector in the space</b>. For example, for the given basis, any vector $v$ can be defined as $v = c_1v_1 + c_2v_2 + ... + c_nv_n$ where $c_1, c_2, ..., c_n$ are <b>the coordiantes in the given set of basis</b>. This means, <b>coordinates comes from the basis, once basis changes, coordinates changes</b>. 

Let a linear transformation from n-dimensional space to m-dimensional space be represented as $T:\mathbb{R}^n \to \mathbb{R}^m$ and can be achieved by a matrix $m \times n$ matrix $A$. <b>The first step is to chosse the basis for $\mathbb{R}^n$ and $\mathbb{R}^m$</b>. Let the basis be $v_1, v_2, ..., v_n$ for $\mathbb{R}^n$ and $w_1, w_2, ..., w_m$ for $\mathbb{R}^m$. One of the good example of basis is the <b>eigenvector basis</b> as it leads to a diagonal transformation matrix $\Lambda$.

Once we have choosen the basid, we can find the transformation matrix $A$. First column of matrix $A$ tells us what happens to the first basis vector $v_1$, i.e. $T(v_1) = a_{11}w_1 + a_{21}w_2 + ... + a_{m1}w_m$. Similarly for second basis vector $v_2$, $T(v_2) = a_{12}w_1 + a_{22}w_2 + ... + a_{m2}w_m$. Finally, we can construct the matrix $A$ using the $a_{ij}$.

One of the classic example of linear transformation is derivatives. Due to derivatives being linear transformation, we can easily calculate derivatives of a larger set of functions just by knowing the derivatives of a few.

## 27.3 Change of Basis and Image Compression

Linear transformation can be viewed as a system which changes the basis. Image Compression is one of examples of it. For an image which has dense pixels, most of the surrounding pixels have similar values and don't have much new information encoded in it. Hence, we can significantly reduce the size of the image by combining the surrounding pixels. This operation may lead to <b>some loss of information or data</b>. For the case of videos, the sequence of images are highly correlated and hence can be compressed. Another way of lossless compress an image is by changing the basis. By changinf the basis, we can use let's say <b>Fourier Basis</b> or <b>Wavelet Basis</b> instead of <b>Standard Basis</b>. Let matrix $W$ be the matrix consisting of basis vectors for the wavelet basis, then change of basis can be represented by the equation $p = Wc$ where $p$ is the vector in the old basis and $c$ is the vector in the wavelet basis or new basis. Hence, the vector in the new-basis (wavelet basis) is $c = W^{-1}p$. <b>This means that the matrix with the chosen basis vectors should be easily invertible</b>. If the basis vectors are orthogonal or to be precise orthonormal, the inverse of the matrix can be calculated by just taking the transpose. Hence, while chosing a new set of basis vectors, we should choose the ones which are orthogonal to each other.

Let $A$ be the matrix having vectors in the old-basis and $B$ be the matrix having the vectors in the new-basis, then the matrices $A$ and $B$ are <b>similar</b>, i.e. $B=M^{-1}AM$ for some $M$.
