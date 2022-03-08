+++
date = "2018-07-09T17:09:41+01:00"
description = "ISLR Unsupervised Learning"
draft = false
tags = ["ISLR", "Unsupervised Learning", "Clustering Methods", "K-Means Clustering"]
title = "ISLR Chapter 10: Unsupervised Learning (Part 3: Clustering Methods, K-Means Clustering)"
topics = ["ISLR"]

+++


### 10.3 Clustering Methods

<b>Clustering</b> is a technique for finding <b>subgroups</b> or <b>clusters</b> in a data set based on similarity between individual observations. For clustering, we need to define the measure of similarity which depends on the knowledge of the data set. Two best known clustering methods are <b>K-means clustering</b> and <b>hierarchical clustering</b>. In K-means clustering, we partition the observations into a pre-defined number of clusters. In hierarchical clustering, the number of clusters is unknown and the results of clustering is represented as a <b>dendrogram</b>, which is a tree-like visualization technique that allows us to view the clustering results for various number of clusters (from 1 to $n$).

#### 10.3.1 K-Means Clustering

To perform K-means clustering, we first specify the number of clusters $K$ and then the K-means algorithm is used to assign each observation to exactly one of the $K$ clusters. Let $C_1, C_2, ..., C_K$ denote sets containing the indices of the observations in each cluster, then these sets satisfy two properties:

 - $C_1 \bigcup C_2 \bigcup ... \bigcup C_K = \{1,2,3,...,n\}$.


 - $C_k \bigcap C _{k^{'}} = \Phi$ for all $k \neq k^{'}$.

In a good clustering, the <b>within-cluster variation</b> is as small as possible. The within-cluster variation, denoted as $W(C_k)$, is a measure of the amount by which the observations within a cluster differ from each other. Hence, for clustering, we want to solve the problem

$$minimize _{C_1, C_2, ..., C_K} \bigg ( \sum _{k=1}^{K} W(C_k)\bigg )$$

In order to solve the above problem, we need to define the within-clustre variation more concretely. The most common choice is <b>squared Euclidean distance</b>, which is defined as

$$W(C_k) = \frac{1}{|C_k|} \sum _{i, i^{'} \in C_k} \sum _{j=1}^{p} (x _{ij} - x _{i^{,}j})^2$$

where $|C_k|$ denotes the number of observations in the $k^{th}$ cluster. Hence, the K-means clustering problem can be defined as:

$$minimize _{C_1, C_2, ..., C_K} \bigg ( \sum _{k=1}^{K} \frac{1}{|C_k|} \sum _{i, i^{'} \in C_k} \sum _{j=1}^{p} (x _{ij} - x _{i^{,}j})^2 \bigg )$$

The above mentioned problem is a very difficlut problem to solve as there are $K^n$ ways to divide $n$ observations into $K$ clusters. Instead, an algorithm that gives a local optimal soultion exists and is given as:

 - Randomly assign a number from 1 to $K$ to all the individual observations. This serves as the initial cluster assignments for the observations


 - Iterate the cluster assignments (by calculating the cluster <b>centroids</b> and reassigning the observations to the clusters to which it is the nearest) until the cluster assignments stop changing.

As the K-means clustering algorithm finds a local-optimal solution, the result will depend on the initial random cluster assignments. Hence, it is important to run the algorithm multiple times with different initial random seeds and the select the <b>best solution</b> (for which the objective is minimum).
