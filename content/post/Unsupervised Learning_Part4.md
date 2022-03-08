+++
date = "2018-07-12T13:01:01+01:00"
description = "ISLR Unsupervised Learning"
draft = false
tags = ["ISLR", "Unsupervised Learning", "Clustering Methods", "Hierarchical Clustering"]
title = "ISLR Chapter 10: Unsupervised Learning (Part 4: Clustering Methods, Hierarchical Clustering)"
topics = ["ISLR"]

+++

#### 10.3.2 Hierarchical Clustering

K-means clustering has a disadvantage that there is a need to pre-specify the number of clusters $K$. Hierarchical clutsring is an alternative approach which is free from this problem which results in an altarnative tree-based representation of the observations, called as <b>dendrogram</b>.

The most common technique used for hierarchical clustering is <b>bottom-up</b> or <b>agglomerative</b> clustering. It is based on the fact that the dendrogram (generally depicted as an upside-down tree) is built starting from leaves and combining the clusters up to the trunk.

##### Interpreting a Dendrogram

Each leaf of a dendrogram represents an observation. As we move up the tree, some leaves begin to <b>fuse</b> into branches. This correspoonds to observations that are similar to each other. As we further move up, more fusion occurs (either of branches or of a branch and a leaf). <b>Earlier (lower) the fusion occurs, more similar the group of observations to each other</b>. Obervations or groups that fuse later (near the top of the tree) can be quite different from each other. Or, <b>height of the fusion on vertical axis</b> represents how different the two observations are.

One common misunderstanding while interpreting the dendrogram is deriving conclusions based on the distance along <b>horizontal axis</b>. It should be noted that distace between two observations on horizontal axis purely depends on the initial reprentation of observations. This does not measure any similarity between the observations. The similarity between the observations is only measured by the <b>location of the fusion on the vertical axis</b>.

To obtain the clustres on the basis of a dendrogram, we make a <b>horizontal cut</b> across the dendrogram. The lower the cut is made, the more number of clusters obtained. Hence, the height of the cut to the dendrogram serves the same role as the $K$ in $K$-means clustering. It controls the number of clustres obtained.

##### The Hierarchical Clustering Algorithm

To obtain hierarchical clustering, first of all, we define some sort of <b>dissimilarity measure</b>. The most common choice is the Euclidean distance. Each of the individual $n$ observations is then treated as its own cluster. The two clusters that are the most similar (based on the dissimilarity measure) to each other are then <b>fused</b> to obtain $n-1$ clusters. The process is repeated until all the observations belong to one single cluster, and the dendrogram is complete.

To fuse individual clusters, the notion of dissimilarity between pair of observations needs to be extended to a pair of <b>groups of observations</b>. This extension is achieved by developing the notion of <b>linkage</b> which defines the dissimilarity between two groups of observations. The most common types of linkage are: <b>complete, average, single</b> and <b>centroid</b>. For the computation of complete, single and average linkage, we find the <b>interclustre dissimilarity</b>, the pairwise dissimilarities between the observations in cluster $A$ and $B$, and record the <b>largest, mean</b> and <b>smallest</b> value as the measure respectively. In the centroid linkage, the dissimilarity between the centroid of clutser $A$ and $B$ serves the purpose.

Average, complete and single linkage are most popular among statiscians. Average and complete linkage generally give a more balanced dendrogram and hence is preffered over single linkage. Centroid linkage suffers from the problem of <b>inversion</b>, whereby two clusters are fused at a height <b>below</b> either of the individual clutsers in the dendrogram.

##### Choice of Dissimilarity Measure

<b>Correlation-based distance</b> can serve as a dissimilarity measure as well. In this case, two observations are considered to be similar if their features are highly correlated even if the observed values are far enough in terms of Euclidean distance. Hence, correlation-based distance focuses on the shapes of observation profiles rather than their magnitudes. The choice of dissimilarity measure and the scaling of features play important roles in determining the results of clustering and hence they should be chosen carefully.

#### 10.3.3 Practical Issues in Clustering

##### Small Decisions with Big Consequences

Certain decisions must be made in order to perform clustering. Some of them are:

 - Should the observations be standardized prior to clustering?

 - For hierarchical clustering, dissimilarity measure, type of linkage and the cut made in the dendrogram is important.

 - For the K-means clustering, the number of clusters is important.

##### Validating the Clusters Obtained

There does not exist a consesus on a single best standardized process to validate the results of clustering.

##### Other Considerations in Clustering

In the clustering process, there may arise a case when some outliers, which should not belong to any cluster, have a significant effect on the results of clustering. Clustering methods are not robust to the perturbations to the data. A new model fit on a subset of data set may provide a quite different result. The results of clustering should not be taken as the absoulte truth about the data set and should constitute a starting point for further study.
