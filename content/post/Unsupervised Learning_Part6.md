+++
date = "2018-07-19T06:25:52+01:00"
description = "ISLR Unsupervised Learning"
draft = false
tags = ["ISLR", "Unsupervised Learning", "Exercises", "Applied"]
title = "ISLR Chapter 10: Unsupervised Learning (Part 6: Exercises - Applied)"
topics = ["ISLR"]

+++


#### Applied

Q7. In the chapter, we mentioned the use of correlation-based distance and Euclidean distance as dissimilarity measures for hierarchical clustering. It turns out that these two measures are almost equivalent: if each observation has been centered to have mean zero and standard deviation one, and if we let $r_{ij}$ denote the correlation between the ith and jth observations, then the quantity $1−r _{ij}$ is proportional to the squared Euclidean distance between the ith and jth observations. On the USArrests data, show that this proportionality holds.

Hint: The Euclidean distance can be calculated using the dist() function, and correlations can be calculated using the cor() function.


```python
import pandas as pd
from sklearn.preprocessing import scale
from scipy.spatial.distance import cdist

df = pd.read_csv("data/USArrests.csv")
df.rename(columns={'Unnamed: 0': 'State'}, inplace=True)
df[['Murder', 'Assault', 'UrbanPop', 'Rape']] = scale(df[['Murder', 'Assault', 'UrbanPop', 'Rape']], axis=1)
d_euclidean = cdist(df[['Murder', 'Assault', 'UrbanPop', 'Rape']], df[['Murder', 'Assault', 'UrbanPop', 'Rape']],
          metric='euclidean')
d_correlation = cdist(df[['Murder', 'Assault', 'UrbanPop', 'Rape']], df[['Murder', 'Assault', 'UrbanPop', 'Rape']],
          metric='correlation')
```


```python
df_relation = pd.DataFrame(d_euclidean**2/(1-d_correlation))
df_relation.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>40</th>
      <th>41</th>
      <th>42</th>
      <th>43</th>
      <th>44</th>
      <th>45</th>
      <th>46</th>
      <th>47</th>
      <th>48</th>
      <th>49</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>50.000000</td>
      <td>50.000000</td>
      <td>50.000000</td>
      <td>50.000000</td>
      <td>50.000000</td>
      <td>50.000000</td>
      <td>50.000000</td>
      <td>50.000000</td>
      <td>50.000000</td>
      <td>50.000000</td>
      <td>...</td>
      <td>50.000000</td>
      <td>50.000000</td>
      <td>50.000000</td>
      <td>50.000000</td>
      <td>50.000000</td>
      <td>50.000000</td>
      <td>50.000000</td>
      <td>50.000000</td>
      <td>50.000000</td>
      <td>50.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.917196</td>
      <td>1.377766</td>
      <td>0.768684</td>
      <td>0.835502</td>
      <td>0.633609</td>
      <td>0.551528</td>
      <td>0.636466</td>
      <td>0.638445</td>
      <td>0.929853</td>
      <td>0.848389</td>
      <td>...</td>
      <td>0.387890</td>
      <td>0.730992</td>
      <td>0.490706</td>
      <td>0.547206</td>
      <td>0.543436</td>
      <td>0.475501</td>
      <td>0.400632</td>
      <td>0.404453</td>
      <td>3.159859</td>
      <td>0.509535</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.491905</td>
      <td>3.625065</td>
      <td>2.114578</td>
      <td>2.286917</td>
      <td>1.747983</td>
      <td>1.476750</td>
      <td>0.579707</td>
      <td>1.779954</td>
      <td>2.525718</td>
      <td>2.318681</td>
      <td>...</td>
      <td>0.806068</td>
      <td>2.020526</td>
      <td>1.366438</td>
      <td>0.562794</td>
      <td>0.564614</td>
      <td>1.318453</td>
      <td>0.867514</td>
      <td>0.971656</td>
      <td>2.161903</td>
      <td>1.421946</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.032012</td>
      <td>0.089498</td>
      <td>0.016312</td>
      <td>0.028225</td>
      <td>0.031815</td>
      <td>0.052559</td>
      <td>0.176210</td>
      <td>0.029639</td>
      <td>0.034926</td>
      <td>0.032570</td>
      <td>...</td>
      <td>0.037003</td>
      <td>0.014472</td>
      <td>0.035702</td>
      <td>0.136008</td>
      <td>0.129466</td>
      <td>0.032512</td>
      <td>0.062962</td>
      <td>0.052856</td>
      <td>1.396576</td>
      <td>0.040784</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.194972</td>
      <td>0.371256</td>
      <td>0.121050</td>
      <td>0.153813</td>
      <td>0.123040</td>
      <td>0.114278</td>
      <td>0.501816</td>
      <td>0.107965</td>
      <td>0.195236</td>
      <td>0.163924</td>
      <td>...</td>
      <td>0.166428</td>
      <td>0.110865</td>
      <td>0.102117</td>
      <td>0.406559</td>
      <td>0.416032</td>
      <td>0.105314</td>
      <td>0.154284</td>
      <td>0.159624</td>
      <td>2.840412</td>
      <td>0.097088</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.661991</td>
      <td>1.044089</td>
      <td>0.525318</td>
      <td>0.588041</td>
      <td>0.389477</td>
      <td>0.295860</td>
      <td>0.981708</td>
      <td>0.393930</td>
      <td>0.673405</td>
      <td>0.599727</td>
      <td>...</td>
      <td>0.392379</td>
      <td>0.488617</td>
      <td>0.266171</td>
      <td>0.846270</td>
      <td>0.819825</td>
      <td>0.265969</td>
      <td>0.355743</td>
      <td>0.274510</td>
      <td>4.886207</td>
      <td>0.267781</td>
    </tr>
    <tr>
      <th>max</th>
      <td>16.482235</td>
      <td>24.396269</td>
      <td>13.874904</td>
      <td>15.058890</td>
      <td>11.408640</td>
      <td>9.635337</td>
      <td>2.790843</td>
      <td>11.655091</td>
      <td>16.705192</td>
      <td>15.286165</td>
      <td>...</td>
      <td>5.393157</td>
      <td>13.240649</td>
      <td>8.948936</td>
      <td>3.163273</td>
      <td>3.277025</td>
      <td>8.635749</td>
      <td>5.784783</td>
      <td>6.473499</td>
      <td>7.762162</td>
      <td>9.310704</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 50 columns</p>
</div>



Q8. In Section 10.2.3, a formula for calculating PVE was given in Equation 10.8. We also saw that the PVE can be obtained using the sdev output of the prcomp() function.

On the USArrests data, calculate PVE in two ways:

(a) Using the sdev output of the prcomp() function, as was done in Section 10.2.3.

(b) By applying Equation 10.8 directly. That is, use the prcomp() function to compute the principal component loadings. Then, use those loadings in Equation 10.8 to obtain the PVE.


```python
import numpy as np
from sklearn.decomposition import PCA

df = pd.read_csv("data/USArrests.csv")
df.rename(columns={'Unnamed: 0': 'State'}, inplace=True)
df[['Murder', 'Assault', 'UrbanPop', 'Rape']] = scale(df[['Murder', 'Assault', 'UrbanPop', 'Rape']], axis=1)

pca = PCA(n_components=1)
principalComponents = pca.fit_transform(df[['Murder', 'Assault', 'UrbanPop', 'Rape']])
print("Variance Explained (from the model): " + str(pca.explained_variance_ratio_[0]))
print("Variance Explained (calculated manually): " + str(np.var(principalComponents)/df.var().sum()))
```

    Variance Explained (from the model): 0.9453253030966985
    Variance Explained (calculated manually): 0.926418797034764


Q9. Consider the USArrests data. We will now perform hierarchical clusteringon the states.

(a) Using hierarchical clustering with complete linkage and Euclidean distance, cluster the states.


```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage

df = pd.read_csv("data/USArrests.csv")
df.rename(columns={'Unnamed: 0': 'State'}, inplace=True)
X = df[['Murder', 'Assault', 'UrbanPop', 'Rape']]

clustering = AgglomerativeClustering(linkage="complete", affinity="euclidean", compute_full_tree=True).fit(X)
```


```python
Z = linkage(X, 'complete')
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z, labels=df['State'].tolist())
```

{{% fluid_img "/img/Unsupervised%20Learning_files/Unsupervised%20Learning_34_0.png" %}}


(b) Cut the dendrogram at a height that results in three distinct clusters. Which states belong to which clusters?


```python
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z, labels=df['State'].tolist(), color_threshold=120)
```

{{% fluid_img "/img/Unsupervised%20Learning_files/Unsupervised%20Learning_36_0.png" %}}


(c) Hierarchically cluster the states using complete linkage and Euclidean distance, after scaling the variables to have standard deviation one.

(d) What effect does scaling the variables have on the hierarchical clustering obtained? In your opinion, should the variables be scaled before the inter-observation dissimilarities are computed? Provide a justification for your answer.

<b>Sol:</b> The variables should be scaled as the units are different for different features.


```python
df[['Murder', 'Assault', 'UrbanPop', 'Rape']] = scale(df[['Murder', 'Assault', 'UrbanPop', 'Rape']], axis=1)
X = df[['Murder', 'Assault', 'UrbanPop', 'Rape']]
Z = linkage(X, 'complete')
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z, labels=df['State'].tolist())
```

{{% fluid_img "/img/Unsupervised%20Learning_files/Unsupervised%20Learning_38_0.png" %}}



```python
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z, labels=df['State'].tolist(), color_threshold=1.1)
```

{{% fluid_img "/img/Unsupervised%20Learning_files/Unsupervised%20Learning_39_0.png" %}}


Q10. In this problem, you will generate simulated data, and then perform PCA and K-means clustering on the data.

(a) Generate a simulated data set with 20 observations in each of three classes (i.e. 60 observations total), and 50 variables.

Hint: There are a number of functions in R that you can use to generate data. One example is the rnorm() function; runif() is another option. Be sure to add a mean shift to the observations in each class so that there are three distinct classes.


```python
np.random.seed(0)
c1 = np.append(np.random.normal(0, 0.01, (20, 50)), np.full((20, 1), 1), axis=1)
c2 = np.append(np.random.normal(0.2, 0.01, (20, 50)), np.full((20, 1), 2), axis=1)
c3 = np.append(np.random.normal(-0.2, 0.01, (20, 50)), np.full((20, 1), 3), axis=1)

df = pd.DataFrame(np.vstack((c1,c2,c3)))
```

(b) Perform PCA on the 60 observations and plot the first two principal component score vectors. Use a different color to indicate the observations in each of the three classes. If the three classes appear separated in this plot, then continue on to part (c). If not, then return to part (a) and modify the simulation so that there is greater separation between the three classes. Do not continue to part (c) until the three classes show at least some separation in the first two principal component score vectors.


```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(df.iloc[:,0:50])
df_pca = pd.DataFrame(principalComponents, columns=['first', 'second'])
df_pca['labels'] = df[[50]]
df_pca.labels = df_pca.labels.astype(int)
```


```python
import seaborn as sns

fig = plt.figure(figsize=(15, 8))
sns.scatterplot(x="first", y="second", hue="labels", palette={1:'green', 2:'red', 3:'blue'}, data=df_pca)
plt.grid()
plt.xlabel("First Principal Componenet")
plt.ylabel("Second Principal Componenet")
plt.show()
```

{{% fluid_img "/img/Unsupervised%20Learning_files/Unsupervised%20Learning_44_0.png" %}}


(c) Perform K-means clustering of the observations with K = 3. How well do the clusters that you obtained in K-means clustering compare to the true class labels?

<b>Sol:</b> The results of clustering are well in sync with the class labels.


```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=0).fit(df.iloc[:,0:50])
print(kmeans.labels_)
```

    [2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]


(d) Perform K-means clustering with K = 2. Describe your results.

<b>Sol:</b> The observations belonging to first two class labels are clustered accordingly. The observations belonging to the third class label is clustered in the same group with the observations belonging to first class label.


```python
kmeans = KMeans(n_clusters=2, random_state=0).fit(df.iloc[:,0:50])
print(kmeans.labels_)
```

    [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]


(e) Now perform K-means clustering with K = 4, and describe your results.

<b>Sol:</b> The observations belonging to the first two class labels are clustered accordingly. The observations belonging to the third class label is further clustered into two subgroups.


```python
kmeans = KMeans(n_clusters=4, random_state=0).fit(df.iloc[:,0:50])
print(kmeans.labels_)
```

    [2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     1 1 1 0 3 0 3 0 3 3 0 0 0 3 0 0 3 3 0 3 3 0 3]


(f) Now perform K-means clustering with K = 3 on the first two principal component score vectors, rather than on the raw data. That is, perform K-means clustering on the 60 × 2 matrix of which the first column is the first principal component score vector, and the second column is the second principal component score vector. Comment on the results.

<b>Sol:</b> The observations belonging to the last two class labels are clustered accordingly. The observations belonging to the first class label is further clustered into two subgroups.


```python
kmeans = KMeans(n_clusters=4, random_state=0).fit(df_pca[['first', 'second']])
print(kmeans.labels_)
```

    [3 2 3 2 2 2 2 2 3 3 2 2 2 2 3 3 3 2 3 3 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]


Q11. On the book website, www.StatLearning.com, there is a gene expression data set (Ch10Ex11.csv) that consists of 40 tissue samples with measurements on 1,000 genes. The first 20 samples are from healthy patients, while the second 20 are from a diseased group.

(a) Load in the data using read.csv(). You will need to select header=F.


```python
df = pd.read_csv("data/GeneExpression.csv", header=None)
df = df.T
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>990</th>
      <th>991</th>
      <th>992</th>
      <th>993</th>
      <th>994</th>
      <th>995</th>
      <th>996</th>
      <th>997</th>
      <th>998</th>
      <th>999</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.961933</td>
      <td>-0.292526</td>
      <td>0.258788</td>
      <td>-1.152132</td>
      <td>0.195783</td>
      <td>0.030124</td>
      <td>0.085418</td>
      <td>1.116610</td>
      <td>-1.218857</td>
      <td>1.267369</td>
      <td>...</td>
      <td>1.325041</td>
      <td>-0.116171</td>
      <td>-1.470146</td>
      <td>-0.379272</td>
      <td>-1.465006</td>
      <td>1.075148</td>
      <td>-1.226125</td>
      <td>-3.056328</td>
      <td>1.450658</td>
      <td>0.717977</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.441803</td>
      <td>-1.139267</td>
      <td>-0.972845</td>
      <td>-2.213168</td>
      <td>0.593306</td>
      <td>-0.691014</td>
      <td>-1.113054</td>
      <td>1.341700</td>
      <td>-1.277279</td>
      <td>-0.918349</td>
      <td>...</td>
      <td>0.740838</td>
      <td>-0.162392</td>
      <td>-0.633375</td>
      <td>-0.895521</td>
      <td>2.034465</td>
      <td>3.003267</td>
      <td>-0.501702</td>
      <td>0.449889</td>
      <td>1.310348</td>
      <td>0.763482</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.975005</td>
      <td>0.195837</td>
      <td>0.588486</td>
      <td>-0.861525</td>
      <td>0.282992</td>
      <td>-0.403426</td>
      <td>-0.677969</td>
      <td>0.103278</td>
      <td>-0.558925</td>
      <td>-1.253500</td>
      <td>...</td>
      <td>-0.435533</td>
      <td>-0.235912</td>
      <td>1.446660</td>
      <td>-1.127459</td>
      <td>0.440849</td>
      <td>-0.123441</td>
      <td>-0.717430</td>
      <td>1.880362</td>
      <td>0.383837</td>
      <td>0.313576</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.417504</td>
      <td>-1.281121</td>
      <td>-0.800258</td>
      <td>0.630925</td>
      <td>0.247147</td>
      <td>-0.729859</td>
      <td>-0.562929</td>
      <td>0.390963</td>
      <td>-1.344493</td>
      <td>-1.067114</td>
      <td>...</td>
      <td>-3.065529</td>
      <td>1.597294</td>
      <td>0.737478</td>
      <td>-0.631248</td>
      <td>-0.530442</td>
      <td>-1.036740</td>
      <td>-0.169113</td>
      <td>-0.742841</td>
      <td>-0.408860</td>
      <td>-0.326473</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.818815</td>
      <td>-0.251439</td>
      <td>-1.820398</td>
      <td>0.951772</td>
      <td>1.978668</td>
      <td>-0.364099</td>
      <td>0.938194</td>
      <td>-1.927491</td>
      <td>1.159115</td>
      <td>-0.240638</td>
      <td>...</td>
      <td>-2.378938</td>
      <td>-0.086946</td>
      <td>-0.122342</td>
      <td>1.418029</td>
      <td>1.075337</td>
      <td>-1.270604</td>
      <td>0.599530</td>
      <td>2.238346</td>
      <td>-0.471111</td>
      <td>-0.158700</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1000 columns</p>
</div>



(b) Apply hierarchical clustering to the samples using correlationbased distance, and plot the dendrogram. Do the genes separate the samples into the two groups? Do your results depend on the type of linkage used?


```python
Z = linkage(df, 'complete')
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z, color_threshold=49)
```

{{% fluid_img "/img/Unsupervised%20Learning_files/Unsupervised%20Learning_56_0.png" %}}


(c) Your collaborator wants to know which genes differ the most across the two groups. Suggest a way to answer this question, and apply it here.

<b>Sol:</b> This can be achieved by doing PCA over the data set and reporting the genes whose loadings are maximum (as loading denotes the weight of each feature in a specified principal component) as the important genes.
