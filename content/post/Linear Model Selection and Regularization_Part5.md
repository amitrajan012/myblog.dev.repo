+++
date = "2018-05-26T11:08:14+01:00"
description = "ISLR Linear Model Selection and Regularization"
draft = false
tags = ["ISLR", "Resampling", "Linear Model Selection", "Regularization", "Exercises", "Applied"]
title = "ISLR Chapter 6: Linear Model Selection and Regularization (Part 5: Exercises - Applied)"
topics = ["ISLR"]

+++

#### Applied

Q8. In this exercise, we will generate simulated data, and will then use this data to perform best subset selection.

(a) Use the rnorm() function to generate a predictor X of length n = 100, as well as a noise vector  of length n = 100.


```python
import numpy as np

np.random.seed(5)
X = np.random.normal(0, 1, 100)
e = np.random.normal(0, 1, 100)
```

(b) Generate a response vector Y of length n = 100 according to the model $Y = β_0 + β_1X + β_2X^2 + β_3X^3 + \epsilon$, where $β_0, β_1, β_2,$ and $β_3$ are constants of your choice.


```python
beta0 = 5
beta1 = 2
beta2 = 3
beta3 = 4

y = beta0 + beta1*X + beta2*(X**2) + beta3*(X**3) + e
```

(c) Use the regsubsets() function to perform best subset selection in order to choose the best model containing the predictors $X,X^2, . . ., X^{10}$. What is the best model obtained according to $C_p$, BIC, and adjusted $R^2$? Show some plots to provide evidence for your answer, and report the coefficients of the best model obtained. Note you will need to use the data.frame() function to create a single data set containing both X and Y .

<b>Sol:</b> Test MSE is minimum for model with size <b>3</b> and having predictors <b>[0, 1, 2]</b>, which is perfectly in accordance with the generated data.


```python
import itertools as it
from sklearn.linear_model import LinearRegression

def select_subset_sizeK(X_, y_, k):
    model = LinearRegression()
    best_score = 0.0
    M_k = []
    for combo in it.combinations(range(X_.shape[1]), k):
        X = X_[:, list(combo)]
        model.fit(X, y_)
        s = model.score(X, y_)
        if s > best_score:
            M_k = list(combo)
            best_score = s
    return M_k

def subset_selection(X_, y_):
    # Fit model with intercept only (Null model)
    train_MSE = {}
    model_cols = {}
    y_pred = np.mean(y_)
    train_MSE[0] = np.sum((y_ - y_pred)**2) / len(y_)
    for s in range(1, X_.shape[1]):
        cols = select_subset_sizeK(X_, y_, s)
        X = X_[:, cols]
        model = LinearRegression()
        model.fit(X, y_)
        y_pred = model.predict(X)
        train_MSE[s] = mean_squared_error(y_pred, y_)
        model_cols[s] = cols
    return (model_cols, train_MSE)
```


```python
from sklearn.model_selection import train_test_split

X_ = np.vstack((X, X**2, X**3, X**4, X**5, X**6, X**7, X**8, X**9, X**10)).T
X_train, X_test, y_train, y_test = train_test_split(X_, y, test_size=0.1)

t = subset_selection(X_train, y_train)
models = t[0]
train_MSE = t[1]

fig = plt.figure(figsize=(15, 8))

lists = sorted(train_MSE.items()) # sorted by key, return a list of tuples
x, y = zip(*lists) # unpack a list of pairs into two tuples
ax = fig.add_subplot(121)
plt.plot(x, y, color='r')
plt.grid()
ax.set_xlabel('Model Size')
ax.set_ylabel('Training MSE')
ax.set_title('Training MSE vs Model Size')

test_MSE = {}
for size, cols in models.items():
    if size == 0:
        test_MSE[size] = np.sum((y_test - cols)**2) / len(y_test)
    else:
        model = LinearRegression()
        model.fit(X_train[:, cols], y_train)
        y_pred = model.predict(X_test[:, cols])
        test_MSE[size] = mean_squared_error(y_pred, y_test)

lists = sorted(test_MSE.items()) # sorted by key, return a list of tuples
x, y = zip(*lists) # unpack a list of pairs into two tuples
ax = fig.add_subplot(122)
plt.plot(x, y, color='g')
plt.grid()
ax.set_xlabel('Model Size')
ax.set_ylabel('Test MSE')
ax.set_title('Test MSE vs Model Size')
```




    Text(0.5,1,'Test MSE vs Model Size')



{{% fluid_img "/img/Linear%20Model%20Selection%20and%20Regularization_files/Linear%20Model%20Selection%20and%20Regularization_68_1.png" %}}

```python
print("Test MSE is minimum for model size: " +str(min(test_MSE, key=test_MSE.get)))
cols = models.get(min(test_MSE, key=test_MSE.get))
print("Columns used in the model: " +str(cols))
model = LinearRegression()
model.fit(X_train[:, cols], y_train)
print("Model Coefficients: " +str(model.coef_))
```

    Test MSE is minimum for model size: 3
    Columns used in the model: [0, 1, 2]
    Model Coefficients: [2.00715291 3.06050471 4.07996844]


(e) Now fit a lasso model to the simulated data, again using $X,X^2, . . ., X^{10}$ as predictors. Use cross-validation to select the optimal value of λ. Report the resulting coefficient estimates, and discuss the results obtained.

<b>Sol:</b> The lasso model has significant coefficients for predictors 0,1,2,3,4. All other coefficients are insignificant.


```python
from sklearn.linear_model import LassoCV

n_alphas = 200
alphas = np.logspace(-10, 2, n_alphas)
# Leave one out cross-validation
model = LassoCV(alphas=alphas, fit_intercept=True, cv=None)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Test Error: " +str(mean_squared_error(y_test, predictions)))
print("Model coefficients: " + str(model.coef_))
```

    Test Error: 4687.204168578645
    Model coefficients: [ 1.83007496e+00  2.54077338e+00  3.91756182e+00  2.22585449e-01
      2.89062042e-01 -3.33340397e-02 -2.88216147e-02 -3.80527416e-03
     -9.90808115e-03  3.31479566e-03]


    /Users/amitrajan/Desktop/PythonVirtualEnv/Python3_VirtualEnv/lib/python3.6/site-packages/sklearn/linear_model/coordinate_descent.py:491: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)
    /Users/amitrajan/Desktop/PythonVirtualEnv/Python3_VirtualEnv/lib/python3.6/site-packages/sklearn/linear_model/coordinate_descent.py:491: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)


Q9. In this exercise, we will predict the number of applications received using the other variables in the College data set.


```python
import pandas as pd

college = pd.read_csv("data/College.csv")
college.head()
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
      <th>Unnamed: 0</th>
      <th>Private</th>
      <th>Apps</th>
      <th>Accept</th>
      <th>Enroll</th>
      <th>Top10perc</th>
      <th>Top25perc</th>
      <th>F.Undergrad</th>
      <th>P.Undergrad</th>
      <th>Outstate</th>
      <th>Room.Board</th>
      <th>Books</th>
      <th>Personal</th>
      <th>PhD</th>
      <th>Terminal</th>
      <th>S.F.Ratio</th>
      <th>perc.alumni</th>
      <th>Expend</th>
      <th>Grad.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Abilene Christian University</td>
      <td>Yes</td>
      <td>1660</td>
      <td>1232</td>
      <td>721</td>
      <td>23</td>
      <td>52</td>
      <td>2885</td>
      <td>537</td>
      <td>7440</td>
      <td>3300</td>
      <td>450</td>
      <td>2200</td>
      <td>70</td>
      <td>78</td>
      <td>18.1</td>
      <td>12</td>
      <td>7041</td>
      <td>60</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Adelphi University</td>
      <td>Yes</td>
      <td>2186</td>
      <td>1924</td>
      <td>512</td>
      <td>16</td>
      <td>29</td>
      <td>2683</td>
      <td>1227</td>
      <td>12280</td>
      <td>6450</td>
      <td>750</td>
      <td>1500</td>
      <td>29</td>
      <td>30</td>
      <td>12.2</td>
      <td>16</td>
      <td>10527</td>
      <td>56</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Adrian College</td>
      <td>Yes</td>
      <td>1428</td>
      <td>1097</td>
      <td>336</td>
      <td>22</td>
      <td>50</td>
      <td>1036</td>
      <td>99</td>
      <td>11250</td>
      <td>3750</td>
      <td>400</td>
      <td>1165</td>
      <td>53</td>
      <td>66</td>
      <td>12.9</td>
      <td>30</td>
      <td>8735</td>
      <td>54</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Agnes Scott College</td>
      <td>Yes</td>
      <td>417</td>
      <td>349</td>
      <td>137</td>
      <td>60</td>
      <td>89</td>
      <td>510</td>
      <td>63</td>
      <td>12960</td>
      <td>5450</td>
      <td>450</td>
      <td>875</td>
      <td>92</td>
      <td>97</td>
      <td>7.7</td>
      <td>37</td>
      <td>19016</td>
      <td>59</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Alaska Pacific University</td>
      <td>Yes</td>
      <td>193</td>
      <td>146</td>
      <td>55</td>
      <td>16</td>
      <td>44</td>
      <td>249</td>
      <td>869</td>
      <td>7560</td>
      <td>4120</td>
      <td>800</td>
      <td>1500</td>
      <td>76</td>
      <td>72</td>
      <td>11.9</td>
      <td>2</td>
      <td>10922</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>



(a) Split the data set into a training set and a test set.


```python
college = college.rename(columns={'Unnamed: 0': 'Name'})
college['Private'] = college['Private'].map({'Yes': 1, 'No': 0})
msk = np.random.rand(len(college)) < 0.8
train = college[msk]
test = college[~msk]
```

(b) Fit a linear model using least squares on the training set, and report the test error obtained.


```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

model = LinearRegression(fit_intercept=True)
model.fit(train.drop(['Name', 'Apps'], axis=1), train['Apps'])
predictions = model.predict(test.drop(['Name', 'Apps'], axis=1))

print("Test Error: " +str(mean_squared_error(test['Apps'], predictions)))
```

    Test Error: 1178528.8421813124


(c) Fit a ridge regression model on the training set, with λ chosen by cross-validation. Report the test error obtained.


```python
from sklearn.linear_model import RidgeCV

n_alphas = 200
alphas = np.logspace(-10, 2, n_alphas)
# Leave one out cross-validation
model = RidgeCV(alphas=alphas, fit_intercept=True, cv=None, store_cv_values=True)
model.fit(train.drop(['Name', 'Apps'], axis=1), train['Apps'])

predictions = model.predict(test.drop(['Name', 'Apps'], axis=1))
print("Test Error: " +str(mean_squared_error(test['Apps'], predictions)))
```

    Test Error: 1188298.4979038904


(d) Fit a lasso model on the training set, with λ chosen by crossvalidation. Report the test error obtained, along with the number of non-zero coefficient estimates.


```python
from sklearn.linear_model import LassoCV

# Leave one out cross-validation
model = LassoCV(alphas=alphas, fit_intercept=True, cv=None)
model.fit(train.drop(['Name', 'Apps'], axis=1), train['Apps'])

predictions = model.predict(test.drop(['Name', 'Apps'], axis=1))
print("Test Error: " +str(mean_squared_error(test['Apps'], predictions)))
print("Number of Non-zero coefficients: " + str(len(model.coef_)))
```

    Test Error: 1207847.543688796
    Number of Non-zero coefficients: 17


    /Users/amitrajan/Desktop/PythonVirtualEnv/Python3_VirtualEnv/lib/python3.6/site-packages/sklearn/linear_model/coordinate_descent.py:491: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)


(e) Fit a PCR model on the training set, with M chosen by crossvalidation. Report the test error obtained, along with the value of M selected by cross-validation.


```python
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def PCR_CV(X_train, Y_train, X_test, Y_test, M):
    X_train_scaled = preprocessing.scale(X_train)
    X_test_scaled = preprocessing.scale(X_test)

    MSE = {}
    test_MSE = {}

    for m in M: # Iterate over number of principal components
        pca = PCA(n_components=m)
        X_train_reduced = pca.fit_transform(X_train_scaled)
        X_test_reduced = pca.fit_transform(X_test_scaled)

        mse = 0
        test_mse = 0
        loo = LeaveOneOut() # Leave one out cross-validation
        for train_index, test_index in loo.split(X_train_reduced):
            X, X_CV = X_train_reduced[train_index], X_train_reduced[test_index]
            Y, Y_CV = Y_train[train_index], Y_train[test_index]
            model = LinearRegression(fit_intercept=True)
            model.fit(X, Y)
            p = model.predict(X_CV)
            mse += mean_squared_error(p, Y_CV)
        MSE[m] = mse/len(X_train_reduced)

        # Compute test MSE for the model
        model = LinearRegression(fit_intercept=True)
        model.fit(X_train_reduced, Y_train)
        p = model.predict(X_test_reduced)
        test_MSE[m] = mean_squared_error(p, Y_test)

    # Plot validation MSE
    lists = sorted(MSE.items()) # sorted by key, return a list of tuples
    x, y = zip(*lists) # unpack a list of pairs into two tuples
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(121)
    plt.plot(x, y, color='r')
    plt.grid()
    ax.set_xlabel('Number of Principal Components')
    ax.set_ylabel('Validation MSE')
    ax.set_title('Validation MSE vs Principal Components')

    lists = sorted(test_MSE.items()) # sorted by key, return a list of tuples
    x, y = zip(*lists) # unpack a list of pairs into two tuples
    ax = fig.add_subplot(122)
    plt.plot(x, y, color='g')
    plt.grid()
    ax.set_xlabel('Number of Principal Components')
    ax.set_ylabel('Test MSE')
    ax.set_title('Test MSE vs Principal Components')
    plt.show()

M = np.arange(1, 17, 1) # Principal components
PCR_CV(train.drop(['Name', 'Apps'], axis=1), train['Apps'].values, test.drop(['Name', 'Apps'], axis=1),
       test['Apps'], M)
```

{{% fluid_img "/img/Linear%20Model%20Selection%20and%20Regularization_files/Linear%20Model%20Selection%20and%20Regularization_83_0.png" %}}


(f) Fit a PLS model on the training set, with M chosen by crossvalidation. Report the test error obtained, along with the value of M selected by cross-validation.


```python
from sklearn.cross_decomposition import PLSRegression

def PLS_CV(X_train, Y_train, X_test, Y_test, M):
    X_train_scaled = preprocessing.scale(X_train)
    X_test_scaled = preprocessing.scale(X_test)

    MSE = {}
    test_MSE = {}

    for m in M: # Iterate over number of principal components
        mse = 0
        test_mse = 0
        loo = LeaveOneOut() # Leave one out cross-validation
        for train_index, test_index in loo.split(X_train_scaled):
            X, X_CV = X_train_scaled[train_index], X_train_scaled[test_index]
            Y, Y_CV = Y_train[train_index], Y_train[test_index]
            model = PLSRegression(n_components=m)
            model.fit(X, Y)
            p = model.predict(X_CV)
            mse += mean_squared_error(p, Y_CV)
        MSE[m] = mse/len(X_train_scaled)

        # Compute test MSE for the model
        model = PLSRegression(n_components=m)
        model.fit(X_train_scaled, Y_train)
        p = model.predict(X_test_scaled)
        test_MSE[m] = mean_squared_error(p, Y_test)

    # Plot validation MSE
    lists = sorted(MSE.items()) # sorted by key, return a list of tuples
    x, y = zip(*lists) # unpack a list of pairs into two tuples
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(121)
    plt.plot(x, y, color='r')
    plt.grid()
    ax.set_xlabel('M')
    ax.set_ylabel('Validation MSE')
    ax.set_title('Validation MSE vs M')

    lists = sorted(test_MSE.items()) # sorted by key, return a list of tuples
    x, y = zip(*lists) # unpack a list of pairs into two tuples
    ax = fig.add_subplot(122)
    plt.plot(x, y, color='g')
    plt.grid()
    ax.set_xlabel('M')
    ax.set_ylabel('Test MSE')
    ax.set_title('Test MSE vs M')
    plt.show()

M = np.arange(1, 17, 1) # Principal components
PLS_CV(train.drop(['Name', 'Apps'], axis=1), train['Apps'].values, test.drop(['Name', 'Apps'], axis=1),
       test['Apps'], M)
```

{{% fluid_img "/img/Linear%20Model%20Selection%20and%20Regularization_files/Linear%20Model%20Selection%20and%20Regularization_85_0.png" %}}


(g) Comment on the results obtained. How accurately can we predict the number of college applications received? Is there much difference among the test errors resulting from these five approaches?

<b>Sol:</b> The test errors (with order of magnitude $10^7$) for various methods are as follows:

 - Least squares linear model : <b>0.118</b>

 - Ridge regression model : <b>0.119</b>

 - Tha lasso: <b>0.121</b>

 - PCR: <b>0.67</b>

 - PLS: <b>0.11</b>

It can be conluded that all the other models perform well as compared to PCR.

Q10. We have seen that as the number of features used in a model increases, the training error will necessarily decrease, but the test error may not. We will now explore this in a simulated data set.

(a) Generate a data set with $p = 15$ features, $n = 1,000$ observations, and an associated quantitative response vector generated according to the model
$$Y = Xβ + \epsilon$$
where β has some elements that are exactly equal to zero.


```python
X = np.random.normal(size=(1000, 15))
beta = np.random.normal(size=15)
beta[3] = 0
beta[5] = 0
beta[9] = 0
e = np.random.normal(size=1000)
y = np.dot(X, beta) + e
```

(b) Split your data set into a training set containing 100 observations and a test set containing 900 observations.


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
```

(c) Perform best subset selection on the training set, and plot the training set MSE associated with the best model of each size.

(d) Plot the test set MSE associated with the best model of each size.


```python
import itertools as it
from sklearn.linear_model import LinearRegression

def select_subset_sizeK(X_, y_, k):
    model = LinearRegression()
    best_score = 0.0
    M_k = []
    for combo in it.combinations(range(X_.shape[1]), k):
        X = X_[:, list(combo)]
        model.fit(X, y_)
        s = model.score(X, y_)
        if s > best_score:
            M_k = list(combo)
            best_score = s
    return M_k

def subset_selection(X_, y_):
    # Fit model with intercept only (Null model)
    train_MSE = {}
    model_cols = {}
    y_pred = np.mean(y_)
    train_MSE[0] = np.sum((y_ - y_pred)**2) / len(y_)
    for s in range(1, X_.shape[1]):
        cols = select_subset_sizeK(X_, y_, s)
        X = X_[:, cols]
        model = LinearRegression()
        model.fit(X, y_)
        y_pred = model.predict(X)
        train_MSE[s] = mean_squared_error(y_pred, y_)
        model_cols[s] = cols
    return (model_cols, train_MSE)

t = subset_selection(X_train, y_train)
models = t[0]
train_MSE = t[1]

fig = plt.figure(figsize=(15, 8))

lists = sorted(train_MSE.items()) # sorted by key, return a list of tuples
x, y = zip(*lists) # unpack a list of pairs into two tuples
ax = fig.add_subplot(121)
plt.plot(x, y, color='r')
plt.grid()
ax.set_xlabel('Model Size')
ax.set_ylabel('Training MSE')
ax.set_title('Training MSE vs Model Size')

test_MSE = {}
for size, cols in models.items():
    if size == 0:
        test_MSE[size] = np.sum((y_test - cols)**2) / len(y_test)
    else:
        model = LinearRegression()
        model.fit(X_train[:, cols], y_train)
        y_pred = model.predict(X_test[:, cols])
        test_MSE[size] = mean_squared_error(y_pred, y_test)

lists = sorted(test_MSE.items()) # sorted by key, return a list of tuples
x, y = zip(*lists) # unpack a list of pairs into two tuples
ax = fig.add_subplot(122)
plt.plot(x, y, color='g')
plt.grid()
ax.set_xlabel('Model Size')
ax.set_ylabel('Test MSE')
ax.set_title('Test MSE vs Model Size')
```




    Text(0.5,1,'Test MSE vs Model Size')



{{% fluid_img "/img/Linear%20Model%20Selection%20and%20Regularization_files/Linear%20Model%20Selection%20and%20Regularization_92_1.png" %}}


(e) For which model size does the test set MSE take on its minimum value? Comment on your results. If it takes on its minimum value for a model containing only an intercept or a model containing all of the features, then play around with the way that you are generating the data in (a) until you come up with a scenario in which the test set MSE is minimized for an intermediate model size.


```python
print("Test MSE is minimum for model size: " +str(min(test_MSE, key=test_MSE.get)))
```

    Test MSE is minimum for model size: 13


(f) How does the model at which the test set MSE is minimized compare to the true model used to generate the data? Comment on the coefficient values.

<b>Sol:</b> The model is well in accordance with the way data is generated, First of all, the columns that are not used for model generation are: <b>5, 9</b>. While generating data, we set the coefficients 3,5, and 9 to 0 and hence the model captures this well. Apart from this, the coefficient of feature 3 is <b>-0.07353929</b>, which is quite low as well.


```python
cols = models.get(min(test_MSE, key=test_MSE.get))
print("Columns used in the model: " +str(cols))
model = LinearRegression()
model.fit(X_train[:, cols], y_train)
print("Model Coefficients: " +str(model.coef_))
```

    Columns used in the model: [0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14]
    Model Coefficients: [-2.50852562 -0.43695144  1.40013156 -0.07353929 -0.85895357 -1.89061122
     -0.30136561  1.12543061  0.09474982 -0.70489182 -0.6278358  -1.40983561
      0.17529716]


Q11. We will now try to predict per capita crime rate in the Boston data set.

(a) Try out some of the regression methods explored in this chapter, such as best subset selection, the lasso, ridge regression, and PCR. Present and discuss results for the approaches that you consider.


```python
boston = pd.read_csv("data/Boston.csv")
boston.dropna(inplace=True)
```


```python
np.random.seed(5)
X_train, X_test, y_train, y_test = train_test_split(boston.iloc[:,1:], boston['crim'], test_size=0.1)

# The lasso
n_alphas = 200
alphas = np.logspace(-10, 2, n_alphas)
# Leave one out cross-validation
model = LassoCV(alphas=alphas, fit_intercept=True, cv=None)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Test Error: " +str(mean_squared_error(y_test, predictions)))
print("Model coefficients: " + str(model.coef_))
```

    Test Error: 54.95744135110768
    Model coefficients: [ 0.04198738 -0.07640927 -0.         -0.          0.32159534 -0.00722933
     -0.67844741  0.5220804  -0.0022908  -0.09016879 -0.00174827  0.13527718
     -0.15964042]



```python
# Ridge Regression
model = RidgeCV(alphas=alphas, fit_intercept=True, cv=None)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Test Error: " +str(mean_squared_error(y_test, predictions)))
print("Model coefficients: " + str(model.coef_))
```

    Test Error: 55.55453079915659
    Model coefficients: [ 0.0406467  -0.08234663 -0.20301002 -0.09532246  0.55364958 -0.00933135
     -0.70209263  0.52848777 -0.00235438 -0.14191045 -0.00127377  0.14595535
     -0.17415457]



```python
# PCR
M = np.arange(1, 14, 1) # Principal components
PCR_CV(X_train, y_train.values, X_test, y_test.values, M)
```

{{% fluid_img "/img/Linear%20Model%20Selection%20and%20Regularization_files/Linear%20Model%20Selection%20and%20Regularization_101_0.png" %}}


```python
# PLS
M = np.arange(1, 14, 1) # Principal components
PLS_CV(X_train, y_train.values, X_test, y_test.values, M)
```

{{% fluid_img "/img/Linear%20Model%20Selection%20and%20Regularization_files/Linear%20Model%20Selection%20and%20Regularization_102_0.png" %}}


(b) Propose a model (or set of models) that seem to perform well on this data set, and justify your answer. Make sure that you are evaluating model performance using validation set error, crossvalidation, or some other reasonable alternative, as opposed to using training error.

<b>Sol:</b> Except PCR, the lasso, PLS and ridge regression performs decently well.

(c) Does your chosen model involve all of the features in the data set? Why or why not?

<b>Sol:</b> If we look at the lasso model with test MSE of <b>54.95744</b>, the coefficients of <b>chas, nox</b> are 0 and those of age, tax and black are pretty low.
