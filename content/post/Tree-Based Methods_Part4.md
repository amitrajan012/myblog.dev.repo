+++
date = "2018-06-16T01:34:53+01:00"
description = "ISLR Tree-Based Methods"
draft = false
tags = ["ISLR", "Tree-Based Methods", "Exercises", "Applied"]
title = "ISLR Chapter 8: Tree-Based Methods (Part 4: Exercises - Applied)"
topics = ["ISLR"]

+++


#### Applied
Q 7. In the lab, we applied random forests to the Boston data using mtry=6 and using ntree=25 and ntree=500. Create a plot displaying the test error resulting from random forests on this data set for a more comprehensive range of values for mtry and ntree. You can model your plot after Figure 8.10. Describe the results obtained.


```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from math import sqrt

boston = pd.read_csv("data/Boston.csv")
boston.dropna(inplace=True)

def random_forest_ntree(X_train, Y_train, X_test, Y_test, nTrees, max_feature):
    test_MSE = {}
    for nTree in nTrees:
        regr = RandomForestRegressor(max_features=max_feature, n_estimators=nTree)
        regr.fit(X_train, Y_train)
        p = regr.predict(X_test)
        test_MSE[nTree] = mean_squared_error(p, Y_test)
    return test_MSE

np.random.seed(5)
predictors = 13

X_train, X_test, y_train, y_test = train_test_split(boston.loc[:, boston.columns != 'medv'],
                                                    boston[['medv']], test_size=0.1)

test_MSE_p = random_forest_ntree(X_train, y_train.values.ravel(), X_test, y_test.values.ravel(), np.arange(1,300,5)
                                 , predictors)
test_MSE_pby2 = random_forest_ntree(X_train, y_train.values.ravel(), X_test, y_test.values.ravel(), np.arange(1,300,5)
                                 , int(predictors/2))
test_MSE_psqrt = random_forest_ntree(X_train, y_train.values.ravel(), X_test, y_test.values.ravel(), np.arange(1,300,5)
                                 , int(sqrt(predictors)))

fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(111)

lists = sorted(test_MSE_p.items())
x, y = zip(*lists)
plt.plot(x, y, color='r', label='m = p')

lists = sorted(test_MSE_pby2.items())
x, y = zip(*lists)
plt.plot(x, y, color='g', label='m = p/2')

lists = sorted(test_MSE_psqrt.items())
x, y = zip(*lists)
plt.plot(x, y, color='b', label='m = sqrt(p)')

ax.set_xlabel('Number of Trees')
ax.set_ylabel('Test MSE')
ax.set_title('Test MSE vs Number of Trees')

plt.grid(b=True)
plt.legend()
plt.show()
```

{{% fluid_img "/img/Tree-Based%20Methods_files/Tree-Based%20Methods_10_0.png" %}}

Q 8. In the lab, a classification tree was applied to the Carseats data set after converting Sales into a qualitative response variable. Now we will seek to predict Sales using regression trees and related approaches, treating the response as a quantitative variable.

(a) Split the data set into a training set and a test set.


```python
carsets = pd.read_csv("data/Carsets.csv")
carsets['US'] = carsets['US'].map({'Yes': 1, 'No': 0})
carsets['Urban'] = carsets['Urban'].map({'Yes': 1, 'No': 0})
carsets = pd.get_dummies(carsets, prefix=['ShelveLoc'])
carsets = carsets.rename(columns={'Unnamed: 0': 'Id'})

X_train, X_test, y_train, y_test = train_test_split(carsets.drop(['Id', 'Sales'], axis=1),
                                                    carsets[['Sales']], test_size=0.1)
```

(b) Fit a regression tree to the training set. Plot the tree, and interpret the results. What test error rate do you obtain?

<b>Sol:</b> The stopping criteria used is: Split until there are more than 20 observations at a node. Reported test MSE is <b>2.6547</b>. The regression tree is displayed below.

{{% fluid_img "/img/Tree-Based%20Methods_files/Q8b.png" %}}

```python
from sklearn.tree import DecisionTreeRegressor, export_graphviz

regressor = DecisionTreeRegressor(random_state=0, min_samples_split=20)
regressor.fit(X_train, y_train)
```




    DecisionTreeRegressor(criterion='mse', max_depth=None, max_features=None,
               max_leaf_nodes=None, min_impurity_decrease=0.0,
               min_impurity_split=None, min_samples_leaf=1,
               min_samples_split=20, min_weight_fraction_leaf=0.0,
               presort=False, random_state=0, splitter='best')




```python
def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        print("Could not run dot, ie graphviz, to "
             "produce visualization")

visualize_tree(regressor, X_train.columns.tolist())

p = regressor.predict(X_test)
print("Test MSE is: " + str(mean_squared_error(p, y_test)))
```

    Could not run dot, ie graphviz, to produce visualization
    Test MSE is: 2.6547338209551623


(c) Use cross-validation in order to determine the optimal level of tree complexity. Does pruning the tree improve the test error rate?

<b>Sol:</b> The optimal depth of tree is <b>8</b>. The test MSE obtained for the mode is <b>3.40359</b>.


```python
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer

parameters = {'max_depth':range(1,30)}
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
clf = GridSearchCV(tree.DecisionTreeRegressor(random_state=1), parameters, n_jobs=4, cv=10,
                   scoring=mse_scorer)
clf.fit(X=X_train, y=y_train)
tree_model = clf.best_estimator_
print (clf.best_score_, clf.best_params_)
```

    -4.917597820641044 {'max_depth': 8}



```python
test_MSE = {}
test_MSE_std = {}
train_MSE = {}
train_MSE_std = {}
for idx, pm in enumerate(clf.cv_results_['param_max_depth'].data):
    test_MSE[pm] = abs(clf.cv_results_['mean_test_score'][idx]) # Taking absolute value as returned value of MSE is negative
    test_MSE_std[pm] = clf.cv_results_['std_test_score'][idx]
    train_MSE[pm] = abs(clf.cv_results_['mean_train_score'][idx])
    train_MSE_std[pm] = clf.cv_results_['std_train_score'][idx]

fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(121)

lists = sorted(test_MSE.items())
x, y = zip(*lists)

lists = sorted(test_MSE_std.items())
x1, y1 = zip(*lists)

plt.plot(x, y, color='g', label='Test Score')
plt.fill_between(x, np.subtract(y, y1), np.add(y, y1), alpha=0.06, color="g")
ax.set_xlabel('Depth of the Tree')
ax.set_ylabel('Test MSE')
ax.set_title('Test MSE vs Depth of the Tree')
ax.set_xlim([1, 29])
ax.grid()

ax = fig.add_subplot(122)
lists = sorted(train_MSE.items())
x, y = zip(*lists)

lists = sorted(train_MSE_std.items())
x1, y1 = zip(*lists)

plt.plot(x, y, color='r', label='Test Score')
plt.fill_between(x, np.subtract(y, y1), np.add(y, y1), alpha=0.06, color="r")
ax.set_xlabel('Depth of the Tree')
ax.set_ylabel('Train MSE')
ax.set_title('Train MSE vs Depth of the Tree')
ax.set_xlim([1, 29])
ax.grid()

plt.legend()
plt.show()
```


{{% fluid_img "/img/Tree-Based%20Methods_files/Tree-Based%20Methods_18_1.png" %}}


```python
p = tree_model.predict(X_test)
print("Test MSE is: " + str(mean_squared_error(p, y_test)))
```

    Test MSE is: 3.4035934469592464


(d) Use the bagging approach in order to analyze this data. What test error rate do you obtain? Use the importance() function to determine which variables are most important.

<b>Sol:</b> The obtained test MSE is <b>2.14596</b>. The pie-chart for importance of different features is shown as well.


```python
from sklearn.ensemble import BaggingRegressor

bagging = BaggingRegressor(random_state=0)
bagging.fit(X=X_train, y=y_train.values.ravel())
```




    BaggingRegressor(base_estimator=None, bootstrap=True,
             bootstrap_features=False, max_features=1.0, max_samples=1.0,
             n_estimators=10, n_jobs=1, oob_score=False, random_state=0,
             verbose=0, warm_start=False)




```python
p = bagging.predict(X_test)
print("Test MSE is: " + str(mean_squared_error(p, y_test)))
```

    Test MSE is: 2.145963949999999



```python
feature_importances = np.mean([tree.feature_importances_ for tree in bagging.estimators_], axis=0)
explode = (0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0)

fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(121)

plt.pie(feature_importances, explode=explode, labels=X_train.columns.tolist(), startangle=90)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')
plt.tight_layout()
plt.title("Importance of Features")
plt.show()
```

{{% fluid_img "/img/Tree-Based%20Methods_files/Tree-Based%20Methods_23_0.png" %}}


(e) Use random forests to analyze this data. What test error rate do you obtain? Use the importance() function to determine which variables are most important. Describe the effect of m, the number of variables considered at each split, on the error rate obtained.

<b>Sol:</b> The test MSE for random forest regressor is <b>2.243</b>. Minimum test MSE is obtained when all the features are used for split.


```python
from sklearn.ensemble import RandomForestRegressor

random_forest = RandomForestRegressor(random_state=0)
random_forest.fit(X=X_train, y=y_train.values.ravel())
p = random_forest.predict(X_test)
print("Test MSE is: " + str(mean_squared_error(p, y_test)))
```

    Test MSE is: 2.243107725



```python
feature_importances = np.mean([tree.feature_importances_ for tree in random_forest.estimators_], axis=0)
explode = (0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0)

fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(121)

plt.pie(feature_importances, explode=explode, labels=X_train.columns.tolist(), startangle=90)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')
plt.tight_layout()
plt.title("Importance of Features")
plt.show()
```

{{% fluid_img "/img/Tree-Based%20Methods_files/Tree-Based%20Methods_26_0.png" %}}


```python
def random_forest_m(X_train, Y_train, X_test, Y_test, features):
    test_MSE = {}
    for m in features:
        regr = RandomForestRegressor(random_state=0, max_features=m)
        regr.fit(X_train, Y_train)
        p = regr.predict(X_test)
        test_MSE[m] = mean_squared_error(p, Y_test)
    return test_MSE

test_MSE = random_forest_m(X_train, y_train.values.ravel(), X_test, y_test.values.ravel(),
                             np.arange(1,len(X_train.columns.tolist()),1))

fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(111)

lists = sorted(test_MSE.items())
x, y = zip(*lists)
plt.plot(x, y, color='g')

ax.set_xlabel('Number of features used for split')
ax.set_ylabel('Test MSE')
ax.set_title('Test MSE vs Number of features used for split')

plt.grid(b=True)
plt.show()
```

{{% fluid_img "/img/Tree-Based%20Methods_files/Tree-Based%20Methods_27_0.png" %}}


Q9. This problem involves the OJ data set which is part of the ISLR package.

(a) Create a training set containing a random sample of 800 observations, and a test set containing the remaining observations.


```python
oj = pd.read_csv("data/OJ.csv")
oj = oj.drop(['Unnamed: 0'], axis=1)
oj['Store7'] = oj['Store7'].map({'Yes': 1, 'No': 0})
oj.head()
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
      <th>Purchase</th>
      <th>WeekofPurchase</th>
      <th>StoreID</th>
      <th>PriceCH</th>
      <th>PriceMM</th>
      <th>DiscCH</th>
      <th>DiscMM</th>
      <th>SpecialCH</th>
      <th>SpecialMM</th>
      <th>LoyalCH</th>
      <th>SalePriceMM</th>
      <th>SalePriceCH</th>
      <th>PriceDiff</th>
      <th>Store7</th>
      <th>PctDiscMM</th>
      <th>PctDiscCH</th>
      <th>ListPriceDiff</th>
      <th>STORE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CH</td>
      <td>237</td>
      <td>1</td>
      <td>1.75</td>
      <td>1.99</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.500000</td>
      <td>1.99</td>
      <td>1.75</td>
      <td>0.24</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.24</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CH</td>
      <td>239</td>
      <td>1</td>
      <td>1.75</td>
      <td>1.99</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>0</td>
      <td>1</td>
      <td>0.600000</td>
      <td>1.69</td>
      <td>1.75</td>
      <td>-0.06</td>
      <td>0</td>
      <td>0.150754</td>
      <td>0.000000</td>
      <td>0.24</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CH</td>
      <td>245</td>
      <td>1</td>
      <td>1.86</td>
      <td>2.09</td>
      <td>0.17</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.680000</td>
      <td>2.09</td>
      <td>1.69</td>
      <td>0.40</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.091398</td>
      <td>0.23</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>MM</td>
      <td>227</td>
      <td>1</td>
      <td>1.69</td>
      <td>1.69</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.400000</td>
      <td>1.69</td>
      <td>1.69</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CH</td>
      <td>228</td>
      <td>7</td>
      <td>1.69</td>
      <td>1.69</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.956535</td>
      <td>1.69</td>
      <td>1.69</td>
      <td>0.00</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_train, X_test, y_train, y_test = train_test_split(oj.drop(['Purchase'], axis=1),
                                                    oj[['Purchase']], train_size=800)
```

(b) Fit a tree to the training data, with Purchase as the response and the other variables except for Buy as predictors. Use the summary() function to produce summary statistics about the tree, and describe the results obtained. What is the training error rate? How many terminal nodes does the tree have?

<b>Sol:</b> With the stopping criteria of <b>min_samples_split=100</b>, training error rate is <b>0.1525</b>. The most inportant features are <b>LoyalCH</b> and <b>PriceDiff</b>.


```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

classifier = DecisionTreeClassifier(random_state=0, min_samples_split=100)
classifier.fit(X_train, y_train)
print("Training Error rate is: " + str(1 - accuracy_score(classifier.predict(X_train), y_train)))
```

    Training Error rate is: 0.15249999999999997



```python
visualize_tree(classifier, X_train.columns.tolist())
```

    Could not run dot, ie graphviz, to produce visualization


(c) Type in the name of the tree object in order to get a detailed text output. Pick one of the terminal nodes, and interpret the information displayed.

(d) Create a plot of the tree, and interpret the results.

<b>Sol:</b> The plot of the tree is as follows.

{{% fluid_img "/img/Tree-Based%20Methods_files/Q8.9c.png" %}}

(e) Predict the response on the test data, and produce a confusion matrix comparing the test labels to the predicted test labels. What is the test error rate?

<b>Sol:</b> The test error rate is <b>0.21</b>.


```python
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

p = classifier.predict(X_test)
print(confusion_matrix(p, y_test))
print(classification_report(y_test, p,))
print("Test Error rate is: " + str(1 - accuracy_score(p, y_test)))
```

    [[139  33]
     [ 24  74]]
                 precision    recall  f1-score   support

             CH       0.81      0.85      0.83       163
             MM       0.76      0.69      0.72       107

    avg / total       0.79      0.79      0.79       270

    Test Error rate is: 0.21111111111111114


(f) Apply the cv.tree() function to the training set in order to determine the optimal tree size.

<b>Sol:</b> The optimal tree depth is <b>3</b>.


```python
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer

classification_error_rate_scorer = make_scorer(accuracy_score)
parameters = {'max_depth':range(1,30)}
clf = GridSearchCV(tree.DecisionTreeClassifier(random_state=0), parameters, n_jobs=4, cv=10,
                  scoring=classification_error_rate_scorer)
clf.fit(X=X_train, y=y_train)
tree_model = clf.best_estimator_
print (clf.best_score_, clf.best_params_)
```

    0.81625 {'max_depth': 3}


(g) Produce a plot with tree size on the x-axis and cross-validated classification error rate on the y-axis.

(h) Which tree size corresponds to the lowest cross-validated classification error rate?

<b>Sol:</b> Tree of depth 3 corresponds to the lowest cross-validated classification error rate.


```python
test_MSE = {}
test_MSE_std = {}
train_MSE = {}
train_MSE_std = {}
for idx, pm in enumerate(clf.cv_results_['param_max_depth'].data):
    test_MSE[pm] = 1 - clf.cv_results_['mean_test_score'][idx]
    test_MSE_std[pm] = clf.cv_results_['std_test_score'][idx]
    train_MSE[pm] = 1 - clf.cv_results_['mean_train_score'][idx]
    train_MSE_std[pm] = clf.cv_results_['std_train_score'][idx]

fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(121)

lists = sorted(test_MSE.items())
x, y = zip(*lists)

lists = sorted(test_MSE_std.items())
x1, y1 = zip(*lists)

plt.plot(x, y, color='g', label='CV Score')
plt.fill_between(x, np.subtract(y, y1), np.add(y, y1), alpha=0.06, color="g")
ax.set_xlabel('Depth of the Tree')
ax.set_ylabel('CV Classification Error Rate')
ax.set_title('CV Classification Error Rate vs Depth of the Tree')
ax.set_xlim([1, 29])
ax.grid()
ax.legend()

ax = fig.add_subplot(122)
lists = sorted(train_MSE.items())
x, y = zip(*lists)

lists = sorted(train_MSE_std.items())
x1, y1 = zip(*lists)

plt.plot(x, y, color='r', label='Train Score')
plt.fill_between(x, np.subtract(y, y1), np.add(y, y1), alpha=0.06, color="r")
ax.set_xlabel('Depth of the Tree')
ax.set_ylabel('CV Classification Error Rate')
ax.set_title('CV Classification Error Rate vs Depth of the Tree')
ax.set_xlim([1, 29])
ax.grid()
ax.legend()

plt.show()
```


{{% fluid_img "/img/Tree-Based%20Methods_files/Tree-Based%20Methods_40_1.png" %}}


(i) Produce a pruned tree corresponding to the optimal tree size obtained using cross-validation. If cross-validation does not lead to selection of a pruned tree, then create a pruned tree with five terminal nodes.

<b>Sol:</b> The purned tree corresponding to optimal tree size is displayed below.

{{% fluid_img "/img/Tree-Based%20Methods_files/Q8.9i.png" %}}


```python
visualize_tree(tree_model, X_train.columns.tolist())
```

    Could not run dot, ie graphviz, to produce visualization


(j) Compare the training error rates between the pruned and unpruned trees. Which is higher?

<b>Sol:</b> Training error rates for unpruned and pruned trees are <b>0.00875</b> and <b>0.15625</b> respectively. It his higher for the unpruned tree.


```python
c = DecisionTreeClassifier(random_state=1)
c.fit(X_train, y_train)
print("Training Error rate for unpurned tree is: " + str(1 - accuracy_score(c.predict(X_train), y_train)))
print("Training Error rate for pruned tree is: " + str(1 - accuracy_score(tree_model.predict(X_train), y_train)))
```

    Training Error rate for unpurned tree is: 0.008750000000000036
    Training Error rate for pruned tree is: 0.15625


(k) Compare the test error rates between the pruned and unpruned trees. Which is higher?

<b>Sol:</b> Test error rates for unpruned and pruned trees are <b>0.2518</b> and <b>0.1926</b> respectively. It his higher for the pruned tree.


```python
print("Test Error rate for unpurned tree is: " + str(1 - accuracy_score(c.predict(X_test), y_test)))
print("Test Error rate for pruned tree is: " + str(1 - accuracy_score(tree_model.predict(X_test), y_test)))
```

    Test Error rate for unpurned tree is: 0.2518518518518519
    Test Error rate for pruned tree is: 0.19259259259259254


Q10. We now use boosting to predict Salary in the Hitters data set.

(a) Remove the observations for whom the salary information is unknown, and then log-transform the salaries.


```python
hitters = pd.read_csv("data/Hitters.csv")
hitters = hitters.rename(columns={'Unnamed: 0': 'Name'})
hitters = hitters.dropna()
hitters["Salary"] = hitters["Salary"].apply(np.log)
hitters['League'] = hitters['League'].map({'N': 1, 'A': 0})
hitters['NewLeague'] = hitters['NewLeague'].map({'N': 1, 'A': 0})
hitters['Division'] = hitters['Division'].map({'W': 1, 'E': 0})
hitters.head()
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
      <th>Name</th>
      <th>AtBat</th>
      <th>Hits</th>
      <th>HmRun</th>
      <th>Runs</th>
      <th>RBI</th>
      <th>Walks</th>
      <th>Years</th>
      <th>CAtBat</th>
      <th>CHits</th>
      <th>...</th>
      <th>CRuns</th>
      <th>CRBI</th>
      <th>CWalks</th>
      <th>League</th>
      <th>Division</th>
      <th>PutOuts</th>
      <th>Assists</th>
      <th>Errors</th>
      <th>Salary</th>
      <th>NewLeague</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>-Alan Ashby</td>
      <td>315</td>
      <td>81</td>
      <td>7</td>
      <td>24</td>
      <td>38</td>
      <td>39</td>
      <td>14</td>
      <td>3449</td>
      <td>835</td>
      <td>...</td>
      <td>321</td>
      <td>414</td>
      <td>375</td>
      <td>1</td>
      <td>1</td>
      <td>632</td>
      <td>43</td>
      <td>10</td>
      <td>6.163315</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-Alvin Davis</td>
      <td>479</td>
      <td>130</td>
      <td>18</td>
      <td>66</td>
      <td>72</td>
      <td>76</td>
      <td>3</td>
      <td>1624</td>
      <td>457</td>
      <td>...</td>
      <td>224</td>
      <td>266</td>
      <td>263</td>
      <td>0</td>
      <td>1</td>
      <td>880</td>
      <td>82</td>
      <td>14</td>
      <td>6.173786</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-Andre Dawson</td>
      <td>496</td>
      <td>141</td>
      <td>20</td>
      <td>65</td>
      <td>78</td>
      <td>37</td>
      <td>11</td>
      <td>5628</td>
      <td>1575</td>
      <td>...</td>
      <td>828</td>
      <td>838</td>
      <td>354</td>
      <td>1</td>
      <td>0</td>
      <td>200</td>
      <td>11</td>
      <td>3</td>
      <td>6.214608</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-Andres Galarraga</td>
      <td>321</td>
      <td>87</td>
      <td>10</td>
      <td>39</td>
      <td>42</td>
      <td>30</td>
      <td>2</td>
      <td>396</td>
      <td>101</td>
      <td>...</td>
      <td>48</td>
      <td>46</td>
      <td>33</td>
      <td>1</td>
      <td>0</td>
      <td>805</td>
      <td>40</td>
      <td>4</td>
      <td>4.516339</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-Alfredo Griffin</td>
      <td>594</td>
      <td>169</td>
      <td>4</td>
      <td>74</td>
      <td>51</td>
      <td>35</td>
      <td>11</td>
      <td>4408</td>
      <td>1133</td>
      <td>...</td>
      <td>501</td>
      <td>336</td>
      <td>194</td>
      <td>0</td>
      <td>1</td>
      <td>282</td>
      <td>421</td>
      <td>25</td>
      <td>6.620073</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



(b) Create a training set consisting of the first 200 observations, and a test set consisting of the remaining observations.


```python
train = hitters.iloc[0:200, :]
test = hitters.iloc[200:, :]
```

(c) Perform boosting on the training set with 1,000 trees for a range of values of the shrinkage parameter λ. Produce a plot with different shrinkage values on the x-axis and the corresponding training set MSE on the y-axis.

(d) Produce a plot with different shrinkage values on the x-axis and the corresponding test set MSE on the y-axis.


```python
from sklearn.ensemble import GradientBoostingRegressor

def boosting_shrinkage(X_train, Y_train, X_test, Y_test, shrinkages):
    train_MSE = {}
    test_MSE = {}
    for s in shrinkages:
        clf = GradientBoostingRegressor(random_state=0, n_estimators=1000, learning_rate=s)
        clf.fit(X_train, Y_train)
        p = clf.predict(X_train)
        train_MSE[s] = mean_squared_error(p, Y_train)
        p = clf.predict(X_test)
        test_MSE[s] = mean_squared_error(p, Y_test)
    return (train_MSE, test_MSE)

X_train = train.drop(['Name', 'Salary'], axis=1)
y_train = train[['Salary']]
X_test = test.drop(['Name', 'Salary'], axis=1)
y_test = test[['Salary']]

res = boosting_shrinkage(X_train, y_train.values.ravel(), X_test, y_test.values.ravel(),
                               np.linspace(0.001, 0.5, 100))

fig = plt.figure(figsize=(15,8))

ax = fig.add_subplot(121)
lists = sorted(res[0].items())
x, y = zip(*lists)
plt.plot(x, y, color='r', label='Training Error')

ax.set_xlabel('Shrinkage/ Learning Rate')
ax.set_ylabel('Train MSE')
ax.set_title('Train MSE vs Shrinkage/ Learning Rate')
ax.grid()

ax = fig.add_subplot(122)
lists = sorted(res[1].items())
x, y = zip(*lists)
plt.plot(x, y, color='g', label='Test Error')

ax.set_xlabel('Shrinkage/ Learning Rate')
ax.set_ylabel('Test MSE')
ax.set_title('Test MSE vs Shrinkage/ Learning Rate')
ax.grid()

plt.grid(b=True)
plt.show()
```

{{% fluid_img "/img/Tree-Based%20Methods_files/Tree-Based%20Methods_52_0.png" %}}


(f) Which variables appear to be the most important predictors in the boosted model?

<b>Sol:</b> The best boosted model is the one with <b>learning rate</b> of 0.1586 and uses 15 trees. The bar graph showing the importance of various features in the model is shown below.


```python
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV

parameters = {'learning_rate': np.linspace(0.001, 0.5, 20), 'n_estimators': np.arange(1, 40, 2)}
clf = GridSearchCV(ensemble.GradientBoostingRegressor(random_state=0), parameters, n_jobs=4, cv=10)
clf.fit(X=X_train, y=y_train.values.ravel())
tree_model = clf.best_estimator_
print (clf.best_score_, clf.best_params_)
p = tree_model.predict(X_test)
print("Test MSE is: " + str(mean_squared_error(p, y_test)))
```

    0.7156560970841344 {'learning_rate': 0.15857894736842104, 'n_estimators': 15}
    Test MSE is: 0.21640274598049566



```python
feature_importances = tree_model.feature_importances_

fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(111)

plt.bar(X_train.columns.tolist(), feature_importances, alpha=0.3)

ax.set_xlabel('Feature')
ax.set_ylabel('Importance')
plt.tight_layout()
plt.title("Importance of Features")
plt.grid()
plt.show()
```

{{% fluid_img "/img/Tree-Based%20Methods_files/Tree-Based%20Methods_55_0.png" %}}


(g) Now apply bagging to the training set. What is the test set MSE for this approach?

<b>Sol:</b> Test MSE for bagging is <b>0.2565</b>.


```python
from sklearn.ensemble import BaggingRegressor

bagging = BaggingRegressor(random_state=0)
bagging.fit(X=X_train, y=y_train.values.ravel())
p = bagging.predict(X_test)
print("Test MSE is: " + str(mean_squared_error(p, y_test)))
```

    Test MSE is: 0.25657647011895646


Q11. This question uses the Caravan data set.

(a) Create a training set consisting of the first 1,000 observations, and a test set consisting of the remaining observations.


```python
caravan = pd.read_csv("data/Caravan.csv")
caravan = caravan.drop(['Unnamed: 0'], axis=1)
train = caravan.iloc[0:1000, :]
test = caravan.iloc[1000:, :]
print(train.shape)
print(test.shape)
```

    (1000, 86)
    (4822, 86)


(b) Fit a boosting model to the training set with Purchase as the response and the other variables as predictors. Use 1,000 trees, and a shrinkage value of 0.01. Which predictors appear to be the most important?


```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, make_scorer

clf = GradientBoostingClassifier(learning_rate=0.01, n_estimators=1000)

X_train = train.drop(['Purchase'], axis=1)
y_train = train[['Purchase']]
X_test = test.drop(['Purchase'], axis=1)
y_test = test[['Purchase']]

clf.fit(X=X_train, y=y_train.values.ravel())
p = clf.predict(X_test)
print("Test Error Rate is: " + str(1-accuracy_score(p, y_test)))
```

    Test Error Rate is: 0.06656988801327246



```python
feature_importances = clf.feature_importances_

columns, importance = zip(*((columns, importance) for columns, importance in
                   zip(X_train.columns.tolist(), clf.feature_importances_)
                   if importance > 0.02))

fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(111)

plt.bar(list(columns), list(importance), alpha=0.3)

ax.set_xlabel('Feature')
ax.set_ylabel('Importance (> 0.02)')
plt.tight_layout()
plt.title("Importance of Features")
plt.grid()
plt.show()
```

{{% fluid_img "/img/Tree-Based%20Methods_files/Tree-Based%20Methods_62_0.png" %}}


(c) Use the boosting model to predict the response on the test data. Predict that a person will make a purchase if the estimated probability of purchase is greater than 20 %. Form a confusion matrix. What fraction of the people predicted to make a purchase do in fact make one? How does this compare with the results obtained from applying KNN or logistic regression to this data set?

<b>Sol:</b> The percentage of people who are predicted to make a purchase and make one is <b>24.19%</b>.


```python
y_pred = pd.Series(p)
y_true = (y_test['Purchase'])
y_true.reset_index(drop=True, inplace=True)

pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
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
      <th>Predicted</th>
      <th>No</th>
      <th>Yes</th>
      <th>All</th>
    </tr>
    <tr>
      <th>True</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No</th>
      <td>4486</td>
      <td>47</td>
      <td>4533</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>274</td>
      <td>15</td>
      <td>289</td>
    </tr>
    <tr>
      <th>All</th>
      <td>4760</td>
      <td>62</td>
      <td>4822</td>
    </tr>
  </tbody>
</table>
</div>
