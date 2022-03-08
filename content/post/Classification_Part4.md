
+++
date = "2018-05-16T10:20:15+01:00"
description = "ISLR Classification"
draft = false
tags = ["ISLR", "Classification", "Exercises", "Applied"]
title = "ISLR Chapter 4: Classification (Part 4: Exercises- Applied)"
topics = ["ISLR"]

+++

#### Applied

Q10. This question should be answered using the Weekly data set.

(a) Produce some numerical and graphical summaries of the Weekly data. Do there appear to be any patterns?


```python
import seaborn as sns
weekly = pd.read_csv("data/Weekly.csv")

sns.pairplot(weekly, vars=['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume'], hue='Direction')
```
{{% fluid_img "/img/Classification_files/Classification_49_2.png" %}}


(b) Use the full data set to perform a logistic regression with Direction as the response and the five lag variables plus Volume as predictors. Use the summary function to print the results. Do any of the predictors appear to be statistically significant? If so, which ones?

<b>Sol:</b> Significant predictors are: <b>Lag2</b>


```python
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit

weekly['trend'] = weekly['Direction'].map({'Down': 0, 'Up': 1})
X = weekly[['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume']]
X = sm.add_constant(X, prepend=True)
y = weekly['trend']

model = Logit(y, X)
result = model.fit()
print(result.summary())
```

    Optimization terminated successfully.
             Current function value: 0.682441
             Iterations 4
                               Logit Regression Results
    ==============================================================================
    Dep. Variable:                  trend   No. Observations:                 1089
    Model:                          Logit   Df Residuals:                     1082
    Method:                           MLE   Df Model:                            6
    Date:                Mon, 10 Sep 2018   Pseudo R-squ.:                0.006580
    Time:                        19:13:02   Log-Likelihood:                -743.18
    converged:                       True   LL-Null:                       -748.10
                                            LLR p-value:                    0.1313
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.2669      0.086      3.106      0.002       0.098       0.435
    Lag1          -0.0413      0.026     -1.563      0.118      -0.093       0.010
    Lag2           0.0584      0.027      2.175      0.030       0.006       0.111
    Lag3          -0.0161      0.027     -0.602      0.547      -0.068       0.036
    Lag4          -0.0278      0.026     -1.050      0.294      -0.080       0.024
    Lag5          -0.0145      0.026     -0.549      0.583      -0.066       0.037
    Volume        -0.0227      0.037     -0.616      0.538      -0.095       0.050
    ==============================================================================


(c) Compute the confusion matrix and overall fraction of correct predictions. Explain what the confusion matrix is telling you about the types of mistakes made by logistic regression.

<b>Sol:</b> The confusion matrix is shown below. The overall fraction of correct predictions is <b>56.11%</b>. The model has higher <b>flase positive</b> rate.


```python
print("\t\t Confusion Matrix")
print("\t Down  Up(Predicted)")
print("Down \t" + str(result.pred_table(threshold=0.5)[0]))
print("Up \t" + str(result.pred_table(threshold=0.5)[1]))
```

    		 Confusion Matrix
    	 Down  Up(Predicted)
    Down 	[ 23. 418.]
    Up 	[ 20. 524.]


(d) Now fit the logistic regression model using a training data period from 1990 to 2008, with Lag2 as the only predictor. Compute the confusion matrix and the overall fraction of correct predictions for the held out data (that is, the data from 2009 and 2010).

<b>Sol:</b> The confusion matrix is shown below. The overall fraction of correct predictions is <b>62.5%</b>.


```python
train = weekly.loc[weekly['Year'] <= 2008]
test = weekly.loc[weekly['Year'] >= 2009]

X = train[['Lag2']]
X = sm.add_constant(X, prepend=True)
y = train['trend']

model = Logit(y, X)
result = model.fit()
print(result.summary())
```

    Optimization terminated successfully.
             Current function value: 0.685555
             Iterations 4
                               Logit Regression Results
    ==============================================================================
    Dep. Variable:                  trend   No. Observations:                  985
    Model:                          Logit   Df Residuals:                      983
    Method:                           MLE   Df Model:                            1
    Date:                Mon, 10 Sep 2018   Pseudo R-squ.:                0.003076
    Time:                        19:53:30   Log-Likelihood:                -675.27
    converged:                       True   LL-Null:                       -677.35
                                            LLR p-value:                   0.04123
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.2033      0.064      3.162      0.002       0.077       0.329
    Lag2           0.0581      0.029      2.024      0.043       0.002       0.114
    ==============================================================================



```python
from sklearn.metrics import confusion_matrix

X_test = test[['Lag2']]
X_test = sm.add_constant(X_test, prepend=True)
y_test = test['trend']
predictions = result.predict(X_test) > 0.5

print("\t\t Confusion Matrix")
print("\t Down Up(Predicted)")
print("Down \t" + str(confusion_matrix(y_test, predictions)[0]))
print("Up \t" + str(confusion_matrix(y_test, predictions)[1]))
```

    		 Confusion Matrix
    	 Down Up(Predicted)
    Down 	[ 9 34]
    Up 	[ 5 56]


(e) Repeat (d) using LDA.

<b>Sol:</b> The confusion matrix is shown below. The overall fraction of correct predictions is <b>62.5%</b>.


```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix

clf = LinearDiscriminantAnalysis()
clf.fit(train[['Lag2']], train['trend'])
y_predict = clf.predict(test[['Lag2']])

print("\t\t Confusion Matrix")
print("\t Down Up(Predicted)")
print("Down \t" + str(confusion_matrix(y_true=test['trend'], y_pred=y_predict)[0]))
print("Up \t" + str(confusion_matrix(y_true=test['trend'], y_pred=y_predict)[1]))
```

    		 Confusion Matrix
    	 Down Up(Predicted)
    Down 	[ 9 34]
    Up 	[ 5 56]


(f) Repeat (d) using QDA.

<b>Sol:</b> The confusion matrix is shown below. The overall fraction of correct predictions is <b>58.65%</b>. The model always predict that the market will go up.


```python
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

clf = QuadraticDiscriminantAnalysis()
clf.fit(train[['Lag2']], train['trend'])
y_predict = clf.predict(test[['Lag2']])

print("\t\t Confusion Matrix")
print("\t Down Up(Predicted)")
print("Down \t" + str(confusion_matrix(y_true=test['trend'], y_pred=y_predict)[0]))
print("Up \t" + str(confusion_matrix(y_true=test['trend'], y_pred=y_predict)[1]))
```

    		 Confusion Matrix
    	 Down Up(Predicted)
    Down 	[ 0 43]
    Up 	[ 0 61]


(g) Repeat (d) using KNN with K = 1.

<b>Sol:</b> The confusion matrix is shown below. The overall fraction of correct predictions is <b>49.04%</b>.


```python
from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(train[['Lag2']], train['trend'])
y_predict = neigh.predict(test[['Lag2']])

print("\t\t Confusion Matrix")
print("\t Down Up(Predicted)")
print("Down \t" + str(confusion_matrix(y_true=test['trend'], y_pred=y_predict)[0]))
print("Up \t" + str(confusion_matrix(y_true=test['trend'], y_pred=y_predict)[1]))
```

    		 Confusion Matrix
    	 Down Up(Predicted)
    Down 	[21 22]
    Up 	[31 30]


(h) Which of these methods appears to provide the best results on this data?

<b>Sol:</b> The logistic regression and LDA have the minimum error rate.

Q11. In this problem, you will develop a model to predict whether a given car gets high or low gas mileage based on the Auto data set.


```python
auto = pd.read_csv("data/Auto.csv")
auto.dropna(inplace=True)
auto = auto[auto['horsepower'] != '?']
auto['horsepower'] = auto['horsepower'].astype(int)
auto.head()
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
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>year</th>
      <th>origin</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>8</td>
      <td>307.0</td>
      <td>130</td>
      <td>3504</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>chevrolet chevelle malibu</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>8</td>
      <td>350.0</td>
      <td>165</td>
      <td>3693</td>
      <td>11.5</td>
      <td>70</td>
      <td>1</td>
      <td>buick skylark 320</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>8</td>
      <td>318.0</td>
      <td>150</td>
      <td>3436</td>
      <td>11.0</td>
      <td>70</td>
      <td>1</td>
      <td>plymouth satellite</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.0</td>
      <td>8</td>
      <td>304.0</td>
      <td>150</td>
      <td>3433</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>amc rebel sst</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.0</td>
      <td>8</td>
      <td>302.0</td>
      <td>140</td>
      <td>3449</td>
      <td>10.5</td>
      <td>70</td>
      <td>1</td>
      <td>ford torino</td>
    </tr>
  </tbody>
</table>
</div>



(a) Create a binary variable, mpg01, that contains a 1 if mpg contains a value above its median, and a 0 if mpg contains a value below its median.


```python
auto['mpg01'] = np.where(auto['mpg']>=auto['mpg'].median(), 1, 0)
auto.head()
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
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>year</th>
      <th>origin</th>
      <th>name</th>
      <th>mpg01</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>8</td>
      <td>307.0</td>
      <td>130</td>
      <td>3504</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>chevrolet chevelle malibu</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>8</td>
      <td>350.0</td>
      <td>165</td>
      <td>3693</td>
      <td>11.5</td>
      <td>70</td>
      <td>1</td>
      <td>buick skylark 320</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>8</td>
      <td>318.0</td>
      <td>150</td>
      <td>3436</td>
      <td>11.0</td>
      <td>70</td>
      <td>1</td>
      <td>plymouth satellite</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.0</td>
      <td>8</td>
      <td>304.0</td>
      <td>150</td>
      <td>3433</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>amc rebel sst</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.0</td>
      <td>8</td>
      <td>302.0</td>
      <td>140</td>
      <td>3449</td>
      <td>10.5</td>
      <td>70</td>
      <td>1</td>
      <td>ford torino</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



(b) Explore the data graphically in order to investigate the association between mpg01 and the other features. Which of the other features seem most likely to be useful in predicting mpg01? Scatterplots and boxplots may be useful tools to answer this question. Describe your findings.

<b>Sol:</b> The scatterplot of the data is shown below. As mpg01 with value 1 is shown with orange and value 0 is shown with blue, it is evident that certain combinations of predictors are present which can be used to model a classifier with high accuracy. For example, if we take a look at the scatter plot of weight and accelaration, it can be noted that the observations are decently segregated based on class.


```python
sns.pairplot(auto, vars=['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration',
                         'year', 'origin'], hue='mpg01')
```

{{% fluid_img "/img/Classification_files/Classification_69_2.png" %}}


(c) Split the data into a training set and a test set.


```python
msk = np.random.rand(len(auto)) < 0.8
train = auto[msk]
test = auto[~msk]
print("Length of training data: " +str(len(train)))
print("Length of test data: " +str(len(test)))
```

    Length of training data: 311
    Length of test data: 81


(d) Perform LDA on the training data in order to predict mpg01 using the variables that seemed most associated with mpg01 in (b). What is the test error of the model obtained?

<b>Sol:</b> The test prediction accuracy for the model is <b>96.30%</b>.


```python
clf = LinearDiscriminantAnalysis()
clf.fit(train[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year']], train['mpg01'])
y_predict = clf.predict(test[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year']])

print("\t\t Confusion Matrix")
print("\t Down Up(Predicted)")
print("Down \t" + str(confusion_matrix(y_true=test['mpg01'], y_pred=y_predict)[0]))
print("Up \t" + str(confusion_matrix(y_true=test['mpg01'], y_pred=y_predict)[1]))
```

    		 Confusion Matrix
    	 Down Up(Predicted)
    Down 	[32  3]
    Up 	[ 0 46]


(e) Perform QDA on the training data in order to predict mpg01 using the variables that seemed most associated with mpg01 in (b). What is the test error of the model obtained?

<b>Sol:</b> The test prediction accuracy for the model is <b>97.53%</b>.


```python
clf = QuadraticDiscriminantAnalysis()
clf.fit(train[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year']], train['mpg01'])
y_predict = clf.predict(test[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year']])

print("\t\t Confusion Matrix")
print("\t Down Up(Predicted)")
print("Down \t" + str(confusion_matrix(y_true=test['mpg01'], y_pred=y_predict)[0]))
print("Up \t" + str(confusion_matrix(y_true=test['mpg01'], y_pred=y_predict)[1]))
```

    		 Confusion Matrix
    	 Down Up(Predicted)
    Down 	[34  1]
    Up 	[ 1 45]


(f) Perform logistic regression on the training data in order to predict mpg01 using the variables that seemed most associated with mpg01 in (b). What is the test error of the model obtained?

<b>Sol:</b> The test prediction accuracy for the model is <b>96.30%</b>.


```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(train[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year']], train['mpg01'])
y_predict = clf.predict(test[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year']])

print("\t\t Confusion Matrix")
print("\t Down Up(Predicted)")
print("Down \t" + str(confusion_matrix(y_true=test['mpg01'], y_pred=y_predict)[0]))
print("Up \t" + str(confusion_matrix(y_true=test['mpg01'], y_pred=y_predict)[1]))
```

    		 Confusion Matrix
    	 Down Up(Predicted)
    Down 	[34  1]
    Up 	[ 2 44]


(g) Perform KNN on the training data, with several values of K, in order to predict mpg01. Use only the variables that seemed most associated with mpg01 in (b). What test errors do you obtain? Which value of K seems to perform the best on this data set?

<b>Sol:</b> The test prediction accuracy for the model is <b>96.30%</b>. The optimal value of K is 20. On further increasing the value, no improvement is achieved.


```python
K_values = [1, 2, 4, 8, 15, 20, 30, 50, 100]
for k in K_values:
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(train[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year']], train['mpg01'])
    y_predict = neigh.predict(test[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year']])

    print("\t\t Confusion Matrix for K = " +str(k))
    print("\t Down Up(Predicted)")
    print("Down \t" + str(confusion_matrix(y_true=test['mpg01'], y_pred=y_predict)[0]))
    print("Up \t" + str(confusion_matrix(y_true=test['mpg01'], y_pred=y_predict)[1]))
```

    		 Confusion Matrix for K = 1
    	 Down Up(Predicted)
    Down 	[33  2]
    Up 	[ 5 41]
    		 Confusion Matrix for K = 2
    	 Down Up(Predicted)
    Down 	[34  1]
    Up 	[ 6 40]
    		 Confusion Matrix for K = 4
    	 Down Up(Predicted)
    Down 	[34  1]
    Up 	[ 6 40]
    		 Confusion Matrix for K = 8
    	 Down Up(Predicted)
    Down 	[34  1]
    Up 	[ 2 44]
    		 Confusion Matrix for K = 15
    	 Down Up(Predicted)
    Down 	[33  2]
    Up 	[ 2 44]
    		 Confusion Matrix for K = 20
    	 Down Up(Predicted)
    Down 	[33  2]
    Up 	[ 1 45]
    		 Confusion Matrix for K = 30
    	 Down Up(Predicted)
    Down 	[34  1]
    Up 	[ 2 44]
    		 Confusion Matrix for K = 50
    	 Down Up(Predicted)
    Down 	[34  1]
    Up 	[ 2 44]
    		 Confusion Matrix for K = 100
    	 Down Up(Predicted)
    Down 	[34  1]
    Up 	[ 2 44]


Q13. Using the Boston data set, fit classification models in order to predict whether a given suburb has a crime rate above or below the median. Explore logistic regression, LDA, and KNN models using various subsets of the predictors. Describe your findings.

<b>Sol:</b> From the scatterplot it is identified that the predictors that can be used to model the classifier are: 'zn', 'chas', 'nox', 'rm', 'age', 'dis', 'black', 'lstat', 'medv'.

The test prediction accuracy for <b>logistic regression is 81.48%</b>. The test prediction accuracy for <b>LDA is 81.48%</b>. The test prediction accuracy for <b>LDA is 77.78%</b>. The test prediction accuracy for <b>KNN (K=120) is 82.41%</b>.


```python
boston = pd.read_csv("data/Boston.csv")
boston['crime01'] = np.where(boston['crim']>=boston['crim'].median(), 1, 0)
boston.head()
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
      <th>crim</th>
      <th>zn</th>
      <th>indus</th>
      <th>chas</th>
      <th>nox</th>
      <th>rm</th>
      <th>age</th>
      <th>dis</th>
      <th>rad</th>
      <th>tax</th>
      <th>ptratio</th>
      <th>black</th>
      <th>lstat</th>
      <th>medv</th>
      <th>crime01</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1</td>
      <td>296</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
      <td>24.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2</td>
      <td>242</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
      <td>21.6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2</td>
      <td>242</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
      <td>34.7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3</td>
      <td>222</td>
      <td>18.7</td>
      <td>394.63</td>
      <td>2.94</td>
      <td>33.4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3</td>
      <td>222</td>
      <td>18.7</td>
      <td>396.90</td>
      <td>5.33</td>
      <td>36.2</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.pairplot(boston, vars=['zn', 'indus', 'chas', 'nox', 'rm',
                         'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat', 'medv'], hue='crime01')
```

{{% fluid_img "/img/Classification_files/Classification_82_2.png" %}}



```python
msk = np.random.rand(len(boston)) < 0.8
train = boston[msk]
test = boston[~msk]
print("Length of training data: " +str(len(train)))
print("Length of test data: " +str(len(test)))
```

    Length of training data: 398
    Length of test data: 108



```python
clf = LogisticRegression()
clf.fit(train[['zn', 'chas', 'nox', 'rm', 'age', 'dis', 'black', 'lstat', 'medv']], train['crime01'])
y_predict = clf.predict(test[['zn', 'chas', 'nox', 'rm', 'age', 'dis', 'black', 'lstat', 'medv']])

print("\t\t Confusion Matrix")
print("\t Down Up(Predicted)")
print("Down \t" + str(confusion_matrix(y_true=test['crime01'], y_pred=y_predict)[0]))
print("Up \t" + str(confusion_matrix(y_true=test['crime01'], y_pred=y_predict)[1]))
```

    		 Confusion Matrix
    	 Down Up(Predicted)
    Down 	[40  6]
    Up 	[14 48]



```python
clf = LinearDiscriminantAnalysis()
clf.fit(train[['zn', 'chas', 'nox', 'rm', 'age', 'dis', 'black', 'lstat', 'medv']], train['crime01'])
y_predict = clf.predict(test[['zn', 'chas', 'nox', 'rm', 'age', 'dis', 'black', 'lstat', 'medv']])

print("\t\t Confusion Matrix")
print("\t Down Up(Predicted)")
print("Down \t" + str(confusion_matrix(y_true=test['crime01'], y_pred=y_predict)[0]))
print("Up \t" + str(confusion_matrix(y_true=test['crime01'], y_pred=y_predict)[1]))
```

    		 Confusion Matrix
    	 Down Up(Predicted)
    Down 	[43  3]
    Up 	[17 45]



```python
clf = QuadraticDiscriminantAnalysis()
clf.fit(train[['zn', 'chas', 'nox', 'rm', 'age', 'dis', 'black', 'lstat', 'medv']], train['crime01'])
y_predict = clf.predict(test[['zn', 'chas', 'nox', 'rm', 'age', 'dis', 'black', 'lstat', 'medv']])

print("\t\t Confusion Matrix")
print("\t Down Up(Predicted)")
print("Down \t" + str(confusion_matrix(y_true=test['crime01'], y_pred=y_predict)[0]))
print("Up \t" + str(confusion_matrix(y_true=test['crime01'], y_pred=y_predict)[1]))
```

    		 Confusion Matrix
    	 Down Up(Predicted)
    Down 	[40  6]
    Up 	[18 44]



```python
K_values = [1, 2, 4, 8, 15, 20, 30, 50, 100, 120, 150]
for k in K_values:
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(train[['zn', 'chas', 'nox', 'rm', 'age', 'dis', 'black', 'lstat', 'medv']], train['crime01'])
    y_predict = neigh.predict(test[['zn', 'chas', 'nox', 'rm', 'age', 'dis', 'black', 'lstat', 'medv']])

    print("\t\t Confusion Matrix for K = " +str(k))
    print("\t Down Up(Predicted)")
    print("Down \t" + str(confusion_matrix(y_true=test['crime01'], y_pred=y_predict)[0]))
    print("Up \t" + str(confusion_matrix(y_true=test['crime01'], y_pred=y_predict)[1]))
```

    		 Confusion Matrix for K = 1
    	 Down Up(Predicted)
    Down 	[40  6]
    Up 	[17 45]
    		 Confusion Matrix for K = 2
    	 Down Up(Predicted)
    Down 	[42  4]
    Up 	[29 33]
    		 Confusion Matrix for K = 4
    	 Down Up(Predicted)
    Down 	[42  4]
    Up 	[22 40]
    		 Confusion Matrix for K = 8
    	 Down Up(Predicted)
    Down 	[42  4]
    Up 	[19 43]
    		 Confusion Matrix for K = 15
    	 Down Up(Predicted)
    Down 	[41  5]
    Up 	[19 43]
    		 Confusion Matrix for K = 20
    	 Down Up(Predicted)
    Down 	[42  4]
    Up 	[20 42]
    		 Confusion Matrix for K = 30
    	 Down Up(Predicted)
    Down 	[43  3]
    Up 	[17 45]
    		 Confusion Matrix for K = 50
    	 Down Up(Predicted)
    Down 	[41  5]
    Up 	[19 43]
    		 Confusion Matrix for K = 100
    	 Down Up(Predicted)
    Down 	[38  8]
    Up 	[12 50]
    		 Confusion Matrix for K = 120
    	 Down Up(Predicted)
    Down 	[38  8]
    Up 	[11 51]
    		 Confusion Matrix for K = 150
    	 Down Up(Predicted)
    Down 	[38  8]
    Up 	[12 50]
