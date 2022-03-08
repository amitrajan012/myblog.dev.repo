+++
date = "2018-06-08T03:17:36+01:00"
description = "ISLR Moving Beyond Linearity"
draft = false
tags = ["ISLR", "Resampling", "Moving Beyond Linearity", "Exercises", "Applied"]
title = "ISLR Chapter 7: Moving Beyond Linearity (Part 6: Exercises - Applied)"
topics = ["ISLR"]

+++

#### Applied

Q6. In this exercise, you will further analyze the Wage data set considered throughout this chapter.

(a) Perform polynomial regression to predict wage using age. Use cross-validation to select the optimal degree d for the polynomial. What degree was chosen, and how does this compare to the results of hypothesis testing using ANOVA? Make a plot of the resulting polynomial fit to the data.

<b>Sol:</b> The optimal degree of polynomial selected from cross-validation is <b>4</b>. From the ANOVA of models of degree 1 to 5, it is found that the models of degree 2 and 3 are significant. The test MSE shows the lowest value for degree 3 model as well.


```python
import pandas as pd
import numpy as np

wage = pd.read_csv("data/Wage.csv")
wage.head()
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
      <th>year</th>
      <th>age</th>
      <th>sex</th>
      <th>maritl</th>
      <th>race</th>
      <th>education</th>
      <th>region</th>
      <th>jobclass</th>
      <th>health</th>
      <th>health_ins</th>
      <th>logwage</th>
      <th>wage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>231655</td>
      <td>2006</td>
      <td>18</td>
      <td>1. Male</td>
      <td>1. Never Married</td>
      <td>1. White</td>
      <td>1. &lt; HS Grad</td>
      <td>2. Middle Atlantic</td>
      <td>1. Industrial</td>
      <td>1. &lt;=Good</td>
      <td>2. No</td>
      <td>4.318063</td>
      <td>75.043154</td>
    </tr>
    <tr>
      <th>1</th>
      <td>86582</td>
      <td>2004</td>
      <td>24</td>
      <td>1. Male</td>
      <td>1. Never Married</td>
      <td>1. White</td>
      <td>4. College Grad</td>
      <td>2. Middle Atlantic</td>
      <td>2. Information</td>
      <td>2. &gt;=Very Good</td>
      <td>2. No</td>
      <td>4.255273</td>
      <td>70.476020</td>
    </tr>
    <tr>
      <th>2</th>
      <td>161300</td>
      <td>2003</td>
      <td>45</td>
      <td>1. Male</td>
      <td>2. Married</td>
      <td>1. White</td>
      <td>3. Some College</td>
      <td>2. Middle Atlantic</td>
      <td>1. Industrial</td>
      <td>1. &lt;=Good</td>
      <td>1. Yes</td>
      <td>4.875061</td>
      <td>130.982177</td>
    </tr>
    <tr>
      <th>3</th>
      <td>155159</td>
      <td>2003</td>
      <td>43</td>
      <td>1. Male</td>
      <td>2. Married</td>
      <td>3. Asian</td>
      <td>4. College Grad</td>
      <td>2. Middle Atlantic</td>
      <td>2. Information</td>
      <td>2. &gt;=Very Good</td>
      <td>1. Yes</td>
      <td>5.041393</td>
      <td>154.685293</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11443</td>
      <td>2005</td>
      <td>50</td>
      <td>1. Male</td>
      <td>4. Divorced</td>
      <td>1. White</td>
      <td>2. HS Grad</td>
      <td>2. Middle Atlantic</td>
      <td>2. Information</td>
      <td>1. &lt;=Good</td>
      <td>1. Yes</td>
      <td>4.318063</td>
      <td>75.043154</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn import preprocessing
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

np.random.seed(5)

def polynomial_regression(X_train, Y_train, X_test, Y_test, M):
    validation_MSE = {}
    test_MSE = {}

    for m in M:
        poly = preprocessing.PolynomialFeatures(degree=m)
        X_ = poly.fit_transform(X_train)

        mse = 0
        loo = LeaveOneOut() # Leave one out cross-validation
        for train_index, test_index in loo.split(X_):
            X, X_CV = X_[train_index], X_[test_index]
            Y, Y_CV = Y_train[train_index], Y_train[test_index]
            # Linear Regression (including higher order predictors)
            model = LinearRegression(fit_intercept=True)
            model.fit(X, Y)
            p = model.predict(X_CV)
            mse += mean_squared_error(p, Y_CV)
        validation_MSE[m] = mse/len(X_)

        # Compute test MSE for the model
        model = LinearRegression(fit_intercept=True)
        model.fit(X_, Y_train)
        p = model.predict(poly.fit_transform(X_test))
        test_MSE[m] = mean_squared_error(p, Y_test)

    # Plot validation MSE
    lists = sorted(validation_MSE.items()) # sorted by key, return a list of tuples
    x, y = zip(*lists) # unpack a list of pairs into two tuples
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(121)
    plt.plot(x, y, color='r')
    plt.grid()
    ax.set_xlabel('Degree of Polynomial')
    ax.set_ylabel('Validation MSE')
    ax.set_title('Validation MSE vs Degree of Polynomial')

    lists = sorted(test_MSE.items()) # sorted by key, return a list of tuples
    x, y = zip(*lists) # unpack a list of pairs into two tuples
    ax = fig.add_subplot(122)
    plt.plot(x, y, color='g')
    plt.grid()
    ax.set_xlabel('Degree of Polynomial')
    ax.set_ylabel('Test MSE')
    ax.set_title('Test MSE vs Degree of Polynomial')
    plt.show()


X_train, X_test, y_train, y_test = train_test_split(wage[['age']], wage[['wage']], test_size=0.1)
polynomial_regression(X_train, y_train.values, X_test, y_test, [1,2,3,4,5,6,8,9,10,11,12,13,14,15])
```

{{% fluid_img "/img/Moving%20Beyond%20Linearity_files/Moving%20Beyond%20Linearity_23_0.png" %}}


```python
import statsmodels.api as sm

poly = preprocessing.PolynomialFeatures(degree=1)
X_ = poly.fit_transform(wage[['age']])
model1 = sm.OLS(wage[['wage']], X_)
model1 = model1.fit()

poly = preprocessing.PolynomialFeatures(degree=2)
X_ = poly.fit_transform(wage[['age']])
model2 = sm.OLS(wage[['wage']], X_)
model2 = model2.fit()

poly = preprocessing.PolynomialFeatures(degree=3)
X_ = poly.fit_transform(wage[['age']])
model3 = sm.OLS(wage[['wage']], X_)
model3 = model3.fit()

poly = preprocessing.PolynomialFeatures(degree=4)
X_ = poly.fit_transform(wage[['age']])
model4 = sm.OLS(wage[['wage']], X_)
model4 = model4.fit()

poly = preprocessing.PolynomialFeatures(degree=5)
X_ = poly.fit_transform(wage[['age']])
model5 = sm.OLS(wage[['wage']], X_)
model5 = model5.fit()

sm.stats.anova_lm(model1, model2, model3, model4, model5)
```

    /Users/amitrajan/Desktop/PythonVirtualEnv/Python3_VirtualEnv/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /Users/amitrajan/Desktop/PythonVirtualEnv/Python3_VirtualEnv/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /Users/amitrajan/Desktop/PythonVirtualEnv/Python3_VirtualEnv/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)





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
      <th>df_resid</th>
      <th>ssr</th>
      <th>df_diff</th>
      <th>ss_diff</th>
      <th>F</th>
      <th>Pr(&gt;F)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2998.0</td>
      <td>5.022216e+06</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2997.0</td>
      <td>4.793430e+06</td>
      <td>1.0</td>
      <td>228786.010128</td>
      <td>143.593107</td>
      <td>2.363850e-32</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2996.0</td>
      <td>4.777674e+06</td>
      <td>1.0</td>
      <td>15755.693664</td>
      <td>9.888756</td>
      <td>1.679202e-03</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2995.0</td>
      <td>4.771604e+06</td>
      <td>1.0</td>
      <td>6070.152124</td>
      <td>3.809813</td>
      <td>5.104620e-02</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2994.0</td>
      <td>4.770322e+06</td>
      <td>1.0</td>
      <td>1282.563017</td>
      <td>0.804976</td>
      <td>3.696820e-01</td>
    </tr>
  </tbody>
</table>
</div>



(b) Fit a step function to predict wage using age, and perform crossvalidation to choose the optimal number of cuts. Make a plot of the fit obtained. Code Source: https://rpubs.com/ppaquay/65563

Q9. This question uses the variables dis (the weighted mean of distances to five Boston employment centers) and nox (nitrogen oxides concentration in parts per 10 million) from the Boston data. We will treat dis as the predictor and nox as the response.

(a) Use the poly() function to fit a cubic polynomial regression to predict nox using dis. Report the regression output, and plot the resulting data and polynomial fits.


```python
boston = pd.read_csv("data/Boston.csv")
boston.dropna(inplace=True)
```


```python
poly = preprocessing.PolynomialFeatures(degree=3)
X_ = poly.fit_transform(boston[['dis']])
model = sm.OLS(boston[['nox']], X_)
result = model.fit()
print(result.summary())
```

                                OLS Regression Results
    ==============================================================================
    Dep. Variable:                    nox   R-squared:                       0.715
    Model:                            OLS   Adj. R-squared:                  0.713
    Method:                 Least Squares   F-statistic:                     419.3
    Date:                Thu, 20 Sep 2018   Prob (F-statistic):          2.71e-136
    Time:                        19:22:18   Log-Likelihood:                 690.44
    No. Observations:                 506   AIC:                            -1373.
    Df Residuals:                     502   BIC:                            -1356.
    Df Model:                           3
    Covariance Type:            nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.9341      0.021     45.110      0.000       0.893       0.975
    x1            -0.1821      0.015    -12.389      0.000      -0.211      -0.153
    x2             0.0219      0.003      7.476      0.000       0.016       0.028
    x3            -0.0009      0.000     -5.124      0.000      -0.001      -0.001
    ==============================================================================
    Omnibus:                       64.176   Durbin-Watson:                   0.286
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               87.386
    Skew:                           0.917   Prob(JB):                     1.06e-19
    Kurtosis:                       3.886   Cond. No.                     2.10e+03
    ==============================================================================

    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 2.1e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.



```python
x = np.linspace(boston['dis'].min()-1, boston['dis'].max()+1, 256, endpoint = True)
y = result.params.const + result.params.x1*x + result.params.x2*(x**2) + result.params.x3*(x**3)

fig = plt.figure(figsize=(15, 8))
plt.plot(x, y, 'red', label="Cubic Fit")
plt.scatter(boston['dis'], boston['nox'], alpha=1, color='green')

plt.xlabel('dis')
plt.ylabel('nox')
plt.title('Cubic Fit')
plt.legend(loc='best')

plt.grid()
plt.show()
```

{{% fluid_img "/img/Moving%20Beyond%20Linearity_files/Moving%20Beyond%20Linearity_29_0.png" %}}


(b) Plot the polynomial fits for a range of different polynomial degrees (say, from 1 to 10), and report the associated residual sum of squares.


```python
def add_polt(ax, x, y, RSS, degree):
    ax.plot(x, y, 'red', label="Degree: " +str(degree))
    ax.scatter(boston['dis'], boston['nox'], alpha=1, color='green')
    ax.set_ylim(boston['nox'].min()-0.1, boston['nox'].max()+0.1)
    ax.set_xlabel('dis')
    ax.set_ylabel('nox')
    ax.set_title('Degree: ' +str(degree) + ' Residuals: ' + str(RSS))
    ax.legend(loc='best')
    ax.grid()

fig = plt.figure(figsize=(15, 40))
x = np.linspace(boston['dis'].min()-1, boston['dis'].max()+1, 256, endpoint = True)

degree = 1
poly = preprocessing.PolynomialFeatures(degree=degree)
X_ = poly.fit_transform(boston[['dis']])
model = sm.OLS(boston[['nox']], X_)
result = model.fit()
y = result.params.const + result.params.x1*x
ax = fig.add_subplot(5, 2, degree)
add_polt(ax, x, y, result.ssr, degree)

degree = 2
poly = preprocessing.PolynomialFeatures(degree=degree)
X_ = poly.fit_transform(boston[['dis']])
model = sm.OLS(boston[['nox']], X_)
result = model.fit()
y = result.params.const + result.params.x1*x + result.params.x2*(x**2)
ax = fig.add_subplot(5, 2, degree)
add_polt(ax, x, y, result.ssr, degree)

degree = 3
poly = preprocessing.PolynomialFeatures(degree=degree)
X_ = poly.fit_transform(boston[['dis']])
model = sm.OLS(boston[['nox']], X_)
result = model.fit()
y = result.params.const + result.params.x1*x + result.params.x2*(x**2) + result.params.x3*(x**3)
ax = fig.add_subplot(5, 2, degree)
add_polt(ax, x, y, result.ssr, degree)

degree = 4
poly = preprocessing.PolynomialFeatures(degree=degree)
X_ = poly.fit_transform(boston[['dis']])
model = sm.OLS(boston[['nox']], X_)
result = model.fit()
y = result.params.const + result.params.x1*x + result.params.x2*(x**2) + result.params.x3*(x**3)
+ result.params.x4*(x**4)
ax = fig.add_subplot(5, 2, degree)
add_polt(ax, x, y, result.ssr, degree)

plt.show()
```

{{% fluid_img "/img/Moving%20Beyond%20Linearity_files/Moving%20Beyond%20Linearity_31_0.png" %}}


(c) Perform cross-validation or another approach to select the optimal degree for the polynomial, and explain your results.

<b>Sol:</b> The optimal degree of polynomial is <b>8</b>. Achieved test MSE at this value is decent as well.


```python
np.random.seed(5)
X_train, X_test, y_train, y_test = train_test_split(boston[['dis']], boston[['nox']], test_size=0.2)
polynomial_regression(X_train, y_train.values, X_test, y_test, [1,2,3,4,5,6,8,9,10])
```

{{% fluid_img "/img/Moving%20Beyond%20Linearity_files/Moving%20Beyond%20Linearity_33_0.png" %}}


(d) Use the bs() function to fit a regression spline to predict nox using dis. Report the output for the fit using four degrees of freedom. How did you choose the knots? Plot the resulting fit.

(e) Now fit a regression spline for a range of degrees of freedom, and plot the resulting fits and report the resulting RSS. Describe the results obtained.

(f) Perform cross-validation or another approach in order to select the best degrees of freedom for a regression spline on this data. Describe your results.

Q10. This question relates to the College data set.

(a) Split the data into a training set and a test set. Using out-of-state-tuition as the response and the other variables as the predictors, perform forward stepwise selection on the training set in order to identify a satisfactory model that uses just a subset of the predictors.


```python
college = pd.read_csv("data/College.csv")
college = college.rename(columns={'Unnamed: 0': 'Name'})
college['Private'] = college['Private'].map({'Yes': 1, 'No': 0})
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
      <th>Name</th>
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
      <td>1</td>
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
      <td>1</td>
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
      <td>1</td>
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
      <td>1</td>
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
      <td>1</td>
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




```python
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error

np.random.seed(1)
msk = np.random.rand(len(college)) < 0.8
train = college[msk]
test = college[~msk]

total_features = 17
validation_MSE = {}
for m in range(1, total_features+1):
    estimator = LinearRegression(fit_intercept=True)
    selector = RFE(estimator, m, step=1)
    selector.fit(train.drop(['Name', 'Outstate'], axis=1), train['Outstate'])
    predictions = selector.predict(test.drop(['Name', 'Outstate'], axis=1))
    mse = mean_squared_error(predictions, test['Outstate'])
    validation_MSE[m] = mse/len(test)

# Plot validation MSE
lists = sorted(validation_MSE.items()) # sorted by key, return a list of tuples
x, y = zip(*lists) # unpack a list of pairs into two tuples
fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(111)
plt.plot(x, y, color='r')
plt.grid()
ax.set_xlabel('Number of features')
ax.set_ylabel('Validation MSE')
ax.set_title('Validation MSE vs Number of features')
```




    Text(0.5,1,'Validation MSE vs Number of features')



{{% fluid_img "/img/Moving%20Beyond%20Linearity_files/Moving%20Beyond%20Linearity_37_1.png" %}}


(b) Fit a GAM on the training data, using out-of-state tuition as the response and the features selected in the previous step as the predictors. Plot the results, and explain your findings.

(c) Evaluate the model obtained on the test set, and explain the results obtained.

(d) For which variables, if any, is there evidence of a non-linear relationship with the response?
