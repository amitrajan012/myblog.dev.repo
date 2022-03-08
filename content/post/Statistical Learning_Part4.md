+++
date = "2018-05-06T07:24:34+01:00"
description = "ISLR Statistical Learning"
draft = false
tags = ["ISLR", "Statistical Learning", "Exercises", "Applied"]
title = "ISLR Chapter 2: Statistical Learning (Part 4: Exercises - Applied)"
topics = ["ISLR"]

+++

#### Applied

<b>Q8. This exercise relates to the College data set, which can be found in the file College.csv. It contains a number of variables for 777 different universities and colleges in the US.</b>


```python
import pandas as pd

college = pd.read_csv("data/College.csv")
college.set_index('Unnamed: 0', drop=True, inplace=True)
college.index.names = ['Name']
college.describe()
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
      <th>count</th>
      <td>777.000000</td>
      <td>777.000000</td>
      <td>777.000000</td>
      <td>777.000000</td>
      <td>777.000000</td>
      <td>777.000000</td>
      <td>777.000000</td>
      <td>777.000000</td>
      <td>777.000000</td>
      <td>777.000000</td>
      <td>777.000000</td>
      <td>777.000000</td>
      <td>777.000000</td>
      <td>777.000000</td>
      <td>777.000000</td>
      <td>777.000000</td>
      <td>777.00000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3001.638353</td>
      <td>2018.804376</td>
      <td>779.972973</td>
      <td>27.558559</td>
      <td>55.796654</td>
      <td>3699.907336</td>
      <td>855.298584</td>
      <td>10440.669241</td>
      <td>4357.526384</td>
      <td>549.380952</td>
      <td>1340.642214</td>
      <td>72.660232</td>
      <td>79.702703</td>
      <td>14.089704</td>
      <td>22.743887</td>
      <td>9660.171171</td>
      <td>65.46332</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3870.201484</td>
      <td>2451.113971</td>
      <td>929.176190</td>
      <td>17.640364</td>
      <td>19.804778</td>
      <td>4850.420531</td>
      <td>1522.431887</td>
      <td>4023.016484</td>
      <td>1096.696416</td>
      <td>165.105360</td>
      <td>677.071454</td>
      <td>16.328155</td>
      <td>14.722359</td>
      <td>3.958349</td>
      <td>12.391801</td>
      <td>5221.768440</td>
      <td>17.17771</td>
    </tr>
    <tr>
      <th>min</th>
      <td>81.000000</td>
      <td>72.000000</td>
      <td>35.000000</td>
      <td>1.000000</td>
      <td>9.000000</td>
      <td>139.000000</td>
      <td>1.000000</td>
      <td>2340.000000</td>
      <td>1780.000000</td>
      <td>96.000000</td>
      <td>250.000000</td>
      <td>8.000000</td>
      <td>24.000000</td>
      <td>2.500000</td>
      <td>0.000000</td>
      <td>3186.000000</td>
      <td>10.00000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>776.000000</td>
      <td>604.000000</td>
      <td>242.000000</td>
      <td>15.000000</td>
      <td>41.000000</td>
      <td>992.000000</td>
      <td>95.000000</td>
      <td>7320.000000</td>
      <td>3597.000000</td>
      <td>470.000000</td>
      <td>850.000000</td>
      <td>62.000000</td>
      <td>71.000000</td>
      <td>11.500000</td>
      <td>13.000000</td>
      <td>6751.000000</td>
      <td>53.00000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1558.000000</td>
      <td>1110.000000</td>
      <td>434.000000</td>
      <td>23.000000</td>
      <td>54.000000</td>
      <td>1707.000000</td>
      <td>353.000000</td>
      <td>9990.000000</td>
      <td>4200.000000</td>
      <td>500.000000</td>
      <td>1200.000000</td>
      <td>75.000000</td>
      <td>82.000000</td>
      <td>13.600000</td>
      <td>21.000000</td>
      <td>8377.000000</td>
      <td>65.00000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3624.000000</td>
      <td>2424.000000</td>
      <td>902.000000</td>
      <td>35.000000</td>
      <td>69.000000</td>
      <td>4005.000000</td>
      <td>967.000000</td>
      <td>12925.000000</td>
      <td>5050.000000</td>
      <td>600.000000</td>
      <td>1700.000000</td>
      <td>85.000000</td>
      <td>92.000000</td>
      <td>16.500000</td>
      <td>31.000000</td>
      <td>10830.000000</td>
      <td>78.00000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>48094.000000</td>
      <td>26330.000000</td>
      <td>6392.000000</td>
      <td>96.000000</td>
      <td>100.000000</td>
      <td>31643.000000</td>
      <td>21836.000000</td>
      <td>21700.000000</td>
      <td>8124.000000</td>
      <td>2340.000000</td>
      <td>6800.000000</td>
      <td>103.000000</td>
      <td>100.000000</td>
      <td>39.800000</td>
      <td>64.000000</td>
      <td>56233.000000</td>
      <td>118.00000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# produce a scatterplot matrix of the first ten columns or variables of the data.
import seaborn as sns

sns.pairplot(college.loc[:, 'Apps':'Books'])
```

{{% fluid_img "/img/Statistical%20Learning_files/Statistical%20Learning_35_1.png" %}}

```python
# produce side-by-side boxplots of Outstate versus Private.
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)

sns.boxplot(x="Private", y="Outstate", data=college)
ax.set_xlabel('Private University')
ax.set_ylabel('Outstate Tution (in USD)')
ax.set_title('Outstate Tution vs University Type')
plt.show()
```

{{% fluid_img "/img/Statistical%20Learning_files/Statistical%20Learning_36_0.png" %}}


```python
# Create a new qualitative variable, called Elite, by binning the Top10perc variable.
college['Elite'] = 0
college.loc[college['Top10perc'] > 50, 'Elite'] = 1
print("Number of elite universities are: " +str(college['Elite'].sum()))

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)

sns.boxplot(x="Elite", y="Outstate", data=college)
ax.set_xlabel('Elite University')
ax.set_ylabel('Outstate Tution (in USD)')
ax.set_title('Outstate Tution vs University Type')
plt.show()
```

    Number of elite universities are: 78

{{% fluid_img "/img/Statistical%20Learning_files/Statistical%20Learning_37_1.png" %}}


```python
# produce some histograms with differing numbers of bins for a few of the quantitative vari- ables.
college.head()

fig = plt.figure(figsize=(15, 8))

ax = fig.add_subplot(221)
sns.distplot(college['Books'], bins=20, kde=False, color='r', hist_kws=dict(edgecolor='black', linewidth=1))
ax.set_xlabel('Estinated Books Cost')
ax.set_ylabel('Count')
ax.set_title('Books Cost')

ax = fig.add_subplot(222)
sns.distplot(college['PhD'], bins=20, kde=False, color='green', hist_kws=dict(edgecolor='black', linewidth=1))
ax.set_xlabel('Percent of faculty with Ph.D.’s')
ax.set_ylabel('Count')
ax.set_title('Percent of faculty with Ph.D.’s')

ax = fig.add_subplot(223)
sns.distplot(college['Grad.Rate'], bins=20, kde=False, color='blue', hist_kws=dict(edgecolor='black', linewidth=1))
ax.set_xlabel('Graduation rate')
ax.set_ylabel('Count')
ax.set_title('Graduation rate')

ax = fig.add_subplot(224)
sns.distplot(college['perc.alumni'], bins=20, kde=False, color='yellow', hist_kws=dict(edgecolor='black', linewidth=1))
ax.set_xlabel('Percent of alumni who donate')
ax.set_ylabel('Count')
ax.set_title('Percent of alumni who donate')

plt.tight_layout() #Stop subplots from overlapping
plt.show()
```

{{% fluid_img "/img/Statistical%20Learning_files/Statistical%20Learning_38_0.png" %}}


<b>Q9. This exercise involves the Auto data set studied in the lab. Make sure that the missing values have been removed from the data.</b>


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



(a) Which of the predictors are quantitative, and which are qualitative?

<b>Sol:</b> Quantitative: displacement, weight, horsepower, acceleration, mpg

Qualitative: cylinders, year, origin

(b) What is the range of each quantitative predictor?


```python
print("Range of displacement: " + str(auto['displacement'].min()) + " - " + str(auto['displacement'].max()))
print("Range of weight: " + str(auto['weight'].min()) + " - " + str(auto['weight'].max()))
print("Range of horsepower: " + str(auto['horsepower'].min()) + " - " + str(auto['horsepower'].max()))
print("Range of acceleration: " + str(auto['acceleration'].min()) + " - " + str(auto['acceleration'].max()))
print("Range of mpg: " + str(auto['mpg'].min()) + " - " + str(auto['mpg'].max()))
```

    Range of displacement: 68.0 - 455.0
    Range of weight: 1613 - 5140
    Range of horsepower: 46 - 230
    Range of acceleration: 8.0 - 24.8
    Range of mpg: 9.0 - 46.6


(c) What is the mean and standard deviation of each quantitative predictor?


```python
auto.describe()[['displacement', 'weight', 'horsepower', 'acceleration', 'mpg']].loc[['mean', 'std']]
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
      <th>displacement</th>
      <th>weight</th>
      <th>horsepower</th>
      <th>acceleration</th>
      <th>mpg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mean</th>
      <td>194.411990</td>
      <td>2977.584184</td>
      <td>104.469388</td>
      <td>15.541327</td>
      <td>23.445918</td>
    </tr>
    <tr>
      <th>std</th>
      <td>104.644004</td>
      <td>849.402560</td>
      <td>38.491160</td>
      <td>2.758864</td>
      <td>7.805007</td>
    </tr>
  </tbody>
</table>
</div>



(d) Now remove the 10th through 85th observations. What is the range, mean, and standard deviation of each predictor in the subset of the data that remains?


```python
temp = auto.drop(auto.index[10:85], axis=0)
temp.describe().loc[['mean', 'std', 'min', 'max']]
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mean</th>
      <td>24.374763</td>
      <td>5.381703</td>
      <td>187.880126</td>
      <td>101.003155</td>
      <td>2938.854890</td>
      <td>15.704101</td>
      <td>77.123028</td>
      <td>1.599369</td>
    </tr>
    <tr>
      <th>std</th>
      <td>7.872565</td>
      <td>1.658135</td>
      <td>100.169973</td>
      <td>36.003208</td>
      <td>811.640668</td>
      <td>2.719913</td>
      <td>3.127158</td>
      <td>0.819308</td>
    </tr>
    <tr>
      <th>min</th>
      <td>11.000000</td>
      <td>3.000000</td>
      <td>68.000000</td>
      <td>46.000000</td>
      <td>1649.000000</td>
      <td>8.500000</td>
      <td>70.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>46.600000</td>
      <td>8.000000</td>
      <td>455.000000</td>
      <td>230.000000</td>
      <td>4997.000000</td>
      <td>24.800000</td>
      <td>82.000000</td>
      <td>3.000000</td>
    </tr>
  </tbody>
</table>
</div>



(e) Using the full data set, investigate the predictors graphically, using scatterplots or other tools of your choice.

<b>Sol:</b> Two scatterplots of all the quantitative variables segregated by cylinders and origin is shown below. It is evident that vehicles with higher number of cylinders have higher displacement, weight and horsepower, while lower acceleration and mpg. The relationship of mpg with displacement, weight and horsepower is somewhat predictable. Similarly, the relationships of horsepower, weight and displacement with all the other variables follow a trend. Vehicles are somehow distinguishable by origin as well.


```python
# Scatter plot of quantitative variables
sns.pairplot(auto, vars=['displacement', 'weight', 'horsepower', 'acceleration', 'mpg'], hue='cylinders')
```

{{% fluid_img "/img/Statistical%20Learning_files/Statistical%20Learning_49_2.png" %}}


```python
sns.pairplot(auto, vars=['displacement', 'weight', 'horsepower', 'acceleration', 'mpg'], hue='origin')
```

{{% fluid_img "/img/Statistical%20Learning_files/Statistical%20Learning_50_2.png" %}}


(f) Suppose that we wish to predict gas mileage (mpg) on the basis of the other variables. Do your plots suggest that any of the other variables might be useful in predicting mpg? Justify your answer.

<b>Sol:</b> From the plot, it is evident that displacement, weight and horsepower can play a significant role in the prediction of mpg. As displacement is highly correlated with weight and horsepower, we can pick any one of them for the prediction. Origin and cylinders can also be used for prediction.

<b>Q10. This exercise involves the Boston housing data set.</b>

(a) How many rows are in this data set? How many columns? What do the rows and columns represent?


```python
from sklearn.datasets import load_boston

boston = load_boston()
print("(Rows, Cols): " +str(boston.data.shape))
print(boston.DESCR)
```

    (Rows, Cols): (506, 13)
    Boston House Prices dataset
    ===========================

    Notes
    ------
    Data Set Characteristics:

        :Number of Instances: 506

        :Number of Attributes: 13 numeric/categorical predictive

        :Median Value (attribute 14) is usually the target

        :Attribute Information (in order):
            - CRIM     per capita crime rate by town
            - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
            - INDUS    proportion of non-retail business acres per town
            - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
            - NOX      nitric oxides concentration (parts per 10 million)
            - RM       average number of rooms per dwelling
            - AGE      proportion of owner-occupied units built prior to 1940
            - DIS      weighted distances to five Boston employment centres
            - RAD      index of accessibility to radial highways
            - TAX      full-value property-tax rate per $10,000
            - PTRATIO  pupil-teacher ratio by town
            - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
            - LSTAT    % lower status of the population
            - MEDV     Median value of owner-occupied homes in $1000's

        :Missing Attribute Values: None

        :Creator: Harrison, D. and Rubinfeld, D.L.

    This is a copy of UCI ML housing dataset.
    http://archive.ics.uci.edu/ml/datasets/Housing


    This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.

    The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic
    prices and the demand for clean air', J. Environ. Economics & Management,
    vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics
    ...', Wiley, 1980.   N.B. Various transformations are used in the table on
    pages 244-261 of the latter.

    The Boston house-price data has been used in many machine learning papers that address regression
    problems.

    **References**

       - Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', Wiley, 1980. 244-261.
       - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.
       - many more! (see http://archive.ics.uci.edu/ml/datasets/Housing)



(b) Make some pairwise scatterplots of the predictors (columns) in this data set. Describe your findings.

<b>Sol:</b> Pairwise scatterplot of <b>nitric oxides concentration</b> vs <b>weighted distances to five Boston employment centres</b> shows that as distance decreases, the concentration of nitrous oxide increases.


```python
df_boston = pd.DataFrame(boston.data, columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
                                               'PTRATIO', 'B', 'LSTAT'])

sns.pairplot(df_boston, vars=['NOX', 'DIS'])
```

{{% fluid_img "/img/Statistical%20Learning_files/Statistical%20Learning_55_1.png" %}}


(c) Are any of the predictors associated with per capita crime rate? If so, explain the relationship.

<b>Sol:</b> As most of the area has crime rate less than 20%, we will analyze the scatterplot for those areas only.


```python
fig = plt.figure(figsize=(15, 8))

ax = fig.add_subplot(111)
sns.distplot(df_boston['CRIM'], bins=20, kde=False, color='r', hist_kws=dict(edgecolor='black', linewidth=1))
ax.set_xlabel('Crime Rate')
ax.set_ylabel('Count')
ax.set_title('Crime Rate Histogram')
```




    Text(0.5,1,'Crime Rate Histogram')


{{% fluid_img "/img/Statistical%20Learning_files/Statistical%20Learning_57_1.png" %}}


```python
temp = df_boston[df_boston['CRIM'] <= 20]
sns.pairplot(temp, y_vars=['CRIM'], x_vars=['NOX', 'RM', 'AGE', 'DIS', 'LSTAT'])
```

{{% fluid_img "/img/Statistical%20Learning_files/Statistical%20Learning_58_1.png" %}}


(d) Do any of the suburbs of Boston appear to have particularly high crime rates? Tax rates? Pupil-teacher ratios? Comment on the range of each predictor.


```python
print("Count of suburbs with higher crime rate: " + str(df_boston[df_boston['CRIM'] > 20].shape[0]))
print("Count of suburbs with higher tax rate: " + str(df_boston[df_boston['TAX'] > 600].shape[0]))
print("Count of suburbs with higher pupil-teacher ratio: " + str(df_boston[df_boston['PTRATIO'] > 20].shape[0]))
```

    Count of suburbs with higher crime rate: 18
    Count of suburbs with higher tax rate: 137
    Count of suburbs with higher pupil-teacher ratio: 201


(e) How many of the suburbs in this data set bound the Charles river?


```python
print("Suburbs bound the Charles river: " + str(df_boston[df_boston['CHAS'] == 1].shape[0]))
```

    Suburbs bound the Charles river: 35


(f) What is the median pupil-teacher ratio among the towns in this data set?


```python
print("Median pupil-teacher ratio is: " + str(df_boston['PTRATIO'].median()))
```

    Median pupil-teacher ratio is: 19.05


(h) In this data set, how many of the suburbs average more than seven rooms per dwelling? More than eight rooms per dwelling?


```python
print("Suburbs with average more than 7 rooms per dwelling: " + str(df_boston[df_boston['RM'] > 7].shape[0]))
print("Suburbs with average more than 7 rooms per dwelling: " + str(df_boston[df_boston['RM'] > 8].shape[0]))
```

    Suburbs with average more than 7 rooms per dwelling: 64
    Suburbs with average more than 7 rooms per dwelling: 13
