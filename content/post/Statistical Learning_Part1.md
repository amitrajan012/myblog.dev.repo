+++
date = "2018-05-04T09:32:21+01:00"
description = "ISLR Statistical Learning"
draft = false
tags = ["ISLR", "Statistical Learning"]
title = "ISLR Chapter 2: Statistical Learning (Part 1: What Is Statistical Learning?)"
topics = ["ISLR"]

+++

<h1><center>Statistical Learning</center></h1>

### 2.1 What Is Statistical Learning?

The <b>Advertising</b> data set consists of the sales of that product in 200 different markets, along with advertising budgets for the product in each of those markets for three different media: <b>TV, radio, and newspaper.</b> The plot of data is shown below. Our goal is to develop an accurate model that can be used to predict sales on the basis of the three media budgets.

In general, suppose that we observe a quantitative response $Y$ and $p$ different predictors, $X_1$, $X_2$, ... , $X_p$, we assume that there is some relationship between $Y$ and $X = (X_1, X_2, ... , X_p)$, which can be written in general form as:

$$Y = f(X) + \epsilon$$

where, $\epsilon$ is a random error term which is independent of X and has mean 0. In this formula, $f$ represents the systematic information that $X$ provides about $Y$. <b>In essence, statistical learning refers to a set of approaches for estimating  $f$.</b>


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

adv = pd.read_csv("data/Advertising.csv")

fig = plt.figure(figsize=(15,8))

ax = fig.add_subplot(131)
sns.regplot(x="TV", y="Sales", color='r', data=adv, order=1, scatter_kws={'alpha':0.4},
            line_kws={'color':'g', 'alpha':0.7})
ax.set_xlabel('TV')
ax.set_ylabel('Sales')

ax = fig.add_subplot(132)
sns.regplot(x="Radio", y="Sales", color='r', data=adv, order=1, scatter_kws={'alpha':0.4},
            line_kws={'color':'g', 'alpha':0.7})
ax.set_xlabel('Radio')
ax.set_ylabel('Sales')

ax = fig.add_subplot(133)
sns.regplot(x="Newspaper", y="Sales", color='r', data=adv, order=1, scatter_kws={'alpha':0.4},
            line_kws={'color':'g', 'alpha':0.7})
ax.set_xlabel('Newspaper')
ax.set_ylabel('Sales')

plt.show()
```

{{% fluid_img "/img/Statistical%20Learning_files/Statistical%20Learning_4_1.png" %}}


#### 2.1.1 Why Estimate f ?

There are two main reasons that we may wish to estimate $f$:
 - <b>Prediction:</b> We can predict $Y$ using $X$ as:

 $$\widehat{Y} = \widehat{f}(X)$$

 where $\widehat{f}$ represents our estimate for $f$ and $\widehat{Y}$ represents the resulting prediction for $Y$. The accuracy of $\widehat{Y}$ as a prediction of $Y$ depends on two quantities: <b>reducible error</b> and <b>irreducible error</b>. Reducible error can be minimized by opting a better prediction. Irreducible error arises due to $Y$'s dependence on $\epsilon$ and hence can not be reduced further.


 - <b>Inference:</b> In this situation we wish to estimate $f$, but our goal is not necessarily to make predictions for $Y$. Instead, we want to understand the relationship between $X$ and $Y$, or more specifically, we want to understand how $Y$ changes as a function of $X$. The three main questions that can be of interest while doing inference analysis are: <b>Which predictors are associated with the response?, What is the relationship (positive or negative) between the response and each predictor?</b> and <b>Can the relationship between Y and each predictor be adequately summarized using a linear equation, or is the relationship more complicated?</b>

#### 2.1.2 How Do We Estimate f ?

$f$ can be estimated by many <b>linear</b> and <b>non-linear</b> approaches. The given dataset is called as <b>training data</b>. Our goal is to apply a statistical learning method to the training data in order to estimate $f$. In other words, we want to find a function $\widehat{f}$ such that $Y \approx \widehat{f}(X)$ for an observation $(X,Y)$. The most statistical learning methods for this task can be characterized as:

 - <b>Parametric Methods:</b> Parametric methods involve a two-step model based approach which are:

    - We make an assumption about the shape of $f$. For a liner model,  $f$ can be represented as:

    $$f(X) = \beta_0 + \beta_1 X_1 + ... + \beta_p X_p$$

    and then we have to estimate $p+1$ coefficients $\beta_0, \beta_1,..., \beta_p$.

    - After the model has been selected, we need a procedure that uses the <b>training data</b> to estimate the parameters.

    The model based approach described above is called as <b>parametric</b>, as it reduces the problem of estimating $f$ down to the estimation of a set of parameters. Parametric form is easy to estimate but with a disadvantage that the chosen model will not match the true unknown form of $f$. We can design more complex models to overcome this issue but this leads to <b>overfitting</b> the data as it follows the <b>error</b> or <b>noise</b> too closely.


 - <b>Non-parametric Methods:</b> Non-parametric methods seek an estimate of $f$ that gets as close to the data points as possible without being too rough or wiggly. They have the potential to accurately fit a wider range of possible shapes of $f$. Since they do not reduce the problem of estimating $f$ to a small number of parameters, a very large number of observations is required to obtain an accurate estimate of $f$.

#### 2.1.3 The Trade-Off Between Prediction Accuracy and Model Interpretability

A common question related to model selection is: <b>why would we ever choose to use a more restrictive method instead of a very flexible approach?</b> When inference is the goal, linear model (restrictive) is a good choice as it is easy to interpret. So we can conclude that when inference is the goal, simple and relatively inflexible statistical learning methods have a clear advantage. For prediction, a more flexible and complex model can be used as interpretability is not a concern.

#### 2.1.4 Supervised Versus Unsupervised Learning

In supervised learning, for each observation of the predictor measurement, there is an associated response measurement which is used to build a model and predict the response of future observations.

In unsupervised learning, there is no response associated with the observations. Cluster analysis is an example of unsupervised learning.

#### 2.1.5 Regression Versus Classification Problems

Variables can be characterized as either <b>quantitative</b> or <b>qualitative (categorical)</b>. Quantitative variables take on numerical values, whereas, qualitative variables take on values in one of the $K$ different classes. We refer to problems with a quantitative response as <b>regression</b> problem, while those involving a qualitative response is called as <b>classification</b> problem. The type of <b>predictors</b> is not of much concern when choosing the model as categorical variables can easily be coded before applying the model.

### 2.2 Assessing Model Accuracy

No one statistical learning method dominates all other over all possible data sets. Hence it is an important task to decide that for any given data set which model fits best.

#### 2.2.1 Measuring the Quality of Fit

In the <b>regression</b> setting, the most commonly used measure for quality of fit is <b>mean squared error (MSE)</b>, which is given as:

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \widehat{f}(x_i))^{2}$$

where $\widehat{f}(x_i)$ is the prediction for the $i$th observation. MSE will be small if the predicted responses are very close to the true responses. While training a model, <b>training MSE</b> is of lesser significance. We should be more interested in <b>test MSE</b>, which is the MSE for the previously unseen test observation not used to train the model. When the test data is available, we can simply compute test MSE and select the model which has the lowest test MSE. In the absence of test data, the basic approach is to simply select a model with the lowest training MSE. Below figure shows the MSE of test and train data for a model.

{{% fluid_img "/img/Statistical%20Learning_files/MSE.png" %}}


In the right image, the grey curve shows training MSE and the red one test MSE. The <b>horizontal dashed line</b> indicates $Var(\epsilon)$, which is the lowest achievable test MSE amongst all methods. It is to be noted that as we increase <b>flexibility (degree of freedom)</b>, training MSE reduces but test MSE tends to increase after a certain point. So the blue curve on the left, which, although has a higher training MSE is the bset fit for the data.

The right hand side figure shows a fundamental property of a statistical model irrespective of the data set or the statistical methods being used. When a small method yields a small training MSE but a large test MSE, we are <b>overfitting</b> the data.

There are various approaches that can be used to find the best model (or find the minimum point) by analysing test MSE. One important method is <b>cross-validation</b>, which is a method for estimating test MSE using the training data.

#### 2.2.2 The Bias-Variance Trade-Off

The expected test MSE for a given value $x_0$ can always be decomposed into the sum of three fundamental quantities: <b>variance of $\widehat{f}(x_0)$</b>, <b>the squared bias of $\widehat{f}(x_0)$</b> and the variance of the <b>error term $\epsilon$</b>.

$$E(y_0 - \widehat{f}(x_0))^{2} = Var(\widehat{f}(x_0)) + [Bias(\widehat{f}(x_0))]^{2} + Var(\epsilon)$$

Here the notion $E(y_0 - \widehat{f}(x_0))^{2}$ defines the <b>expected test MSE</b> and refers to the average test MSE that would be obtained if we repeatedly estimated $f$ using a large number of training sets ane tested each at $x_0$. The overall expected test MSE can be computed by averaging it over all possible values of $x_0$ in the test set.

In order to minimize the expected test error, we need to select a statistical learning method that simultaneously achieves <b>low variance</b> and <b>low bias</b>.

<b>Variance</b> refers to the amount by which $\widehat{f}$ would change if we estimated it using a different training data. If a statistical method ($\widehat{f})$ has high variance, small change in the training data can result in a largr change in $\widehat{f}$. <b>More flexible statistical methods have higher variance.</b>

<b>Bias</b> refers to the error that is introduced by approximating a real-life problem, which may be extremely complicated, by a much simpler model. Generally, <b>more flexible methods result in less bias.</b>

As we use more flixible methods, <b>variance will increase and bias will decrease.</b> As we increase the flexibility of a statistical method, <b>the bias tends to initially decrease faster than the variance increases,</b> and hence the test MSE declines. Below figure illustrates the bias-variance tradeoff (with increasing flexibility) for the example shown above. Blue curve represents the squared bias, orange curve the variance and red curve the test MSE. It should be noted that as we increase the flexibility, bias decreases and variance increases. This phenomenon is referred to as <b>bias-variance tradeoff</b>, as it is easy to obtain a method with extremely low bias but high variance or a method with very low variance but high bias.

{{% fluid_img "/img/Statistical%20Learning_files/Bias Variance Tradeoff.png" %}}

#### 2.2.3 The Classification Setting

The most common approach for quantifying the accuracy of the estimate $\widehat{f}$ in the classification setting is the <b>training error rate</b>, defined as the proportion of mistakes that are made if we apply our estimate $\widehat{f}$ to the training observations:

$$\frac{1}{n}\sum_{i=1}^{n} I(y_i \neq \widehat{y_i})$$

where $\widehat{y_i}$ is the predicted class label for the $i$th observation and $I(y_i \neq \widehat{y_i})$ is the <b>indicator variable</b> that equals 1 if $y_i \neq \widehat{y_i}$. Hence, <b>the above equation computes the fraction of incorrect classifications.</b> Similarly, the <b>test error rate</b> associated with the test observations of the form $(x_0, y_0)$ can be calculated as:

$$Ave(I(y_0 \neq \widehat{y_0}))$$

#### The Bayes Classifier

In the classification setting, the test error rate can be minimized on an average by a simple classifier that <b>assign each observations to the most likely class, given its predictor values.</b> We can simply assign a test observation with predictor value $x_0$ to the class $j$ for which

$$Pr(Y=j \ | \  X = x_0)$$

is largest. This classifier is called as <b>Bayes Classifier.</b> In a two class classifier, Bayes classifier predicts the class <b>1</b>, if $Pr(Y=1 \ | \  X = x_0) > 0.5$, and class <b>2</b> otherwise. The boundary that divides the data set into classes (when probability of being in different classes is equal) is called as the <b>Bayes decision boundary</b>.

The Bayes classifier produces the lowest possible test error rate called as the <b>Bayes error rate.</b> Since the bayes classifier will always choose the class for which the probability is maximum, the error rate at $X = x_ 0$ will be
$1 - max_{j}Pr(Y=j \ | \ X=x_0)$. Hence, overall bayes error rate is given as:

$$1 - E(max_{j}Pr(Y=j \ | \ X=x_0)),$$

where exception averages the probability over all possible values in $X$.

#### K-Nearest Neighbors

For real data, we do not know the conditional distribution of Y given X and hence computing Bayes Classifier is impossible. Many approaches attempt to estimate the <b>conditional distribution of Y given X</b> and then classify a given observation to the calss with highest estimated probability. One such method is called the <b>K-nearest neighbors (KNN) classifier</b>.

Given a positive integer <b>K</b> and a test observation $x_0$, the KNN classifier first identifies the $K$ points in the training data that are closest to $x_0$, represented by $N_0$. It then estimates the conditional probability of class $j$ as the fraction of points in $N_0$ whose response values equal $j$:

$$Pr(Y = j \ | \ X = x_0) = \frac {1}{K} \sum _{i \in N_0}I(y_i = j)$$

Finally, KNN applies Bayes rule and classifies the test observation $x_0$ to the class with the largest probability.

Despite the fact that it is a very simple approach, KNN can often produce classifiers that are surprisingly close to the optimal Bayes classifier. <b>Choice of K has a drastic effect on KNN classifier.</b> Below figure shows the effect of K on <b>KNN decision boundary</b>. When <b>K=1</b>, KNN classifier is highly flexible and find patterns in the data that don't correspond to the Bayes decision boundary. Hence, <b>lower value of K corresponds to a classifier that has low bias but very high variance.</b> As K grows, the method becomes less flexible and produces a decision boundary that is close to linear. <b>Higher value of K corresponds to a low-variance but high-bias classifier.</b> For KNN, <b>1\K</b> serves as a measure of flexibility. As K decreases, 1\K increases and hence flexibility increases.

{{% fluid_img "/img/Statistical%20Learning_files/KNN.png" %}}

<b>In both the regression and classification settings, choosing the correct level of flexibility is critical to the success of any statistical learning method. The bias-variance tradeoff, and the resulting U-shape in the test error, can make this a difficult task.</b>

### 2.4 Exercises

#### Conceptual

<b>Q1. For each of parts (a) through (d), indicate whether we would generally expect the performance of a flexible statistical learning method to be better or worse than an inflexible method. Justify your answer. </b>

(a) The sample size n is extremely large, and the number of predic- tors p is small.

<b>Sol:</b> Better

(b) The number of predictors p is extremely large, and the number of observations n is small.

<b>Sol:</b> Worse

(c) The relationship between the predictors and response is highly non-linear.

<b>Sol:</b> Better

(d) The variance of the error terms, i.e. σ2 = Var(ε), is extremely high.

<b>Sol:</b> Worse, as flixible model will try to fit the error term too.


<b>Q2. Explain whether each scenario is a classification or regression prob- lem, and indicate whether we are most interested in inference or pre- diction. Finally, provide n and p.</b>

(a) We collect a set of data on the top 500 firms in the US. For each firm we record profit, number of employees, industry and the CEO salary. We are interested in understanding which factors affect CEO salary.

<b>Sol:</b> Regression and Inference; n=500, p=3

(b) We are considering launching a new product and wish to know whether it will be a success or a failure. We collect data on 20 similar products that were previously launched. For each product we have recorded whether it was a success or failure, price charged for the product, marketing budget, competition price, and ten other variables.

<b>Sol:</b> Classification and Prediction; n=20, p=13

(c) We are interested in predicting the % change in the USD/Euro exchange rate in relation to the weekly changes in the world stock markets. Hence we collect weekly data for all of 2012. For each week we record the % change in the USD/Euro, the % change in the US market, the % change in the British market, and the % change in the German market.

<b>Sol:</b> Regression and Prediction; n=Weeks in a year, p=3

<b>Q7. The table below provides a training data set containing six observa- tions, three predictors, and one qualitative response variable.</b>

{{% fluid_img "/img/Statistical%20Learning_files/2_7.png" %}}

Suppose we wish to use this data set to make a prediction for Y when X1 = X2 = X3 = 0 using K-nearest neighbors.

(a) Compute the Euclidean distance between each observation and thetestpoint,X1 =X2 =X3 =0.

<b>Sol:</b> The Euclidean distance between the test point and the observations is as follows:

{Obs 1: 3, Obs 2: 2, Obs 3: $\sqrt{10}$, Obs 4: $\sqrt{5}$, Obs 5: $\sqrt{2}$, Obs 6: $\sqrt{3}$}

(b) What is our prediction with K = 1? Why?

<b>Sol:</b> The prediction with K=1 is the class of the nearest point which is <b>Green</b>.

(c) What is our prediction with K = 3? Why?

<b>Sol:</b> The three nearest points are : Observations 5,6 and 2 with their classes as Green, Red and Red and hence the prediction will be <b>Red</b>.

(d) If the Bayes decision boundary in this problem is highly non- linear, then would we expect the best value for K to be large or small? Why?

<b>Sol:</b> For highly non-linear Bayes decision boundary, the value of K will be small.

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
