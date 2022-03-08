+++
date = "2018-08-05T10:02:08+01:00"
description = "Think Stats: Chapter 1"
draft = false
tags = ["Think Stats"]
title = "Think Stats: Chapter 1"
topics = ["Think Stats"]

+++


### 1.1 Do first babies arrive late?

<b>Anecdotal Evidence </b> is based on data that is unpublished and usually personal. For example,
<i><center>"My two friends that have given birth recently to their first babies,
BOTH went almost 2 weeks overdue before going into
labour or being induced.” </center></i>
Anecdotal Evidence usually fail because of <b>Small number of observations</b>, <b>Selection bias</b> (People who join a discussion of this question might be interested because their first babies were late.), <b>Confirmation bias</b> (People who believe the claim might be more likely to contribute examples that confirm it) and <b>Inaccuracy</b>.



### 1.2 A Statistical Approach

Limitations of Anecdotal Evidence can be addressed by using the tools of statistics, which include <b>Data Collection</b>, <b>Descriptive Statistics</b>, <b>Exploratory Data Analysis</b>, <b>Hypothesis Testing</b> and <b>Estimation</b>.

### 1.3 The National Survey of Family Growth (NSFG)

NSFG is a <b>cross-sectional</b> study (it captures a snapshot of a group at a point in time). The alternative is a <b>longitudinal</b> study which observes a group repeatedly over a period of time. The people who participate in a survey are called <b>respondents</b>. Cross-sectional studies are meant to be <b>representative</b>, which means that every member of the target population has an equal chance of participating. NSFG is deliberately <b>oversampled</b> as certain groups are sampled at higher rates compared to their representation in US population. Drawback of oversampling is that it is hard to arrive at a conclusion based on statistics from the survey.
<br></br>
<b>Exercise 1.2</b> Download data from the NSFG:


```python
import pandas as pd
# Reference to extract the columns: http://greenteapress.com/thinkstats/survey.py
pregnancies = pd.read_fwf("2002FemPreg.dat",
                         names=["caseid", "nbrnaliv", "babysex", "birthwgt_lb",
                               "birthwgt_oz", "prglength", "outcome", "birthord",
                               "agepreg", "finalwgt"],
                         colspecs=[(0, 12), (21, 22), (55, 56), (57, 58), (58, 60),
                                (274, 276), (276, 277), (278, 279), (283, 285), (422, 439)])
pregnancies.head()
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
      <th>caseid</th>
      <th>nbrnaliv</th>
      <th>babysex</th>
      <th>birthwgt_lb</th>
      <th>birthwgt_oz</th>
      <th>prglength</th>
      <th>outcome</th>
      <th>birthord</th>
      <th>agepreg</th>
      <th>finalwgt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>8.0</td>
      <td>13.0</td>
      <td>39</td>
      <td>1</td>
      <td>1.0</td>
      <td>33.0</td>
      <td>6448.271112</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>7.0</td>
      <td>14.0</td>
      <td>39</td>
      <td>1</td>
      <td>2.0</td>
      <td>39.0</td>
      <td>6448.271112</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>9.0</td>
      <td>2.0</td>
      <td>39</td>
      <td>1</td>
      <td>1.0</td>
      <td>14.0</td>
      <td>12999.542264</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>39</td>
      <td>1</td>
      <td>2.0</td>
      <td>17.0</td>
      <td>12999.542264</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>39</td>
      <td>1</td>
      <td>3.0</td>
      <td>18.0</td>
      <td>12999.542264</td>
    </tr>
  </tbody>
</table>
</div>



The description for the fields are as follows:

| caseid | prglength | outcome | birthord | finalwgt
| --- | --- | --- | --- | --- |
| Integer ID of Respondent | Integer Duration of pregnancy in weeks | 1 indicates a live birth | code for first child: 1 | Number of people in US population this respondant represents

</br>

<b>Exercise 1.3</b> Explore the data in the Pregnancies table. Count the number of live births and compute the average pregnancy length (in weeks) for first babies and others for the live births.


```python
pregnancies.describe()
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
      <th>caseid</th>
      <th>nbrnaliv</th>
      <th>babysex</th>
      <th>birthwgt_lb</th>
      <th>birthwgt_oz</th>
      <th>prglength</th>
      <th>outcome</th>
      <th>birthord</th>
      <th>agepreg</th>
      <th>finalwgt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>13593.000000</td>
      <td>9148.000000</td>
      <td>9144.000000</td>
      <td>9144.000000</td>
      <td>9087.000000</td>
      <td>13593.000000</td>
      <td>13593.000000</td>
      <td>9148.000000</td>
      <td>13241.000000</td>
      <td>13593.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>6216.526595</td>
      <td>1.025907</td>
      <td>1.494532</td>
      <td>6.653762</td>
      <td>7.403874</td>
      <td>29.531229</td>
      <td>1.763996</td>
      <td>1.824552</td>
      <td>24.230949</td>
      <td>8196.422280</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3645.417341</td>
      <td>0.252864</td>
      <td>0.515295</td>
      <td>1.588809</td>
      <td>8.097454</td>
      <td>13.802523</td>
      <td>1.315930</td>
      <td>1.037053</td>
      <td>5.824302</td>
      <td>9325.918114</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>10.000000</td>
      <td>118.656790</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3022.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>6.000000</td>
      <td>3.000000</td>
      <td>13.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>20.000000</td>
      <td>3841.375308</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>6161.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>7.000000</td>
      <td>7.000000</td>
      <td>39.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>23.000000</td>
      <td>6256.592133</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>9423.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>8.000000</td>
      <td>11.000000</td>
      <td>39.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>28.000000</td>
      <td>9432.360931</td>
    </tr>
    <tr>
      <th>max</th>
      <td>12571.000000</td>
      <td>9.000000</td>
      <td>9.000000</td>
      <td>9.000000</td>
      <td>99.000000</td>
      <td>50.000000</td>
      <td>6.000000</td>
      <td>9.000000</td>
      <td>44.000000</td>
      <td>261879.953864</td>
    </tr>
  </tbody>
</table>
</div>




```python
live_births = pregnancies[pregnancies['outcome'] == 1]
print("Number of live births is: " + str(live_births.shape[0]))
mean_first = live_births[live_births['birthord'] == 1]['prglength'].mean()
mean_other = live_births[live_births['birthord'] != 1]['prglength'].mean()
print("Mean Pregnancy length for live births of first babies is: " + str(mean_first))
print("Mean Pregnancy length for live births of other babies is: " + str(mean_other))
print("Difference in Mean Pregnancy length for first and other babies is : " + str(mean_first - mean_other))
```

    Number of live births is: 9148
    Mean Pregnancy length for live births of first babies is: 38.6009517335
    Mean Pregnancy length for live births of other babies is: 38.5229144667
    Difference in Mean Pregnancy length for first and other babies is : 0.0780372667775

</br>

### 1.5 Significance

From the above analysis, it is evident that the difference in mean pregnancy lengths of first and other babies is <b>13.11 hours</b>. A difference like this is called an <b>apparent effect</b> which means that there must be something going on but we are not sure yet. If the difference occurred by chance, we can conlcude that thet effect was not <b>statistically significant</b>. An apparent effect that is caused by bias, measurement error, or
some other kind of error is called <b>artifact</b>.
