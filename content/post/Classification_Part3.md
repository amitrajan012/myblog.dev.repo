+++
date = "2018-05-15T09:02:08+01:00"
description = "ISLR Classification"
draft = false
tags = ["ISLR", "Classification", "Exercises", "Conceptual"]
title = "ISLR Chapter 4: Classification (Part 3: Exercises- Conceptual)"
topics = ["ISLR"]

+++

### 4.7 Exercises
#### Conceptual

Q1. Using a little bit of algebra, prove that the logistic function representation and logit representation
for the logistic regression model are equivalent.

<b>Sol: </b> Logistic function representation is given as:

$$p(X) = \frac{e^{\beta_0 + \beta_1X}}{1 + e^{\beta_0 + \beta_1X}}$$

then $1-p(X) = \frac{1}{1 + e^{\beta_0 + \beta_1X}}$, Taking the ratio of these two and then taking the log, we get

$$log\bigg( \frac{p(X)}{1-p(X)} \bigg) = \beta_0 + \beta_1X$$

Q2. Under the assumption that the observations in the $k$th class is drawn from a normal distribution, the Bayes' classifier assign an observation to the class for which the discriminant function is maximized.

<b>Sol:</b> The posterior probability for the Bayes' classifier is given as:

$$p_k(X) = \frac{\pi_k \frac{1}{\sqrt{2\pi}\sigma} exp \bigg( {-\frac{1}{2\sigma^2}(x - \mu_k)^2} \bigg)}
{\sum_{l=1}^{k}\pi_l\frac{1}{\sqrt{2\pi}\sigma} exp \bigg( {-\frac{1}{2\sigma^2}(x - \mu_l)^2} \bigg)}$$

In the Bayes' classifier, we have to find the class for which this is maximum. As log function is monotonically increasing, we can maximize the log of the posterior probability. Here the denominator is constant and hence for the maximization process, this can be ignored. For the constant variance case, the term $\frac{1}{\sqrt{2\pi}\sigma}$ in the neumerator is constant and hence can be ignored as well. Now taking the logarithm and expanding the quadratic term, we get:

$$log(p_k(X)) = \delta_k(X) = log(\pi_k) - \frac{\mu_k^2}{2\sigma^2} + x.\frac{\mu_k}{\sigma^2} - \frac{x^2}{2\sigma^2}$$

The last term is constant in $k$ as well and hence can be ignored, which gives:

$$\delta_k(X) = log(\pi_k) - \frac{\mu_k^2}{2\sigma^2} + x.\frac{\mu_k}{\sigma^2}$$

Q3. This problem relates to the QDA model, in which the observations within each class are drawn from a normal distribution with a classspecific mean vector and a class specific covariance matrix. We can not assume that the variance of each class is same. Prove that in this case, the Bayes’ classifier is not linear. Argue that it is in fact
quadratic.

<b>Sol:</b> Replacing $\sigma$ with $\sigma_k$ in the second equation in question 2 and not ignoring the term $\frac{1}{\sqrt{2\pi}\sigma_k}$ in the neumerator before taking the log, we get the <b>discriminant function</b> as:

$$\delta_k(X) = log(\pi_k) - log(\sigma_k) - \frac{\mu_k^2}{2\sigma_k^2} + x.\frac{\mu_k}{\sigma_k^2} - \frac{x^2}{2\sigma_k^2}$$

And hence Bayes' classifier is not linear in $x$. In fact, it is quadratic.

Q4. When the number of features p is large, there tends to be a deterioration in the performance of KNN and other local approaches that perform prediction using only observations that are near the test observation for which a prediction must be made. This phenomenon is known as the <b>curse of dimensionality</b>, and it ties into the fact that parametric approaches often perform poorly when p is large.

(a) Suppose that we have a set of observations, each with measurements on p = 1 feature, X. We assume that X is uniformly distributed on [0, 1]. Associated with each observation is a response value. Suppose that we wish to predict a test observation’s response using only observations that are within 10% of the range of X closest to that test observation. For instance, in order to predict the response for a test observation with X = 0.6, we will use observations in the range [0.55, 0.65]. On average, what fraction of the available observations will we use to make
the prediction?

<b>Sol:</b> As for each test observation, we use the observations in the range <b>0.1</b> (half left and half right) of the entire range of observation, and hence this accounts to the average value of <b>10%</b> of the available observation.

To be more specific, for the test observation, x < 0.05, we will use the observations in the range [0, x+0.05] and for x > 0.95, we will use the observations in the range [x-0.05, 1]. As the observations are uniformly distributed, we can find the average fraction of observations used by integrating the individual ranges:

$$fraction = \int_{0}^{0.05}(100x+5) dx + \int_{0.05}^{0.95} 10 dx + \int_{0.95}^{1} (105-100x)dx = 9.75$$

And hence to be more specific, we use <b>9.75%</b> of the observations.

(b) Solve the above case for 2 predictors $X_1$ and $X_2$.

<b>Sol:</b> For the two independent predictors, the fraction of observations used will be 9.75% for $X_1$ and 9.75% for $X_2$. Hence the total fraction of observations used is 9.75% $\times$ 9.75% = 0.95%.

(c) Now suppose that we have a set of observations on p = 100 features. What fraction of the available observations will we use to make the prediction?

<b>Sol:</b> For p=100, we will use (9.75%)$^{100}$ = 0% of the total observations for the prediction.

(e) Now suppose that we wish to make a prediction for a test observation by creating a p-dimensional hypercube centered around the test observation that contains, on average, 10% of the training observations. For p = 1, 2, and 100, what is the length of each side of the hypercube? Comment on your answer.

<b>Sol:</b> For $p=l$, the length of the hypercube is $(0.1)^{\frac{1}{l}}$. Hence, as $l$ increases, the lenghth of the cube increases.

Q5. (a) If the Bayes decision boundary is linear, do we expect LDA or QDA to perform better on the training set? On the test set?

<b>Sol:</b> QDA may perform better on training set as it has higher flexibility. It may try to imitate training data as close as possible and hence may result in overfitting. LDA will perform better on test set.

(b) If the Bayes decision boundary is non-linear, do we expect LDA or QDA to perform better on the training set? On the test set?

<b>Sol:</b> QDA will perform better both on the training as well as test sets.

(c) In general, as the sample size n increases, do we expect the test prediction accuracy of QDA relative to LDA to improve, decline, or be unchanged? Why?

<b>Sol:</b> As in QDA, we assume different variance among classes. If sample size is small, QDA performs better. As sample size increases, the assumption of equal variance among classes (as assumed while doing LDA) may be true and hence the performance of LDA and QDA in this case solely depends on the Bayes' decision boundary.

(d) True or False: Even if the Bayes decision boundary for a given problem is linear, we will probably achieve a superior test error rate using QDA rather than LDA because QDA is flexible enough to model a linear decision boundary.

<b>Sol:</b> False

Q6. Suppose we collect data for a group of students in a statistics class with variables X1 =hours studied, X2 =undergrad GPA, and Y = receive an A. We fit a logistic regression and produce estimated coefficient, $\widehat{\beta_0}= −6$ , $\widehat{\beta_1} = 0.05$, $\widehat{\beta_2}= 1$.

(a) Estimate the probability that a student who studies for 40h and has an undergrad GPA of 3.5 gets an A in the class.

<b>Sol:</b> The probability in logistic regression is given as:

$$p(X) = \frac{e^{\beta_0 + \beta_1X_1 + ... + \beta_pX_p}}{1 + e^{\beta_0 + \beta_1X_1 + ... + \beta_pX_p}}$$

Hence, the probability of a student getting A with the given data is

$$p = \frac{e^{-6 + 0.05\times40 + 1\times 3.5}}{1 + e^{-6 + 0.05\times40 + 1\times 3.5}} = 0.3775$$

(b) How many hours would the student in part (a) need to study to have a 50% chance of getting an A in the class?

<b>Sol:</b> Here probability is 0.5. Hence,

$$e^{-6 + 0.05\times h + 1\times 3.5} = \frac{p}{1-p} = 1$$

$$-2.5 + 0.05\times h = 0$$

and hence value of h is <b>50</b>.

Q7. Suppose that we wish to predict whether a given stock will issue a dividend this year (“Yes” or “No”) based on X, last year’s percent profit.We examine a large number of companies and discover that the mean value of X for companies that issued a dividend was $\bar{X}$ = 10, while the mean for those that didn’t was $\bar{X}$ = 0. In addition, the
variance of X for these two sets of companies was $\widehat{\sigma^2}$ = 36. Finally, 80% of companies issued dividends. Assuming that X follows a normal distribution, predict the probability that a company will issue a dividend this year given that its percentage profit was X = 4 last year.

<b>Sol:</b> From Bayes' theorem:

$$p_k(X) = \frac{\pi_k \frac{1}{\sqrt{2\pi}\sigma} exp \bigg( {-\frac{1}{2\sigma^2}(x - \mu_k)^2} \bigg)}
{\sum_{l=1}^{k}\pi_l\frac{1}{\sqrt{2\pi}\sigma} exp \bigg( {-\frac{1}{2\sigma^2}(x - \mu_l)^2} \bigg)}$$

Here, $\pi_{YES} = 0.8, \pi_{NO} = 0.2, \mu_{YES} = 10, \mu_{N0} = 0, \widehat{\sigma^2} = 36 $. Hence, $f_{YES}(4) = 0.04033, \ f_{NO}(4) = 0.05324$. Which gives the probability as:

$$p_YES{4} = \frac{0.8 \times 0.04033}{0.8 \times 0.04033 + 0.2 \times 0.05324} = 0.75186$$

Q8. Suppose that we take a data set, divide it into equally-sized training and test sets, and then try out two different classification procedures. First we use logistic regression and get an error rate of 20% on the training data and 30% on the test data. Next we use 1-nearest neighbors (i.e. K = 1) and get an average error rate (averaged over both test and training data sets) of 18%. Based on these results, which method should we prefer to use for classification of new observations? Why?

<b>Sol:</b> For KNN with K=1, the training errro rate is 0 as all the training data will be classified correctly. Hence for the KNN with average error rate of 18%, the test error rate is 36% which makes the logistic regression with test error rate 30% is better for classification.

Q9. This problem has to do with odds.

(a) On average, what fraction of people with an odds of 0.37 of defaulting on their credit card payment will in fact default?

<b>Sol:</b> Odds is given as:

$$\frac{p}{1-p} = 0.37$$

which gives the probability as <b>0.27</b>.

(b) Suppose that an individual has a 16% chance of defaulting on her credit card payment. What are the odds that she will default?

<b>Sol:</b> Here, p = 0.16. Hence odds of defaulting is

$$odds = \frac{p}{1-p} = \frac{0.16}{1-0.16} = 0.19$$
