+++
date = "2018-05-17T05:22:19+01:00"
description = "ISLR Resampling Methods"
draft = false
tags = ["ISLR", "Resampling", "Cross-Validation"]
title = "ISLR Chapter 5: Resampling Methods (Part 1: Cross-Validation)"
topics = ["ISLR"]

+++

<h1><center>Resampling Methods</center></h1>

<b>Resampling methods</b> involve repeatedly drawing samples from training data and refitting a model on them. Resampling approaches can be computationally expensive. <b>Cross-validation</b> and <b>bootstrap</b> are two of the most commonly used resampling methods. Cross-validation can be used to estimate the test error rate associated with a given model in order to evaluate its performance. The process of evaluating a model's performance is called <b>model assessment</b>. The process of selecting proper level of flexibility for a model is known as <b>model selection</b>.

### 5.1 Cross-Validation

<b>Test error rate</b> can be easily calculated if a designated test set is available. This is usually not the case. One approach is to estimate the test error rate by <b>holding out</b> a subset of the training observation from the fitting process and then apply the model to the held out observations.

#### 5.1.1 The Validation Set Approach

The validation set approach involves dividing the available set of observations into two part: a <b>training set</b> and a <b>validation set</b>. The model is fit on the training set and the fitted model is used to compute the validation set error rate. Validation set approach is simple and easy to implement but it has two drawbacks:

 - Validation set error rate can be highly variable, depending on which observations are included in the training set and which in the validation set.


 - As only a subset of the observations is used to fit the model, the modle tends to perform worse. This suggests that the validation set error rate may tend to <b>overestimate</b> the test error rate for the model fitted on the entire data set.

#### 5.1.2 Leave-One-Out Cross-Validation

<b>Leave-one-out cross-validation (LOOCV)</b> is closely related to the validation set approach. LOOCV also involves splitting the data set into two parts. Instead of creating two subsets of comaparable size, a <b>single observation</b> is used for the validation set and the remaining observations are used for the trainig set. The <b>MSE</b> of a single observation of the validation set can provide an unbiased estimate for the test error but it is <b>highly variable</b>.

The LOOCV process can be repeated $n$ times, producing $n$ different MSEs, when each time validation set consists of $i$th observation, where value of $i$ ranges from $1$ to $n$. The MSE for validation sets is given as:

$$MSE_{i} = (y_i - \widehat{y_i})^2$$

and the overall estimate of <b>test MSE</b> is the average of these $n$ validation errors:

$$CV_{(n)} = \frac{1}{n} \sum _{i=1}^{n} MSE _{i}$$

LOOCV has far <b>less bias</b> when compared to validation set approach. It does not tend to overestimate the test error rate as much as the validation set approach does. Performing LOOCV multiple times will always yield the same results as there is no randomness in the training/validation set split. LOOCV can be expensive to implement as the model has to be fitted n-times. LOOCV is a very generic approach and can be used with any kind of predictive modeling.

#### 5.1.3 k-Fold Cross-Validation

<b>k-fold CV</b> is an alternative to LOOCV. In this aproach the observations are randomly divided into $k$ groups of approximately equal size. The first fold is treated as a validation set and the method is fit on the remaining $k-1$ folds. The $MSE_{1}$ is computed on the observations in the held-out fold. The process is repeated $k$ times. Each time a different group of observation is treated as the validation set. The $k$-fold CV estimate is given as:

$$CV_{(k)} = \frac{1}{k} \sum _{i=1}^{k} MSE _{i}$$

LOOCV is a special case of $k$-fold CV with $k=n$. Generally, a typical value of $K=5$ or $k=10$ is chosen. $k$-fold CV has an advantage of less computational complexity as model needs to be fitted $k$ times only. There may arise some variability in the CV estimates as there is some variability in the divison of the observations into the folds.

Sometimes instead of arriving on a correct estimate of test MSE, we are more interested in selecting the model for which the test MSE will be minimum. Hence, for this purpose, the location of the minimum point in the estimated test MSE curve is more important, which can help in deciding the correct flecibility of the model. $k$-fold CV does a pretty decent job in this.

#### 5.1.4 Bias-Variance Trade-Off for k-Fold Cross-Validation

Apart from computational advantage, $k$-fold CV gives more accurate estimates of the test error rate than LOOCV. Since LOOCV uses $n-1$ observations as training set, it will give approximately unbiased (low bias) estimates of the test error. Compared to validation set approach, $k$-fold CV has low bias as well. But from the prespective of <b>bias reduction</b>, LOOCV is to be preferred over $k$-fold CV.

LOOCV has <b>higher variance</b> compared to $k$-fold CV for $k < n$. As we know that, <b>mean of highly correlated quantities has higher variance compared to the mean of quantities that are not highly correlated.</b> For the LOOCV, as the different models share higher amount of data compared to $k$-fold CV, they are somewhat more correlated. Hence, the test error estimate of LOOCV has <b>higher variance</b> compared to $k$-fold CV. $k$-fold CV with $k=5$ or $k=10$, yields test error estimates that suffer neither from excessively high bias, nor from very high variance.

<b>Mean of highly correlated quantities have higher variance</b> as:

 - When the elements of each sample are positively correlated, when one value is high the others tend to be high, too. Their mean will then be high. When one value is low the others tend to be low, too. Their mean will then be low. Thus, the means tend either to be high or low.


 - When elements of each sample are not correlated, the amount by which some elements are high is often balanced (or "canceled out") by other low elements. Overall the mean tends to be very close to the average of the population from which the samples are drawn--and rarely much greater or much less than that.

#### 5.1.5 Cross-Validation on Classification Problems

In the case of classification, cross-validation works in the same way. However, instead of using MSE to quantify the test error, we can use the number of misclassified observations. For example, in classification setting, the LOOCV error rate takes the form:

$$CV_{n} = \frac{1}{n} \sum _{i=1}^{n} Err_i$$

where $Err_i = I(y_i \neq \widehat{y_i})$. $k$-fold CV error rate and validation set error rate are defined similarly.

In general <b>10-fold CV</b> error rate provides a good approximation to the test error rate (though it somewhat underestimates it). It reaches a minimum value for the correct flexibiliy of model.
