+++
date = "2018-06-13T13:04:28+01:00"
description = "ISLR Tree-Based Methods"
draft = false
tags = ["ISLR", "Tree-Based Methods", "Bagging", "Random Forests", "Boosting"]
title = "ISLR Chapter 8: Tree-Based Methods (Part 2: Bagging, Random Forests, Boosting)"
topics = ["ISLR"]

+++

### 8.2 Bagging, Random Forests, Boosting

#### 8.2.1 Bagging

The decision trees discussed above suffers from a problem of <b>high variance</b>. <b>Bootstrap aggregation</b> or <b>bagging</b> is a procedure that reduces the variance of a statistical learning method.

Give a set of $n$ independent observation sets $Z_1, Z_2, ..., Z_n$, each with variance $\sigma^2$, the variance of the mean $\bar{Z}$ is given by $\sigma^2/n$, i.e. <b>averaging a set of observations reduces variance</b>. Hence, a natural way to reduce the variance of a statistical model is to take many training samples from the population, fit individual models on them, and give the average of them as the final model. Hence, we can calculate $\widehat{f}^1(x), \widehat{f}^2(x), ..., \widehat{f}^B(x)$ using $B$ seperate training sets, and finally obtain the low variance statistical model as:

$$\widehat{f} _{avg}(x) = \frac{1}{B} \sum _{b=1}^{B} \widehat{f}^b(x)$$

As we don't have access to different training sets, the above decribed process is not feasible. Instead, we can bootstrap, taking repeated samples from the same training set (sampling with replacement), and then take the average of the individual models as the final one

$$\widehat{f} _{bag}(x) = \frac{1}{B} \sum _{b=1}^{B} \widehat{f}^{*b}(x)$$

This procedure is called <b>bagging</b>. Bagging can improve the predictions of the decision trees dramatically. To apply bagging to regression trees, we construct $B$ regression trees (<b>not pruned</b>) using $B$ bootstrapped training sets and average them as the final predictions. Each individual tree has high variance and low bias and averaging them reduces the variance. In the case of qualitative response variable, the prediction can be decided by <b>majority vote</b>, the overall prediction is the most common occurring class among the $B$ predictions. The number of trees $B$ is not a critical parameter, as increasing $B$ will not lead to overfitting. A value of $B$ is choosen for which test error stabalizes (almost stops reducing).

##### Out-of-Bag Error Estimation

As it turns out that in the bootstrap procedure, each bagged tree uses only 2/3 of the observations. This can be proved as: The chance of a sample getting not selected is $1-1/n$, where $n$ is total number of samples. Hence, the probability of a sample not being selected in each draw is $(1-1/n)^n$. As $n \to \infty$, this value tends to $1/e$ which is $\frac{1}{3}$. These one-third of the samples which is not used to fit the given bagged tree is called as <b>out-of-bag (OOB)</b> observations.

Hence, the $i$th observation will be OOB in a total of $B/3$ trees. The prediction of the $i$th observation is the average of these $B/3$ predictions of the trees in which it is OOB. An OOB prediction can be obtained for each of the $n$ observations, from which the overall <b>OOB MSE</b> (for regression problem) or classification error rate (for classification problem) can be computed. This OOB error can serve as a valid estimate for the test error.

##### Variable Importance Measures

Bagging improves the prediction accuracy but the interpretation of the resulting model is quite difficult. Hence, bagging improves prediction accuracy at the expense of interpretability. Instead, we can obtain an overall summary of the importance of each of the predictor using the RSS (for regression), or the Gini index (for classification). For the bagging regression trees, we can obtain the overall decrease in RSS averaged over all the $B$ trees due to splits over a given predictor. This can serve as a measure of <b>variable importance</b>.

#### 8.2.2 Random Forests

<b>Random forests</b> provide an improvement over bagged trees by <b>decorrelating</b> the trees. In bagging, for the splits of the bagged tree, a <b>random sample of $m$ predictors</b> out of $p$ is chosen as the split candidates. Typically, $m$ is chosen as $m \approx \sqrt{p}$. Hence, in the case of bagged trees, if there is a strong predictor, there is a chance that most of the bagged trees will use this predictor for the top split. Hence, the resulting bagged trees can be highly correlated. Averaging highly correlated items does not result in higher reduction in variance compared to the averaging of uncorrelated items.

Random forests overcome this problem by forcing each split to consider only a <b>subset of predictors</b>. Hence, on an average $(p-m)/p$ of the trees will not even consider the strong predictor for the split. As a result, the produced trees will be <b>decorrelated</b>, making the average less variable and hence more reliable.

When we have a large number of correlated predictors, using a small value of $m$ will lead to more reliable predictions. Like bagging, random forests will <b>not</b> lead to overfitting if we increase the value of $B$.

#### 8.2.3 Boosting

Like bagging, <b>boosting</b> is a general approach that can be applied to many statistical learning methods for regression and classification. In bagging, each tree is built on a a bootstrap data set, independent of other trees. Boosting works in a similar way, except that the trees are grown <b>sequentially</b> (each tree is grown using the information from previously grown trees). Boosting does not involbe bootstrap sampling, instead, each tree is fit on a modified version of origianl data set. Boosting algorithm for regression trees is as follows:

 - Set $\widehat{f}(x) = 0$ and $r_i = y_i$ for all $i$ in the training set, where $r_i$ is the individual residual.


 - For $b=1,2,..., B$, repeat

     (a) Fit a tree $\widehat{f}^b$ with $d$ splits, i.e. $d+1$ terminal nodes to the new training data $(X, r)$.

     (b) Update $\widehat{f}$ by adding a shrunken version of the new tree:

     $$\widehat{f}(x) = \widehat{f}(x) + \lambda \widehat{f}^b(x)$$

     (c) Update the residual as:

     $$r_i = r_i - \lambda \widehat{f}^b(x_i)$$

 - Output the boosted model:

    $$\widehat{f}(x) = \sum_{b=1}^{B} \lambda \widehat{f}^b(x)$$

In boosting instead of fitting response, we fit the <b>current residuals</b> and then update the current residuals by subtracting the explained part of the residual by the model and repeat the procedure further. Hence, the boosting approach <b>learns slowly</b>. Each of the fitted model (trees) will be small depending on the split parameter $d$. By fitting small trees to the residuals, we slowly improve the model in the area it does not perform well. The shrinkage parameter $\lambda$ slows the process further down.

Boosting has three tuning parameters:

 - The number of trees $B$. Unlike bagging, a larger value of $B$ can result in overfitting. Appropriate value of $B$ is selected by cross-validation.


 - The <b>shrinkage parameter</b> $\lambda$, a small positive number, controls the rate of learning of the boosting algorithm. Typical values are 0.01 or 0.001 and depends on the problem. For a small value of $\lambda$, the value of $B$ needed is large.


 - The number of splits $d$ controls the complexity of the boosted ensemble. $d=1$ works well often, in which case each tree is a <b>stump</b>, consisting of a single split. In this case, the boosted ensemble is like an additive model. $d$ is often called as <b>interaction depth</b> as it controls the interaction order of the boosted model.
