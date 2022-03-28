+++
date = "2018-10-29T12:11:37+01:00"
description = "Performance Metrics for Classification Algorithms"
draft = false
tags = ["Classification", "Precision", "Recall", "Specificity", "Sensitivity", "F1 Score", "ROC Curve"]
title = "Performance Metrics for Classification Algorithms"
topics = ["Performance Metrics for Classification Algorithms"]

+++



There are several metrics that can be used to measure the performance of a classification algorithm. The choice for the same depends on the problem statement and serves an important role in model selection.

</br>
### Confusion Matrix :

<b>Confusion matrix</b> is one of the easiest and the most intutive way to find the correctness and accuracy of the model. It serves as the building block for all the other performance measures. A sample confusion matrix is shown below:

{{% fluid_img "/img/performance_metric/cm.png" %}}

<b>True Positive (TP)</b> is the cell which has both the actual and predicted classes as <b>True</b>. <b>False Positive (FP)</b> is the cell for which the actual class label is <b>False</b> but the predicted class label is <b>True</b>. <b>True Negative (TN)</b> is the cell for which both the actual and predicted class labels are <b>False</b>. <b>False Negative (FN)</b> is the cell for which the actual class label is <b>True</b> but the predicted class label is <b>False</b>. In the above confusion matrix, <b>TP = 100, FP = 10, TN = 50, FN = 5</b>.

All these quantities have different and important effects when it comes to the problem statement. For example, in the case of a model which detects the cancerous cell, we need to minimize the False Negative as miss classifying a cancerous tumor as non-cancerous will have a significant impact. In the case of the model which identifies spam emails, we need to minimize False Positive rate, as they will lead to the classification of important emails as spams.

</br>
### Accuracy :

Accuracy of a classification model is given as:

$$Accuracy = \frac{TP+TN}{Number \ of \ Observations} = \frac{TP+TN}{TP+FP+TN+FN}$$

Accuracy is a good measure for the fit when the target class labels are balanced. We should refrain ourselves from using accuracy as the measure when the target class labels has a majority of one class.

</br>
### Precision and Recall (Sensitivity) :

<b>Precision</b> is the measure which tells us that out of all the observations that are predicted as true, what fraction is actually true. It is given as:

$$Precision = \frac{TP}{TP+FP}$$

<b>Recall</b> is a measure that tells us that out of all the actual true class labels, how many are correctly classified by the model. It is given as:

$$Recall = \frac{TP}{TP+FN}$$

If we want to minimize FP, we should be maximizing precision and if we want to minimize FN, we should be maximizing recall. Hence, in the case of cancerous cell detection, we have to minimize FN and hence we should maximize recall. For the model that identifies spam emails, our goal is to minimize FP and hence we should maximize precision.

</br>
### Specificity and F1 Score :

<b>Specificity</b> measures the classification accuracy for the <b>true negative</b> observations in the data set. It is given as:

$$Specificity = \frac{TN}{TN+FP}$$

<b>F1 Score</b> is the <b>harmonic mean</b> of precision and recall.

$$F1 \ Score = \frac{2 \times Precision \times Recall}{Precision + Recall}$$

</br>
### AUC-ROC Curve :

<b>AUC(Area under the Curve)-ROC(Receiver Operating Characteristics) Curve</b> is one of the most important measure of the classification performance. ROC curve is plotted as <b>true positive rate (TPR)</b> against <b>false positive rate (FPR)</b>. TPR is the classification accuracy rate for the true positive observations, which can be given as $\frac{TP}{TP+FN} = Recall$. FPR is the false positive rate and tells us about the fraction of true negative observations which are falsely classified as positive and is given as $\frac{FP}{FP+TN} = 1 - Specificity$. An ideal or favorable ROC curve should be hugging the top left corner and should have an area closed to 1 under it. The ROC curve with an area of 0.5 tells that the classifier has no classification power as such.

</br>
### Reference :

https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/

https://medium.com/greyatom/performance-metrics-for-classification-problems-in-machine-learning-part-i-b085d432082b

https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5
