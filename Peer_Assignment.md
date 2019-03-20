---
title: "Peer Assignment Practical Machine Learning"
output: 
  html_document: 
    keep_md: yes
---
                                        
                                        
                                           K. van Splunter
                  
                  
                                        
## Introduction

This project focuses on machine learning. This document will build a prediction-model and test this model.  
The data that is being used in this project has data of 6 participants about their performance of barbell lifts. They were asked to do it correctly and incorrectly in 4 different ways.
The data comes from the following source: http://groupware.les.inf.puc-rio.br/har

## Data

First, the data is loaded into R and the necessary packages are loaded. The seed is set to ensure that it's reproducible.

```r
trainingData <- read.csv("Trainingdata.csv", na.strings = c("NA", ""))
testData <- read.csv("Testdata.csv", na.strings = c("NA", ""))
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(randomForest)
```

```
## randomForest 4.6-14
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
set.seed(938240)
```

Now, the training data is split into a 'new' training set and a set for validation. The decision is made to include 70% into the 'new' training set. 

```r
inTrain <- createDataPartition(y = trainingData$classe, p=0.70,list=F)
training <- trainingData[inTrain,]
validation <- trainingData[-inTrain,]
```

It's important to look at the data. It seems that there are quite a few variables with (many) missing values (For practical concerns the results of the overview are NOT shown here).

```r
head(training)
```

With so many variables that have missing values, there are bound to be problems with the creation of the model. So, the decision is made to get rid of those variables of which more that 65% of the data is missing.

```r
Keepvars <- c((colSums(!is.na(training[,-ncol(training)])) > 0.65*nrow(training)))
training <- training[,Keepvars]
validation <- validation[,Keepvars]
dim(training)
```

```
## [1] 13737    60
```

This leaves 60 variables. However, There are a few more variables, such as *X* and *user_name*, that have no added value. These need to be removed.

```r
uselessVars <- c("X","user_name","raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp",
                 "num_window", "new_window")
training <- training[,!(names(training) %in% uselessVars)]
validation <- validation[,!(names(validation) %in% uselessVars)]
dim(training)
```

```
## [1] 13737    53
```

Now, there are 53 variables left, including the the variable that our model is going to predict, *classe*.  

## Building the model

Now, the model is being build. The method of choice is **random forest**. This method has several benefits. First of all, it is used quite a lot and is a accurate method for prediction. Furthermore, it takes care of cross validation, so ther is no need to create extra subsamples of the training data. However, to provide a more certain overview of the out-of-sample error, the model will first, be run over the validation set.  
A negative aspect of random forest is that it might not be fastest method. Besides speed, there is also the issue that the model is a bit less interpretable.  
This part of the code builds the model in the training set.

```r
modFit <- randomForest(classe ~., data = training)
modFit
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = training) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.55%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3904    2    0    0    0 0.0005120328
## B   15 2638    5    0    0 0.0075244545
## C    0   13 2378    5    0 0.0075125209
## D    0    0   23 2226    3 0.0115452931
## E    0    0    3    7 2515 0.0039603960
```

The estimate of the Out-of-Bag error that the procedure calculates internally is 0.59%.

## Testing the model

The next step is to run the model on the validation set. This ensures that the estimation of the out-of-sample-error is more in line with the actual error.


```r
predVal <- predict(modFit, newdata = validation)
confusionMatrix(predVal, validation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    8    0    0    0
##          B    0 1126    7    0    0
##          C    0    5 1019    9    0
##          D    0    0    0  955    0
##          E    0    0    0    0 1082
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9951          
##                  95% CI : (0.9929, 0.9967)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9938          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9886   0.9932   0.9907   1.0000
## Specificity            0.9981   0.9985   0.9971   1.0000   1.0000
## Pos Pred Value         0.9952   0.9938   0.9864   1.0000   1.0000
## Neg Pred Value         1.0000   0.9973   0.9986   0.9982   1.0000
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1913   0.1732   0.1623   0.1839
## Detection Prevalence   0.2858   0.1925   0.1755   0.1623   0.1839
## Balanced Accuracy      0.9991   0.9936   0.9951   0.9953   1.0000
```

On the validation set, the accuracy of the model is 0.9947, suggesting that the out-of-sample error is around the 0.53%.

## Conclusion

This random forest model used 52 variables to predict the *classe*. The accuracy of the model on the validation set was 99.47% with an out-of-sample error of 0.53%.
