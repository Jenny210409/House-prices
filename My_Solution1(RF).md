# **House Prices**
*Predict House Prices*

## Table of contents

- [Introduction](#introduction)
- [Preparation](#preparation)
- [Prediction](#prediction)


## Introduction
I will use RandomForest to predict house prices. As there are too many Nas, I will use Mice to replace missing values.

## Preparation
#### Initial works
```
library(plyr)
library(dplyr)
library(mice)
library(xgboost)
library(randomForest)
library(hydroGOF)
```
```
setwd('c:/kaggle/house prices')
#retrieve train and test
train <- read.csv('train.csv', na.strings = c("", "NA"), stringsAsFactors = F)
test <- read.csv('test.csv', na.strings = c("", "NA"), stringsAsFactors = F)
#combine train and test
total <- bind_rows(train, test)
total <- as.data.frame(unclass(total))
#check duplicate
nrow(train) - nrow(unique(train))
```
Compare to the Titanic project, it seems like I don't need to separate certain column or add meaningful column.
#### Replace missing values
```
miceMod <- mice(total, method="rf")
miceOutput <- complete(miceMod)
#Check whether there is missing values
colnames(miceOutput)[colSums(is.na(miceOutput)) > 0]
```



#### Separating back
```

#change to numeric
indx <- sapply(miceOutput, is.factor)
miceOutput[indx] <- lapply(miceOutput[indx], function(x) as.numeric(as.factor(x)))
#separate
train <- miceOutput[1:1460,]
test <- miceOutput[1461:2919,]
```

#### Outlier handling
This could be one of the most important and time consuming process. I will use Multivariate Model Approach, Cooks Distance, to find out outliers.

```
mod <- lm(SalePrice ~ ., data=train)
cooksd <- cooks.distance(mod)
plot(cooksd, pch="*", cex=2, main="Influential Obs by Cooks distance")  # plot cook's distance
abline(h = 4*mean(cooksd, na.rm=T), col="red")  # add cutoff line
text(x=1:length(cooksd)+1, y=cooksd, labels=ifelse(cooksd>4*mean(cooksd, na.rm=T),names(cooksd),""), col="red")  # add labels
```
![Alt text](/path/to/img.jpg)
It shows that there are 4 outstanding outliers.
I will remove those 4 rows.
```
train <- train[-c(524,692,1183,1299),]
```

How about test data? Let's check the outliers in test data.
```
mod <- lm(SalePrice ~ ., data=test)
cooksd <- cooks.distance(mod)
plot(cooksd, pch="*", cex=2, main="Influential Obs by Cooks distance")  # plot cook's distance
abline(h = 4*mean(cooksd, na.rm=T), col="red")  # add cutoff line
text(x=1:length(cooksd)+1, y=cooksd, labels=ifelse(cooksd>4*mean(cooksd, na.rm=T),names(cooksd),""), col="red")  # add labels
```

It shows that row 2550 is the most outstanding outlier. As we need to predict the house price for this row, I will not remove the row; however, I will check which numbers made this row as an outlier.

```
test[test$Id == 2550,]
summary(test)
```
By comparing those 2 results, it seems like we need to replace values in 3 columns (BsmtFinSF1, TotalBsmtSF, X1stFlrSF)
```
#replace outliers
model <- randomForest(BsmtFinSF1 ~ ., data=test)
prediction <- predict(model, test1[test1$Id == 2550,])
test1[test1$Id == 2550,"BsmtFinSF1"] <- prediction

model <- randomForest(TotalBsmtSF ~ ., data=test)
prediction <- predict(model, test1[test1$Id == 2550,])
test1[test1$Id == 2550,"TotalBsmtSF"] <- prediction

model <- randomForest(X1stFlrSF ~ ., data=test)
prediction <- predict(model, test1[test1$Id == 2550,])
test1[test1$Id == 2550,"X1stFlrSF"] <- prediction
```
## Prediction

#### I will predict the test data using Random Forest model
```
#predict with Random Forest
model_1 <- randomForest(SalePrice ~ ., data=train2)
prediction <- predict(model_1, test)
solution <- data.frame(Id = test$Id, SalePrice = prediction)
write.csv(solution, file = 'random_forest_Sol.csv', row.names = F)
```


