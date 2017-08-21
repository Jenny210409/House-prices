# **House Prices**
*Predict House Prices*

## Table of contents

- [Introduction](#introduction)
- [Preparation](#preparation)
- [Prediction](#prediction)
- [Conclusion](#conclusion)


## Introduction
Last time I used RandomForest to predict house prices. I will use XGBOOST this time and check whether there is an improvement.

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

![Alt text](https://github.com/ur4me/House-prices/blob/master/Influentail%20Obs%20by%20Cooks%20distance(train).png)


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

![Alt text](https://github.com/ur4me/House-prices/blob/master/Influential%20Obs%20by%20Cooks%20distance(test).png)


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

#### I will predict the test data using XGBOOST
```
#remove Id and SalePrice columns
train1 <- train[, -c(1,81)]
test1 <- test[, -c(1,81)]
```
```
#set up to use XGBOOST
train1[] <- lapply(train1, as.numeric)
test1[]<-lapply(test1, as.numeric)

dtrain=xgb.DMatrix(as.matrix(train1),label= train$SalePrice)
dtest=xgb.DMatrix(as.matrix(test1))
```
```
#xgboost parameters
xgb_params = list(
  seed = 0,
  colsample_bytree = 0.5,
  subsample = 0.8,
  eta = 0.02, 
  objective = 'reg:linear',
  max_depth = 12,
  alpha = 1,
  gamma = 2,
  min_child_weight = 1,
  base_score = 7.76
)

xg_eval_mae <- function (yhat, dtrain) {
  y = getinfo(dtrain, "label")
  err= mae(exp(y),exp(yhat) )
  return (list(metric = "error", value = err))
}

best_n_rounds=150 # try more rounds
```

#### Predict and save
```
#train data
gb_dt=xgb.train(xgb_params,dtrain,nrounds = as.integer(best_n_rounds))
prediction <- predict(gb_dt,dtest)
solution <- data.frame(Id = test$Id, SalePrice = prediction)
write.csv(solution, file = 'xgb_Sol2.csv', row.names = F)
```

## Conclusion
My public score(RMSE) for Random Forest model was 0.14739 whereas 0.14654 for this time. Accordingly, it increased slightly and it seems like I can get better score if I amend the procedure and use ensemble learning method.


