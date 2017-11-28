# **House Prices**
*Predict House Prices in IOWA*

## Table of contents

- [Introduction](#introduction)
- [Preparation](#preparation)
- [Prediction](#prediction)
- [Conclusion](#conclusion)


## Introduction
I will do data cleanings and use different types of models. I will check which model performs best.

## Preparation
#### Initial works
```
library(plyr)
library(dplyr)
library(mice)
library(xgboost)
library(caret)
library(data.table)
library(Boruta)
library(pROC)
library(hydroGOF)
library(tibble)
library(corrplot)
```
```
setwd('c:/kaggle/house prices')
#retrieve train and test
train <- read.csv('train.csv', na.strings = c("", "NA"), stringsAsFactors = F)
test <- read.csv('test.csv', na.strings = c("", "NA"), stringsAsFactors = F)
train1 <- train
test1 <- test
```

### Boruta Feature Importance Analysis
As there are more than 80 variables, I will use Boruta package and remove some variables which seem to be not important. 

```
# preparation
ROOT.DIR <- ".."

ID.VAR <- "Id"
TARGET.VAR <- "SalePrice"

# extract only candidate feture names
candidate.features <- setdiff(names(train),c(ID.VAR,TARGET.VAR))
data.type <- sapply(candidate.features,function(x){class(train[[x]])})


# deterimine data types
explanatory.attributes <- setdiff(names(train),c(ID.VAR,TARGET.VAR))
data.classes <- sapply(explanatory.attributes,function(x){class(train[[x]])})

# categorize data types in the data set?
unique.classes <- unique(data.classes)

attr.data.types <- lapply(unique.classes,function(x){names(data.classes[data.classes==x])})
names(attr.data.types) <- unique.classes


# pull out the response variable
response <- train$SalePrice

# remove identifier and response variables
train <- train[candidate.features]

# for numeric set missing values to -1 for purposes of the random forest run
for (x in attr.data.types$integer){
  train[[x]][is.na(train[[x]])] <- -1
}

for (x in attr.data.types$character){
  train[[x]][is.na(train[[x]])] <- "*MISSING*"}



set.seed(13)
bor.results <- Boruta(train,response,
                      maxRuns=101,
                      doTrace=0)



getSelectedAttributes(bor.results)
plot(bor.results)
```

![Alt text](https://github.com/ur4me/House-prices/blob/master/Boruta%20importance.png)

```
options(width=125)
arrange(cbind(attr=rownames(attStats(bor.results)), attStats(bor.results)),desc(medianImp))
```
       
```            

            attr      meanImp   medianImp      minImp     maxImp normHits  decision
1      GrLivArea 21.213486312 21.18182159 18.56320442 23.3509629     1.00 Confirmed
2    OverallQual 17.419759567 17.51118974 15.18490247 20.1088880     1.00 Confirmed
3      X2ndFlrSF 15.324977047 15.23999674 12.27627567 17.7623584     1.00 Confirmed
4    TotalBsmtSF 14.738306523 14.82359498 12.55157147 16.7271743     1.00 Confirmed
5      X1stFlrSF 14.559316009 14.51487167 11.79769607 16.4087207     1.00 Confirmed
6     GarageArea 13.613082342 13.59108331 11.30643784 15.5103477     1.00 Confirmed
7     GarageCars 13.182238675 13.11951025 11.20162834 15.1819502     1.00 Confirmed
8      YearBuilt 12.709941381 12.89255957  7.61285659 15.0176206     1.00 Confirmed
9      ExterQual 12.269345019 12.29364751 10.01314829 14.0301997     1.00 Confirmed
10  YearRemodAdd 11.243848720 11.21827443  8.99236628 13.2948661     1.00 Confirmed
11   GarageYrBlt 10.928420645 10.98194115  8.79802895 12.7022809     1.00 Confirmed
12       LotArea 10.995693416 10.94215722  7.55161056 13.8523053     1.00 Confirmed
13   FireplaceQu 10.824240446 10.82149755  7.02055955 13.2138912     1.00 Confirmed
14      FullBath 10.546645906 10.50958298  8.55738430 12.5726165     1.00 Confirmed
15   KitchenQual 10.535291917 10.49909280  8.92550132 12.3515693     1.00 Confirmed
16    MSSubClass 10.097045262 10.22609550  8.12709086 12.1927953     1.00 Confirmed
17    Fireplaces  9.759653455 10.09332439  3.00238928 12.1028993     1.00 Confirmed
18    BsmtFinSF1  9.963425918  9.92758131  6.61758776 12.9345391     1.00 Confirmed
19      MSZoning  9.439840434  9.46883429  6.25434083 11.9951855     1.00 Confirmed
20      BsmtQual  8.939732863  9.06866230  6.49689324 10.4152676     1.00 Confirmed
21    GarageType  8.823674427  8.86952594  6.62853320 10.3938804     1.00 Confirmed
22  TotRmsAbvGrd  8.915987581  8.84115514  5.84285142 11.3705574     1.00 Confirmed
23  Neighborhood  8.639393284  8.67665151  7.11039986 10.3734061     1.00 Confirmed
24      HalfBath  7.888679757  7.92368353  5.61241416  9.2976799     1.00 Confirmed
25  GarageFinish  7.501356729  7.47969538  5.07694058  9.6012529     1.00 Confirmed
26    Foundation  7.364588431  7.44552121  5.72620729  8.7488225     1.00 Confirmed
27  BedroomAbvGr  7.278330274  7.32161736  3.87871181  9.0392459     1.00 Confirmed
28      BldgType  7.221605923  7.29966937  4.85680858  9.5605227     1.00 Confirmed
29    HouseStyle  6.859039482  6.81210671  5.26826144  8.7957280     1.00 Confirmed
30   OpenPorchSF  6.717736192  6.58636248  4.77400928  9.0395079     1.00 Confirmed
31    CentralAir  6.420932998  6.51242868  3.52635978  8.2710064     1.00 Confirmed
32     HeatingQC  6.372955041  6.41470052  4.92838354  7.7262153     1.00 Confirmed
33     BsmtUnfSF  6.297346607  6.16995794  3.96846583  8.5832424     1.00 Confirmed
34    MasVnrArea  5.868179263  5.89423793  3.28129502  7.9982495     1.00 Confirmed
35  BsmtFinType1  5.890363023  5.81832629  3.64806622  9.1769751     1.00 Confirmed
36    GarageCond  5.564450286  5.75625868  1.77529799  7.4727605     0.96 Confirmed
37   OverallCond  5.644335308  5.46359827  2.84746706  9.7622152     1.00 Confirmed
38    GarageQual  5.238924431  5.33243648  3.18374982  7.0379745     1.00 Confirmed
39  KitchenAbvGr  5.071142411  5.05882361  2.95307018  6.4455318     1.00 Confirmed
40      BsmtCond  4.632671207  4.64313491  2.44788938  7.2984430     0.98 Confirmed
41  BsmtFullBath  4.658869646  4.47270150  2.49124508  7.5697534     0.98 Confirmed
42   Exterior1st  4.326084785  4.28208986  1.93145637  6.4076248     0.98 Confirmed
43   Exterior2nd  4.103504584  4.23115645  1.33805696  6.3534962     0.91 Confirmed
44    WoodDeckSF  4.155213817  4.23099100  2.09465335  6.1137947     0.95 Confirmed
45    PavedDrive  4.113872738  4.11483213  2.12931370  7.0399183     0.98 Confirmed
46   LandContour  3.326170303  3.46210659  1.09432450  5.5143139     0.83 Confirmed
47    Functional  3.250540884  3.20824134  1.05781565  5.1482064     0.79 Confirmed
48      LotShape  3.135409745  3.09764115 -0.35223412  5.3703551     0.76 Confirmed
49  BsmtExposure  2.933637904  2.99051143  0.04308952  5.8413670     0.66 Tentative
50  BsmtFinType2  2.891174635  2.91536546  0.51735035  4.8991001     0.72 Confirmed
51    MasVnrType  2.900260329  2.89506382  0.25907410  4.6572811     0.70 Confirmed
52         Fence  2.796742141  2.80074690  1.24765864  4.6020268     0.75 Confirmed
53    Electrical  2.677566615  2.67721683  1.18304064  4.0278317     0.59 Tentative
54 SaleCondition  2.632313454  2.65778339  0.24132008  4.2145169     0.57 Tentative
55     RoofStyle  2.523254387  2.57462804 -0.09461237  5.2032537     0.51 Tentative
56         Alley  2.525096284  2.52907005  1.05320246  3.9928812     0.60 Tentative
57     LandSlope  2.324912666  2.33883289  0.02039976  4.6565101     0.46 Tentative
58    Condition1  2.286716541  2.32765283  0.42222738  4.1469750     0.49 Tentative
59   LotFrontage  1.917300752  1.95857420 -1.09374851  5.2938467     0.22  Rejected
60 EnclosedPorch  1.948072287  1.87012731 -0.00627533  3.9902482     0.20  Rejected
61    BsmtFinSF2  1.593693751  1.80184226 -1.44548065  3.0443413     0.04  Rejected
62      SaleType  1.264447182  1.76034811 -0.14247053  2.4257111     0.00  Rejected
63     ExterCond  1.290990253  1.48290911 -0.21037564  2.3533618     0.02  Rejected
64   ScreenPorch  1.514862167  1.39277142  0.56604533  2.7948843     0.02  Rejected
65      RoofMatl  1.294204770  1.26587794 -0.62480721  3.0135669     0.01  Rejected
66  BsmtHalfBath  1.229746069  1.09956743  0.16546234  2.4073907     0.00  Rejected
67       Heating  0.597955625  0.58413019 -1.74924266  2.6042708     0.00  Rejected
68        YrSold  0.335667842  0.47757857 -1.06539765  3.1962519     0.01  Rejected
69   MiscFeature  0.345402539  0.45822765 -1.46171039  1.6487265     0.00  Rejected
70       MiscVal  0.155252783  0.27178594 -1.38348293  1.6913963     0.00  Rejected
71     Utilities  0.000000000  0.00000000  0.00000000  0.0000000     0.00  Rejected
72     LotConfig  0.008046634 -0.01174023 -1.49974602  2.2319272     0.00  Rejected
73    X3SsnPorch -0.043726565 -0.09401563 -1.36526640  1.8564044     0.00  Rejected
74        Street -0.172282773 -0.25379056 -1.18381282  1.6959385     0.00  Rejected
75    Condition2 -0.748169011 -0.75970863 -1.55825754  0.6151877     0.00  Rejected
76        MoSold -0.750372544 -0.82194815 -2.16873893  1.0234136     0.00  Rejected
77  LowQualFinSF -0.779259956 -1.12970777 -2.06881059  0.7705054     0.00  Rejected
78      PoolArea -0.701352686 -1.17867338 -2.15356689  1.2478794     0.00  Rejected
79        PoolQC -1.171231817 -1.19712846 -2.22743669 -0.2361224     0.00  Rejected
```

```

CONFIRMED_ATTR <- c("MSSubClass","MSZoning","LotArea","LotShape","LandContour","Neighborhood",
                    "BldgType","HouseStyle","OverallQual","OverallCond","YearBuilt",
                    "YearRemodAdd","Exterior1st","Exterior2nd","MasVnrArea","ExterQual",
                    "Foundation","BsmtQual","BsmtCond","BsmtFinType1","BsmtFinSF1",
                    "BsmtFinType2","BsmtUnfSF","TotalBsmtSF","HeatingQC","CentralAir",
                    "X1stFlrSF","X2ndFlrSF","GrLivArea","BsmtFullBath","FullBath","HalfBath",
                    "BedroomAbvGr","KitchenAbvGr","KitchenQual","TotRmsAbvGrd","Functional",
                    "Fireplaces","FireplaceQu","GarageType","GarageYrBlt","GarageFinish",
                    "GarageCars","GarageArea","GarageQual","GarageCond","PavedDrive","WoodDeckSF",
                    "OpenPorchSF","Fence")
```


#### Combine train and test data sets

```
#combine train and test
total <- bind_rows(train1, test1)
total <- as.data.frame(unclass(total))
#check duplicate
nrow(train) - nrow(unique(train))
#take variables that I got from Boruta package 
total <- total[,CONFIRMED_ATTR]
```


#### Impute missing values using Mice package

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
train1 <- miceOutput[1:1460,]
test1 <- miceOutput[1461:2919,]
#add SalePrice column to train1
SalePrice <- train$SalePrice
train1 <- cbind(train1,SalePrice)
```

#### Correlation Analysis

I would like to see correlation among numeric variables before proceeding to the modeling procedures.

```{r}
numeric_var <- names(train1)[which(sapply(train1, is.numeric))]
corMatrix <- cor(train1[, numeric_var])
corrplot(corMatrix, order = "FPC", method = "color", type = "lower", 
         tl.cex = 0.6, tl.col = rgb(0, 0, 0))
```

![Alt text](https://github.com/ur4me/House-prices/blob/master/Correlation.png)

#### Exploratory analysis

I will use Tableau to visualise our data. I will see the relationship between important variables and sales prices.

![Alt text](https://github.com/ur4me/House-prices/blob/master/Price%20vs%20TotalBsmtSF.PNG)
![Alt text](https://github.com/ur4me/House-prices/blob/master/price%20vs%201stFlrSF.PNG)
![Alt text](https://github.com/ur4me/House-prices/blob/master/price%20vs%20GrLivArea.PNG)
![Alt text](https://github.com/ur4me/House-prices/blob/master/price%20vs%20LotArea.PNG)
![Alt text](https://github.com/ur4me/House-prices/blob/master/price%20vs%20Remodel%20date.PNG)
![Alt text](https://github.com/ur4me/House-prices/blob/master/price%20vs%20fireplaces.PNG)
![Alt text](https://github.com/ur4me/House-prices/blob/master/price%20vs%20garage%20size.PNG)
![Alt text](https://github.com/ur4me/House-prices/blob/master/price%20vs%20garageArea.PNG)
![Alt text](https://github.com/ur4me/House-prices/blob/master/price%20vs%20quality.PNG)
![Alt text](https://github.com/ur4me/House-prices/blob/master/price%20vs%20year%20built.PNG)

#### Outlier handling
I will use Multivariate Model Approach, Cooks Distance, to find out outliers.

```
#outlier handling
mod <- lm(SalePrice ~ ., data=train1)
cooksd <- cooks.distance(mod)
plot(cooksd, pch="*", cex=2, main="Influential Obs by Cooks distance")  # plot cook's distance
abline(h = 4*mean(cooksd, na.rm=T), col="red")  # add cutoff line
text(x=1:length(cooksd)+1, y=cooksd, labels=ifelse(cooksd>4*mean(cooksd, na.rm=T),names(cooksd),""), col="red")  # add labels

which(cooksd >0.2)
```

![Alt text](https://github.com/ur4me/House-prices/blob/master/Cooks%20distance2.png)

It shows that there are 2 outstanding outliers.
I will remove those 2 rows.

```
train1 <- train1[-c(524,1299),]
```





## Prediction

#### Preparation
```
train1[] <- lapply(train1, as.numeric)
test1[]<-lapply(test1, as.numeric)

#convert SalePrice
train1 <- train1 %>% mutate(SalePrice = log(SalePrice + 1))
```
I will separate train set into subtrain and subtest to set a model and test the model. 

```
#split train
set.seed(54321)
outcome <- train1$SalePrice

partition <- createDataPartition(y=outcome,
                                 p=.7,
                                 list=F)
training <- train1[partition,]
testing <- train1[-partition,]
```

#### Using XGBOOST

```
#xgb matrix
withoutRV <- training %>% select(-SalePrice)
dtrain <- xgb.DMatrix(as.matrix(withoutRV),label = training$SalePrice)
withoutRV1 <- testing %>% select(-SalePrice)
dtest <- xgb.DMatrix(as.matrix(withoutRV1))
```
```
#XGBOOST parameter tuning (Grid Search)
train.control <- trainControl(method = "repeatedcv", repeats = 2,number = 3, search = "grid")

tune.grid <- expand.grid(nrounds = c(100,150),
                         max_depth = c(5,6,7),
                         eta = c(0.10, 0.2),
                         gamma = c(0.0, 0.2),
                         colsample_bytree = c(0.5,0.7,1),
                         min_child_weight= c(5,7), 
                         subsample =c(0.5,0.7,1))



caret.cv <-caret::train(SalePrice ~.,
                        data=training,
                        method="xgbTree",
                        metric = "RMSE",
                        tuneGrid=tune.grid,
                        trControl=train.control)
```

The final values used for the model were nrounds = 100, max_depth = 7, eta = 0.1, gamma = 0, colsample_bytree =
  0.5, min_child_weight = 5 and subsample = 0.7.
I will use those parameters to make a model.

```
#xgboost parameters
xgb_params <- list(colsample_bytree = 0.5, #variables per tree 
                   subsample = 0.7, #data subset per tree 
                   booster = "gbtree",
                   max_depth = 7, #tree levels
                   eta = 0.1, #shrinkage
                   eval_metric = "rmse", 
                   objective = "reg:linear",
                   min_child_weight = 5,
                   gamma=0)

#cross-validation and checking iterations
set.seed(4321)
xgb_cv <- xgb.cv(xgb_params,dtrain,early_stopping_rounds = 10, nfold = 4, print_every_n = 5, nrounds=1000, nthread=6)
```
101 was the best iteration.

```
# check the model
gb_dt <- xgb.train(params = xgb_params,
                   data = dtrain,
                   verbose = 1, maximize =F,
                   nrounds = 101, nthread=6)

prediction <- predict(gb_dt,dtest)

#Check RMSE
rmse(testing$SalePrice, prediction)
```
I got 0.1070478 RMSE which looks good.


#### predicting real test data using same parameters
  
```
withoutRV <- train1 %>% select(-SalePrice)


dtest1 <- xgb.DMatrix(as.matrix(test1))



prediction <- predict(gb_dt,dtest1)
```

```
#save the file (Need to use exp and -1 to change it back)
solution <- data.frame(id = test$Id, SalePrice = exp(prediction)-1)

#check negative value just in case
which(solution$SalePrice < 0)

#save
write.csv(solution, file = 'xgb_Sol12.csv', row.names = F)
```

Finally, I will check importance.
```
imp_matrix <- as.tibble(xgb.importance(feature_names = colnames(train1 %>% select(-SalePrice)), model = gb_dt))

imp_matrix %>%
  ggplot(aes(reorder(Feature, Gain, FUN = max), Gain, fill = Feature)) +
  geom_col() +
  coord_flip() +
  theme(legend.position = "none") +
  labs(x = "Features", y = "Importance")
```

![Alt text](https://github.com/ur4me/House-prices/blob/master/importance.png)

My public score(RMSE) is dropped down to 0.13227 which is a great improvement. 

#### Caret Ensemble model
This time I will ensemble 7 models and check whether I can get better RMSE.

```
library("caretEnsemble")
train.control <- trainControl(method = "repeatedcv", repeats = 2,number = 3)
model_list <- caretList(
  SalePrice~., data=training,
  trControl=train.control, metric="RMSE",
  methodList=c("glm", "rf" , "glmboost", "neuralnet", "blackboost", "nnet", "gbm"))
  
greedy_ensemble <- caretEnsemble(
  model_list, 
  metric="RMSE",
  trControl=train.control)
summary(greedy_ensemble)
rmse(testing$SalePrice, prediction)
```
I got 0.1034 RMSE so seems like I can get better score.

```
# real prediction
model_list1 <- caretList(
  SalePrice~., data=train1,
  trControl=train.control, metric="RMSE",
  methodList=c("glm", "rf" , "glmboost", "neuralnet", "blackboost", "nnet", "gbm"))

greedy_ensemble1 <- caretEnsemble(
  model_list1, 
  metric="RMSE",
  trControl=train.control)
summary(greedy_ensemble)

prediction1 <- predict(greedy_ensemble1, test1)

#save the file (Need to use exp and -1 to change it back)
solution <- data.frame(id = test$Id, SalePrice = exp(prediction1)-1)

#check negative value just in case
which(solution$SalePrice < 0)

#save
write.csv(solution, file = 'xgb_Sol8.csv', row.names = F)
```
I got public score 0.12844 which is slightly better than just using XGBOOST. 

#### H2o Ensemble model

```
options(java.parameters = "- Xmx1024m")
library(h2oEnsemble)
localH2o = h2o.init(nthreads = -1, max_mem_size = "4g")


training_frame <- as.h2o(training1)
validation_frame <- as.h2o(testing1)

# Specify the base learner library & the metalearner
learner <- c("h2o.glm.wrapper", "h2o.randomForest.wrapper", 
             "h2o.gbm.wrapper", "h2o.deeplearning.wrapper")
metalearner <- "SL.glm"



y <- "SalePrice"
x <- setdiff(names(training_frame), y)
family <- "AUTO"

fit <- h2o.ensemble(x = x, y = y, 
                    training_frame = training_frame, 
                    family = family, 
                    learner = learner, 
                    metalearner = metalearner,
                    cvControl = list(V = 5, shuffle = TRUE))



pred <- predict.h2o.ensemble(fit, validation_frame)

pred$pred

rmse(testing$SalePrice, pred$pred)
```
I got 0.1017 RMSE which is lower(better) than XGBOOST and Caret Ensemble.


```
#Real prediction
train2 <- train1
test2 <- test1


localH2o = h2o.init(nthreads = 5)


training_frame <- as.h2o(train2)
validation_frame <- as.h2o(test2)

# Specify the base learner library & the metalearner
learner <- c("h2o.glm.wrapper", "h2o.randomForest.wrapper", 
             "h2o.gbm.wrapper", "h2o.deeplearning.wrapper")
metalearner <- "SL.glm"


y <- "SalePrice"
x <- setdiff(names(training_frame), y)
family <- "AUTO"

fit <- h2o.ensemble(x = x, y = y, 
                    training_frame = training_frame, 
                    family = family, 
                    learner = learner, 
                    metalearner = metalearner,
                    cvControl = list(V = 5, shuffle = TRUE))



pred <- predict.h2o.ensemble(fit, validation_frame)


#save the file (Need to use exp and -1 to change it back)
solution <- data.frame(id = test$Id, SalePrice = exp(pred$pred)-1)

#check negative value just in case
which(solution$SalePrice < 0)

#save
write.csv(solution, file = 'h2o.csv', row.names = F)

h2o.shutdown()
```
I got public score 0.12633 RMSE.


## Conclusion
Boruta package improved my RMSE dramatically as my XGBOOST score with Boruta package was much better than not using Boruta package. I found out that using ensemble is better than just using one XGBOOST. My Kaggle score for using Caret Ensemble and H2o Ensemble is almost same but in terms of speed, H2o Ensemble was faster than Caret Ensemble. 


