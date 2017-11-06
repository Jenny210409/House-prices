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
library(randomForest)

setwd('c:/kaggle/house')
#retrieve train and test
train <- read.csv('train.csv', na.strings = c("", "NA"), stringsAsFactors = F)
test <- read.csv('test.csv', na.strings = c("", "NA"), stringsAsFactors = F)
train1 <- train
test1 <- test

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

#pick important features
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

#combine train and test
total <- bind_rows(train1, test1)
total <- as.data.frame(unclass(total))
#check duplicate
nrow(train) - nrow(unique(train))
#take variables that I got from Boruta package 
total <- total[,CONFIRMED_ATTR]

miceMod <- mice(total, method="rf")
miceOutput <- complete(miceMod)
#Check whether there is missing values
colnames(miceOutput)[colSums(is.na(miceOutput)) > 0]


#separate
train1 <- miceOutput[1:1460,]
test1 <- miceOutput[1461:2919,]
#add SalePrice column to train1
SalePrice <- train$SalePrice
train1 <- cbind(train1,SalePrice)

#checking outlier
rf <- randomForest(SalePrice ~., train1)
prediction <- predict(rf ,train1)
plot((prediction-train1$SalePrice)^2)
which((prediction-train1$SalePrice)^2 >0.5e+10)
train1 <- train1[-c(524,692,804,899,1183,1299),]

#save train1 and test1
write.csv(train1, file = 'cleaned_train.csv', row.names = F)
write.csv(test1, file = 'cleaned_test.csv', row.names = F)

#preparation for analysis
train1[] <- lapply(train1, as.numeric)
test1[]<-lapply(test1, as.numeric)

#convert SalePrice
train1 <- train1 %>% mutate(SalePrice = log(SalePrice + 1))
train1 <- train1 %>% select(OverallQual, GrLivArea, TotalBsmtSF, Fireplaces, YearBuilt, LotArea, X1stFlrSF, GarageCars, GarageArea, YearRemodAdd, SalePrice)


#split train
set.seed(54321)
outcome <- train1$SalePrice

partition <- createDataPartition(y=outcome,
                                 p=.7,
                                 list=F)
training <- train1[partition,]
testing <- train1[-partition,]

#xgb matrix
withoutRV <- training %>% select(-SalePrice)
dtrain <- xgb.DMatrix(as.matrix(withoutRV),label = training$SalePrice)
withoutRV1 <- testing %>% select(-SalePrice)
dtest <- xgb.DMatrix(as.matrix(withoutRV1))

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
xgb_cv <- xgb.cv(xgb_params,dtrain,early_stopping_rounds = 10, nfold = 4, print_every_n = 5, nrounds=1000)

gb_dt <- xgb.train(params = xgb_params,
                   data = dtrain,
                   verbose = 1, maximize =F,
                   nrounds = 74)

prediction <- predict(gb_dt,dtest)

#Check RMSE
rmse(testing$SalePrice, prediction)

#prediction with train1
#xgb matrix
withoutRV <- train1 %>% select(-SalePrice)
dtrain <- xgb.DMatrix(as.matrix(withoutRV),label = train1$SalePrice)


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
xgb_cv <- xgb.cv(xgb_params,dtrain,early_stopping_rounds = 10, nfold = 4, print_every_n = 5, nrounds=1000)

gb_dt <- xgb.train(params = xgb_params,
                   data = dtrain,
                   verbose = 1, maximize =F,
                   nrounds = 90)

#importance
imp_matrix <- as.tibble(xgb.importance(feature_names = colnames(train1 %>% select(-SalePrice)), model = gb_dt))

imp_matrix %>%
  ggplot(aes(reorder(Feature, Gain, FUN = max), Gain, fill = Feature)) +
  geom_col() +
  coord_flip() +
  theme(legend.position = "none") +
  labs(x = "Features", y = "Importance")
