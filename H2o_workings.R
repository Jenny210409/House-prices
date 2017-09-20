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


setwd('c:/kaggle/house prices')
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


train1 <- train1[-c(524,1299),]
train1 <- train1 %>% mutate(SalePrice = log(SalePrice + 1))



#H2o Ensemble
train2 <- train1
test2 <- test1
train2[] <- lapply(train2, as.numeric)
test2[]<-lapply(test2, as.numeric)


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
