#preparation
train1 <- read.csv("./data/cleaned_train.csv", na.strings = c("", "NA"), stringsAsFactors = T)
train1 <- train1 %>% select(OverallQual, GrLivArea, TotalBsmtSF, Fireplaces, YearBuilt, LotArea, X1stFlrSF, GarageCars, GarageArea, YearRemodAdd, SalePrice)
train1 <- train1 %>% mutate(SalePrice = log(SalePrice + 1))
train1$OverallQual <- as.factor(train1$OverallQual)
train1$Fireplaces <- as.factor(train1$Fireplaces)
train1$GarageCars <- as.factor(train1$GarageCars)

#split train
set.seed(54321)
outcome <- train1$SalePrice

partition <- createDataPartition(y=outcome,
                                 p=.7,
                                 list=F)
training <- train1[partition,]
testing <- train1[-partition,]

#XGBOOST parameter tuning (Grid Search)
train.control <- trainControl(method = "repeatedcv", repeats = 2,number = 3, search = "grid")

tune.grid <- expand.grid(nrounds = c(50,100,150,200),
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


prediction <- predict(caret.cv,testing)

#Check RMSE
rmse(testing$SalePrice, prediction)

#random forest
control <- trainControl(method="repeatedcv", number=2, repeats=2)
seed <- 7
metric <- "RMSE"
set.seed(seed)
mtry <- 3
tunegrid <- expand.grid(.mtry=mtry)
modelFit <- train(SalePrice~., data=training, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)

prediction <- predict(modelFit,testing)

#Check RMSE
rmse(testing$SalePrice, prediction)

#XGBOOST parameter tuning (Grid Search)
train.control <- trainControl(method = "repeatedcv", repeats = 2,number = 3, search = "grid")

tune.grid <- expand.grid(nrounds = c(50,100,150,200),
                         max_depth = c(5,6,7),
                         eta = c(0.10, 0.2),
                         gamma = c(0.0, 0.2),
                         colsample_bytree = c(0.5,0.7,1),
                         min_child_weight= c(5,7), 
                         subsample =c(0.5,0.7,1))



caret.cv <-caret::train(SalePrice ~.,
                        data=train1,
                        method="xgbTree",
                        metric = "RMSE",
                        tuneGrid=tune.grid,
                        trControl=train.control)


#save model
save(caret.cv, file = "my_model.rda")
