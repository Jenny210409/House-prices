training <- training %>% select(OverallQual, GrLivArea, TotalBsmtSF, Fireplaces, YearBuilt, LotArea, X1stFlrSF, GarageCars, GarageArea, YearRemodAdd, SalePrice)
testing <- testing %>% select(OverallQual, GrLivArea, TotalBsmtSF, Fireplaces, YearBuilt, LotArea, X1stFlrSF, GarageCars, GarageArea, YearRemodAdd, SalePrice)


mod <- train(SalePrice ~ ., data  = training, method  = "xgbLinear")

prediction <- predict(mod,testing)

#Check RMSE
rmse(testing$SalePrice, prediction)
