require(e1071)
require(survival)
require(randomForest)
require(gbm)
require(plyr)
require(dplyr)
require(caret)
require(shiny)
require(xgboost)



train1 <- read.csv("./data/cleaned_train.csv", na.strings = c("", "NA"), stringsAsFactors = T)
train1 <- train1 %>% select(OverallQual, GrLivArea, TotalBsmtSF, Fireplaces, YearBuilt, LotArea, X1stFlrSF, GarageCars, GarageArea, YearRemodAdd, SalePrice)
train1 <- train1 %>% mutate(SalePrice = log(SalePrice + 1))
train1$OverallQual <- as.factor(train1$OverallQual)
train1$Fireplaces <- as.factor(train1$Fireplaces)
train1$GarageCars <- as.factor(train1$GarageCars)


load("./data/my_model.rda")
modelFit <- caret.cv


shinyServer(function(input, output){
  
  
  values <- reactiveValues()
  
  newEntry <- observe({
    values$df$OverallQual <- as.factor(input$OverallQual)
    values$df$GrLivArea <- as.integer(input$GrLivArea)
    values$df$TotalBsmtSF <- as.integer(input$TotalBsmtSF)
    values$df$Fireplaces <- as.factor(input$Fireplaces)
    values$df$YearBuilt <- as.integer(input$YearBuilt)
    values$df$LotArea <- as.integer(input$LotArea)
    values$df$X1stFlrSF <- as.integer(input$X1stFlrSF)
    values$df$GarageCars <- as.factor(input$GarageCars)
    values$df$GarageArea <- as.integer(input$GarageArea)
    values$df$YearRemodAdd <- as.integer(input$YearRemodAdd)
  })
  output$results <- renderPrint({
    ds1 <- values$df
    a <- predict(modelFit, newdata = data.frame(ds1))
    a = exp(a)-1
    cat(a)
  })
})