training1 <- training
testing1 <- testing
# training1[] <- lapply(training1, as.numeric)
# testing1[]<-lapply(testing1, as.numeric)


localH2o = h2o.init(nthreads = 5)


training_frame <- as.h2o(training)
validation_frame <- as.h2o(testing)

# Specify the base learner library & the metalearner
learner <- c("h2o.glm.wrapper", "h2o.randomForest.wrapper", 
             "h2o.gbm.wrapper", "h2o.deeplearning.wrapper")
metalearner <- "SL.glm"



y <- "SalePrice"
x <- setdiff(names(training_frame), y)
family <- "AUTO"
training_frame[,c(y)] <- as.factor(training_frame[,c(y)]) #Force Binary classification
validation_frame[,c(y)] <- as.factor(validation_frame[,c(y)]) # check to validate that this guarantees the same 0/1 mapping?

fit <- h2o.ensemble(x = x, y = y, 
                    training_frame = training_frame, 
                    family = family, 
                    learner = learner, 
                    metalearner = metalearner,
                    cvControl = list(V = 5, shuffle = TRUE))


prediction <- predict(fit, validation_frame)
pred <- predict.h2o.ensemble(fit, validation_frame)

prediction$pred

rmse(testing$SalePrice, prediction$pred)