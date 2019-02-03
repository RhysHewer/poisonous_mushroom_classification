#------------------------------------------------------------------------------#
# Developer: Rhys Hewer
# Project: Datathon Mushroom
# Version: 1
# Purpose: Classify muschrooms as poisonous or edible
#------------------------------------------------------------------------------#

#Load libraries
library(tidyverse)
library(caret)
library(parallel)
library(doParallel)
library(InformationValue)

#Load Data 
data <- read.csv("data/training1.csv")

##### FEATURE SELECTION #######################################################

#Select features
modData <- data %>% select(class, 
                             cap.shape, 
                             cap.color, 
                             stalk.color.above.ring, 
                             stalk.color.below.ring)

##### MODELLING PREPARATION ###################################################

#Parallel processing
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)

#Cross Validation
fitControl<- trainControl(method = "cv", 
                          number = 5, 
                          savePredictions = TRUE, 
                          allowParallel = TRUE)

#Training/Testing
set.seed(111)
trainIndex <- createDataPartition(modData$class, p = 0.7, list = FALSE)
training <- modData[ trainIndex,]
testing  <- modData[-trainIndex,]


##### CLASSIFICATION ###########################################

#train model glm
model <- glm(class ~., data = training, family = "binomial")
save(model, file = "output/model1.RDS")

#Predictions
preds <- plogis(predict(model, testing))
testing$predprob <- preds

#tuning predictions using log regressions probability feature.
testing$preds <- ifelse(testing$predprob > 0.999, "p", "e") %>% as.factor()

#Confusion Matrix
confMat <- confusionMatrix(testing$class, testing$preds)
confMat

