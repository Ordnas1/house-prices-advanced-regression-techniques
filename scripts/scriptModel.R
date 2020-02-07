### Advanced regression techniques, model building

## Paquetes----------

library(tidyverse)
library(caret)
library(magrittr)
library(mice)
library(foreach)
library(doMC)
library(MLmetrics)

set.seed(2345)

## functions-------

RMSLESummary <- function(data, lev = NULL, model = NULL){
  mypostResample(data$pred,data$obs)
}

mypostResample <- function(pred,obs){
    isNA <- is.na(pred)
    pred <- pred[!isNA]
    obs <- obs[!isNA]
    if (!is.factor(obs) && is.numeric(obs)) {
      if (length(obs) + length(pred) == 0) {
        out <- rep(NA, 3)
      }
      else {
        if (length(unique(pred)) < 2 || length(unique(obs)) < 
            2) {
          resamplCor <- NA
        }
        else {
          resamplCor <- try(cor(pred, obs, use = "pairwise.complete.obs"), 
                            silent = TRUE)
          if (class(resamplCor) == "try-error") 
            resamplCor <- NA
        }
        RMSLE <- mean((log(pred) - log(obs))^2)
        mae <- mean(abs(pred - obs))
        out <- c(sqrt(RMSLE), resamplCor^2, mae)
      }
      names(out) <- c("RMSLE", "Rsquared", "MAE")
    }
    else {
      if (length(obs) + length(pred) == 0) {
        out <- rep(NA, 2)
      }
      else {
        pred <- factor(pred, levels = levels(obs))
        requireNamespaceQuietStop("e1071")
        out <- unlist(e1071::classAgreement(table(obs, pred)))[c("diag", 
                                                                 "kappa")]
      }
      names(out) <- c("Accuracy", "Kappa")
    }
    if (any(is.nan(out))) 
      out[is.nan(out)] <- NA
    out
  }


## Parallel----------
registerDoMC(cores = 8)

## Data--------------

data <- read_csv("train_imp.csv")
sapply(data, class)

## Change Character to factor

data %<>% mutate_if(is.character, factor)
sapply(data, class)

## Remove near zero var


## Cross

trainIndex <- createDataPartition(data$SalePrice, p = 0.75, list = FALSE)
train <- data[trainIndex,]
test <- data[-trainIndex,]


## Resample control

ctrl <- trainControl(method = "cv", number = 10, summaryFunction = RMSLESummary)


## Model training

model_lr <- train(SalePrice ~ ., data = train,
                  metric = "RMSLE",
                  method = "lm",
                  trControl = ctrl,
                  preProcess = c("BoxCox", "center", "scale", "nzv"))

model_lr
modelo <- data.frame(obs = test$SalePrice, pred = predict(model_lr,test))

plot(modelo)
abline(a=0,b=1)
