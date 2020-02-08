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

## Función copiada de caret para poder usar la métrica que pide el concurso
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

## Fución para evaluar rápidamente la predicción del modelo
testRMSLES <- function(dataTest, model) {
  df <- data.frame(obs = dataTest$SalePrice, pred = predict(model, dataTest))
  print(plot(df))
  abline(a=0, b=1)
  RMSLESummary(df)
}

## Parallel-------------------------------
## Nota: Modelos basados en RWeka no funcionan con esta opción 
registerDoMC(cores = 8)

## Data--------------

data <- read_csv("imputados/train_imp.csv")
sapply(data, class)

## Change Character to factor

data %<>% mutate_if(is.character, factor)
sapply(data, class)

## Remove near zero var
## Removido, se va a hacer FE. Todo preprocesado se va a pasar directo a caret


## Cross

trainIndex <- createDataPartition(data$SalePrice, p = 0.75, list = FALSE)
train <- data[trainIndex,]
test <- data[-trainIndex,]


## Resample control y pre proce-----------------------------------------------------------------------------------

ctrl <- trainControl(method = "cv", number = 10, summaryFunction = RMSLESummary, 
                     verboseIter = T)
pr <- c("BoxCox", "center", "scale" ,"nzv")

## Model training-------------------------------------------------------------------------------------

## Modelo Lineal
model_lr <- train(SalePrice ~ ., data = train[-1],
                  metric = "RMSLE",
                  method = "lm",
                  trControl = ctrl,
                  preProcess = c("BoxCox", "center", "scale", "nzv"))

model_lr
modelo <- data.frame(obs = test$SalePrice, pred = predict(model_lr,test))

plot(modelo)
abline(a=0,b=1)

RMSLESummary(modelo)


##  CART (Posiblemente no converge) 0.245721

model_rpart <- train(SalePrice ~ ., data = train[-1],
                      metric = "RMSLE",
                      maximize = F,
                      method = "rpart",
                      tuneLength = 10,
                      trControl = ctrl,
                      preProcess = pr)
model_rpart
testRMSLES(test, model_rpart)

## Arbol de reglas, 0.17277

registerDoMC(cores = 1) ## Desactivar para modelos basados en RWeka
model_m5 <- train(SalePrice ~ ., data = train[-1],
                     metric = "RMSLE",
                     maximize = F,
                     method = "M5",
                     control = RWeka::Weka_control(M = 10),
                     trControl = ctrl)

model_m5
testRMSLES(test, model_m5)

