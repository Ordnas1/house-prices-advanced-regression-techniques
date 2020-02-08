### Advanced regression techniques, model building

## Paquetes----------

library(tidyverse)
library(caret)
library(magrittr)
##library(mice)
##library(foreach)
##library(doMC)
##library(MLmetrics)
##library(doSNOW)
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
#cl <- snow::makeCluster(4, type = "SOCK")
#registerDoSNOW(cl)

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
                      tuneLength = 100,
                      trControl = ctrl,
                      preProcess = pr)
model_rpart
testRMSLES(test, model_rpart)
saveRDS(model_rpart,"modelos/model_rpart.rds")
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


## GBM  0.1518--------------------------------------


## Tuneo 1
gbmGrid1 <- expand.grid(.interaction.depth = 3,
                       .n.trees = seq(100, 1000, by = 50),
                       .shrinkage =  0.1,
                       .n.minobsinnode = 10 )

model_gbm1 <- train(SalePrice ~ ., data = train[-1],
                   metric = "RMSLE",
                   maximize = F,
                   method = "gbm",
                   tuneGrid = gbmGrid1,
                   trControl = ctrl)
model_gbm1
model_gbm1$bestTune
testRMSLES(test, model_gbm1)

## Tuneo 2 

gbmGrid2 <- expand.grid(.interaction.depth = 3,
                        .n.trees = model_gbm1$bestTune$n.trees,
                        .shrinkage =  c(0.001, 0.01, 0.025, 0.03, 0.07, 0.1),
                        .n.minobsinnode = 10)

model_gbm2 <- train(SalePrice ~ ., data = train[-1],
                    metric = "RMSLE",
                    maximize = F,
                    method = "gbm",
                    tuneGrid = gbmGrid2,
                    trControl = ctrl)
model_gbm2
model_gbm2$bestTune
testRMSLES(test, model_gbm2)
plot(model_gbm2)

## Tuneo 3

gbmGrid3 <- expand.grid(.interaction.depth = seq(3, 8, by = 1),
                        .n.trees = model_gbm1$bestTune$n.trees,
                        .shrinkage =  model_gbm2$bestTune$shrinkage,
                        .n.minobsinnode = c(5, 10, 15))

model_gbm3 <- train(SalePrice ~ ., data = train[-1],
                    metric = "RMSLE",
                    maximize = F,
                    method = "gbm",
                    tuneGrid = gbmGrid3,
                    trControl = ctrl)
model_gbm3
model_gbm3$bestTune
testRMSLES(test, model_gbm3)

saveRDS(model_gbm3, "modelos/model_gbm_tuned.rds")
##XGBoost---------------------------------------

## Esta data no tiene rango completo, no sirve para modelos lineales
data_dummy <- dummyVars(SalePrice ~ . , data = data)
data_dummy_pred <- predict(data_dummy, data)
data_dummy_pred <- cbind(as.data.frame(data_dummy_pred)[-1],data$SalePrice)
names(data_dummy_pred)[304] <- "SalePrice"
names(data_dummy_pred) <- gsub("\\(", "", names(data_dummy_pred))
names(data_dummy_pred) <- gsub("\\)", "", names(data_dummy_pred))
rownames(data_dummy_pred) <- NULL

train_dummy <- data_dummy_pred[trainIndex,]
test_dummy <- data_dummy_pred[-trainIndex,]  

# grid1
xgb_grid1 <- expand_grid(
  .nrounds = c(100,101),
  .max_depth = 6,
  .eta = 0.3,
  .gamma = 0,
  .colsample_bytree = 1,
  .min_child_weight = 1,
  .subsample = 1
)
## tuneLength = 3, 0.16878

start_time <- Sys.time()
model_xgb1 <- train(SalePrice ~ . , data = train_dummy,
                    metric = "RMSLE",
                    maximize = F,
                    method = "xgbTree",
                    tuneLength = 1,
                    trControl = ctrl)
end_time <- Sys.time()
end_time - start_time
model_xgb1
testRMSLES(test_dummy, model_xgb1)

## Tuneado( 0.16290)
nrounds_max <- 1000

xgb_grid2 <- data.frame(expand_grid(
  .nrounds = seq(200, nrounds_max, by = 50),
  .max_depth = c(2, 3, 4, 5, 6),
  .eta = c(0.025, 0.05, 0.1, 0.3),
  .gamma = 0,
  .colsample_bytree = 1,
  .min_child_weight = 1,
  .subsample = 1
))

model_xgb2 <- train(SalePrice ~ . , data = train_dummy,
                    metric = "RMSLE",
                    maximize = F,
                    method = "xgbTree",
                    tuneGrid = xgb_grid2,
                    trControl = ctrl)
model_xgb2

testRMSLES(test_dummy, model_xgb2)



model_xgb2$bestTune # eta 0.05 max_depth 2

##Tuneo 3 1.629021
xgb_grid3 <- data.frame(expand.grid(
  .nrounds = seq(200, nrounds_max, by = 50),
  .max_depth = ifelse(model_xgb2$bestTune$max_depth == 2,                                   ## Javi@ kaggle max depth 2 a 4, o +- 1
                      c(model_xgb2$bestTune$max_depth:4),
                      model_xgb2$bestTune$max_depth - 1:xgb_tune$bestTune$max_depth + 1),
  .eta = model_xgb2$bestTune$eta,
  .gamma = 0,
  .colsample_bytree = 1,
  .min_child_weight = c(1,2,3),
  .subsample = 1
))

model_xgb3 <- train(SalePrice ~ . , data = train_dummy,
                    metric = "RMSLE",
                    maximize = F,
                    method = "xgbTree",
                    tuneGrid = xgb_grid3,
                    trControl = ctrl)

model_xgb3
testRMSLES(test_dummy,model_xgb3)

## Tuneo 4 (0.15896)

xgb_grid4 <- data.frame(expand.grid(
  .nrounds = seq(200, nrounds_max, by = 50),
  .max_depth = model_xgb3$bestTune$max_depth,
  .eta = model_xgb2$bestTune$eta,
  .gamma = 0,
  .colsample_bytree = c(0.4, 0.6, 0.8, 1.0),
  .min_child_weight = model_xgb3$bestTune$min_child_weight,
  .subsample = c(0.5, 0.75, 1.0)
))

model_xgb4 <- train(SalePrice ~ . , data = train_dummy,
                    metric = "RMSLE",
                    maximize = F,
                    method = "xgbTree",
                    tuneGrid = xgb_grid4,
                    trControl = ctrl)

model_xgb4
testRMSLES(test_dummy,model_xgb4)


## Tuneo 5 0.1594537

xgb_grid5 <- data.frame(expand.grid(
  .nrounds = seq(200, nrounds_max, by = 50),
  .max_depth = model_xgb3$bestTune$max_depth,
  .eta = model_xgb2$bestTune$eta,
  .gamma = c(0, 0.05, 0.01, 0.5, 0.7 ,0.9, 1),
  .colsample_bytree = model_xgb4$bestTune$colsample_bytree,
  .min_child_weight = model_xgb3$bestTune$min_child_weight,
  .subsample = model_xgb4$bestTune$subsample
))

model_xgb5 <- train(SalePrice ~ . , data = train_dummy,
                    metric = "RMSLE",
                    maximize = F,
                    method = "xgbTree",
                    tuneGrid = xgb_grid5,
                    trControl = ctrl)

model_xgb5
testRMSLES(test_dummy,model_xgb5)

## Tuneo 6 0.156

xgb_grid6 <- data.frame(expand.grid(
  .nrounds = seq(200, nrounds_max, by = 50),
  .max_depth = model_xgb3$bestTune$max_depth,
  .eta = c(0.01, 0.015, 0.025, 0.05, 0.01),
  .gamma = model_xgb5$bestTune$gamma,
  .colsample_bytree = model_xgb4$bestTune$colsample_bytree,
  .min_child_weight = model_xgb3$bestTune$min_child_weight,
  .subsample = model_xgb4$bestTune$subsample
))

model_xgb6 <- train(SalePrice ~ . , data = train_dummy,
                    metric = "RMSLE",
                    maximize = F,
                    method = "xgbTree",
                    tuneGrid = xgb_grid6,
                    trControl = ctrl)

model_xgb6
testRMSLES(test_dummy,model_xgb6)

saveRDS(model_xgb6, "modelos/model_xgb6_tuned.rds")

