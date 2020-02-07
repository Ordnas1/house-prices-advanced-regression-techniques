### Advanced regression techniques

## Paquetes----------

library(tidyverse)
library(caret)
library(magrittr)
library(mice)
library(foreach)
library(doMC)

## Parallel----------
registerDoMC(cores = 8)


sumNA <- function(df) {
  ### Summarise NAs in a data.frame
  ### Requires tidyverse
  summ <- as.data.frame(
    sapply(df, function(x) sum(is.na(x)))
  )
  summ <- rownames_to_column(summ)
  names(summ) <- c("var","n")
  summ %<>% filter(n > 0) %>% arrange(desc(n))
  return(summ)
}

## Data--------------

train <- read_csv("train.csv")
test <- read_csv("test.csv") 

## Fullset para imputación y feature engineering
test$SalePrice = NA
full <- rbind(train,test)

## Data explore

glimpse(full)
sapply(full,class)

## Change Character to factor

full %<>% mutate_if(is.character,factor)
sapply(full,class)

## Check NA

NAdf <- sumNA(full)

## NA Imputation

full %<>% mutate(PoolQC = fct_explicit_na(PoolQC),
                 MiscFeature = fct_explicit_na(MiscFeature),
                 Alley = fct_explicit_na(Alley),
                 Fence = fct_explicit_na(Fence),
                 FireplaceQu = fct_explicit_na(FireplaceQu),
                 GarageFinish = fct_explicit_na(GarageFinish),
                 GarageQual = fct_explicit_na(GarageQual),
                 GarageCond = fct_explicit_na(GarageCond),
                 GarageType = fct_explicit_na(GarageType),
                 BsmtCond = fct_explicit_na(BsmtCond),
                 BsmtExposure = fct_explicit_na(BsmtExposure),
                 BsmtQual = fct_explicit_na(BsmtQual),
                 BsmtFinType1 = fct_explicit_na(BsmtFinType1),
                 BsmtFinType2 = fct_explicit_na(BsmtFinType2),
                 MasVnrType = fct_explicit_na(MasVnrType))
full$GarageYrBlt[is.na(full$GarageYrBlt)] = 0

NAdf <- sumNA(full)
NAdf

## Change names of var

names(full)[44] <- "FirstFlrSF"
names(full)[45] <- "SecondFlrSF"
names(full)[70] <- "ThreeSnPorch"

## MICE Imputation

ini <- mice(full, m = 1, method = "mean",maxit = 0)
pred <- ini$pred
pred[,"SalePrice"] <- 0
pred["SalePrice",] <- 0

imp <- mice(full[,-81], method = "cart")

full_imp <- complete(imp)
full_imp <- cbind(full_imp,full[,81])

sumNA(full_imp)

train_imp <- full_imp %>% filter(!is.na(SalePrice))
test_imp <- full_imp %>% filter(is.na(SalePrice))

write_csv(train_imp, "train_imp.csv")
write_csv(test_imp, "test_imp.csv")
