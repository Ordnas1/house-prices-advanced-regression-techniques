## Script para predecir valores finales

data_concurso <- read_csv("imputados/test_imp.csv")
data_concurso %<>% mutate_if(is.character, factor)
modelo_final <- readRDS("modelos/model_m5.rds")
SalePrice <- predict(model_m5,data.frame(data_concurso))

win <- data.frame(Id = data_concurso$Id, SalePrice = SalePrice)

write_csv(win, "data/final.csv")
