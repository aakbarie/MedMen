
# loading packages ------------------------------------------------------------------------------------------------

library(tidyposterior)
library(plyr)
library(caret)
library(randomForest)
library(tidymodels)
library(gbm)
library(DataExplorer)
library(tidyverse)
library(pROC)


# loading data ----------------------------------------------------------------------------------------------------

churn_data <- read_csv("data/src/WA_Fn-UseC_-Telco-Customer-Churn.csv") %>% 
     filter(!is.na(TotalCharges)) %>%
     select(-customerID)


# data sampling ---------------------------------------------------------------------------------------------------

seed <- 7312
set.seed(seed)
data_split <- initial_split(churn_data, strata = "Churn")
churn_train <- training(data_split)
churn_test  <- testing(data_split)

trainX <- churn_train %>%
     select(-Churn)
trainY <- churn_train %>%
     select(Churn)


# creating recipe -------------------------------------------------------------------------------------------------

basic_rec <- recipe(Churn ~ ., data = churn_train) %>%
     step_zv(all_predictors()) %>%
     step_dummy(all_predictors()) %>%
     step_center(all_predictors()) %>%
     step_scale(all_predictors())


# Model 1 setup ---------------------------------------------------------------------------------------------------

metric <- "ROC"

gbmGrid <- expand.grid(
                       n.trees = seq(100, 500, 100),
                       interaction.depth = 2,
                       shrinkage = .07,
                       n.minobsinnode = 10)
     
ctrl <- trainControl(method = "cv",
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary,
                     savePredictions = "final")


# Model 1 run -----------------------------------------------------------------------------------------------------

set.seed(seed)

gbm_mod <- train(basic_rec, 
                 data = churn_train, 
                 method = "gbm",
                 tuneGrid = gbmGrid,
                 trControl = ctrl,
                 metric = metric,
                 verbose = FALSE)



# model accuracy --------------------------------------------------------------------------------------------------

ggplot(gbm_mod) + theme(legend.position = "top")   


# Plot:
plot_roc <- function(x, ...) {
     averaged <- x %>%
          group_by(rowIndex, obs) %>%
          summarise(Yes = mean(Yes, na.rm = TRUE))
     roc_obj <- roc(
          response = x[["obs"]], 
          predictor = x[["Yes"]], 
          levels = rev(levels(x$obs))
     )
     plot(roc_obj, ...)
}

plot_roc(gbm_mod$pred)


# Model 2 Setup ---------------------------------------------------------------------------------------------------

set.seed(seed)

rfGrid <- expand.grid(.mtry = seq(1, 10, 0.5))


# Model 2 run -----------------------------------------------------------------------------------------------------

rf_mod <- train(basic_rec, 
                data = churn_train, 
                method = "rf",
                tuneGrid = rfGrid,
                trControl = ctrl,
                metric = metric,
                ntree = 1000,
                verbose = FALSE)


# model accuracy --------------------------------------------------------------------------------------------------

ggplot(rf_mod) + theme(legend.position = "top")   

plot_roc(gbm_mod$pred)
plot_roc(rf_mod$pred, col = "red", add = TRUE)


# Model 3 ---------------------------------------------------------------------------------------------------------

set.seed(seed)
is_two_levels <- function(x) length(unique(x)) == 2 & is.numeric(x)
probably_dummy <- vapply(churn_train, is_two_levels, logical(1))
dummies <- names(probably_dummy)[probably_dummy]

no_dummies <- recipe(Churn ~ ., data = churn_train) %>%
     step_bin2factor(!!! dummies) %>%
     step_zv(all_predictors())

nbGrid <- expand.grid(usekernel = TRUE,
                              fL = 0.4, 
                              adjust = seq(0.5, 5, by = 0.5))

nb_mod <- train(
     no_dummies,
     data = churn_train,
     method = "nb",
     metric = "ROC",
     tuneGrid = nbGrid,
     trControl = ctrl
)


# model accuracy --------------------------------------------------------------------------------------------------

ggplot(nb_mod) + theme(legend.position = "top")   

plot_roc(gbm_mod$pred,
         legacy.axes = TRUE,
         print.thres = c(.1, .2), 
         print.thres.pattern = "cut = %.2f (Sp = %.3f, Sn = %.3f)",
         print.thres.cex = .8)
plot_roc(rf_mod$pred, 
         legacy.axes = TRUE,
         print.thres = c(.2, .3), 
         print.thres.pattern = "cut = %.2f (Sp = %.3f, Sn = %.3f)",
         print.thres.cex = .8,
         col = "red", 
         add = TRUE)
plot_roc(nb_mod$pred,
         legacy.axes = TRUE,
         print.thres = c(.15), 
         print.thres.pattern = "cut = %.2f (Sp = %.3f, Sn = %.3f)",
         print.thres.cex = .8,
         col = "blue", 
         add = TRUE)


# model selection -------------------------------------------------------------------------------------------------

rs <- resamples(
     list(Boosting = gbm_mod, RndmFrst = rf_mod, NaiveB = nb_mod)
)

roc_mod <- perf_mod(rs, seed = seed, iter = 5000)

roc_dist <- tidy(roc_mod)

summary(roc_dist)

ggplot(roc_dist)

differences <-
     contrast_models(
          roc_mod,
          list_1 = c("RndmFrst", "NaiveB"),
          list_2 = c("Boosting", "RndmFrst"),
          seed = 650
     )

summary(differences, size = 0.025)


differences %>%
     mutate(contrast = paste(model_2, "vs", model_1)) %>%
     ggplot(aes(x = difference, col = contrast)) + 
     geom_line(stat = "density") + 
     geom_vline(xintercept = c(-0.025, 0.025), lty = 2)


test_res <- churn_test %>%
     dplyr::select(Churn) %>%
     mutate(
          prob = predict(rf_mod, churn_test, type = "prob")[, "Yes"],
          pred = predict(rf_mod, churn_test)
     )
roc_curve <- roc(test_res$Churn, test_res$prob, levels = c("No", "Yes"))


roc_curve
getTrainPerf(rf_mod)

plot(
     roc_curve, 
     print.thres = .15,
     print.thres.pattern = "cut = %.2f (Sp = %.3f, Sn = %.3f)",
     legacy.axes = TRUE
)

ggplot(test_res, aes(x = prob)) + geom_histogram(binwidth = .04) + facet_wrap(~Churn)
