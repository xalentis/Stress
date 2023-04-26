# Experiment 3

# Ensemble Machine Learning Model Trained on a New Synthesized Dataset Generalizes Well for Stress Prediction Using Wearable Device
# Gideon Vos, Master of Philosophy, James Cook University, 2022

# Citations:

# WESAD (Wearable Stress and Affect Detection)
# Philip Schmidt, Attila Reiss, Robert Duerichen, Claus Marberger, and Kristof Van Laerhoven. 2018. 
# Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection. 
# In Proceedings of the 20th ACM International Conference on Multimodal Interaction (ICMI '18). 
# Association for Computing Machinery, New York, NY, USA, 400–408. DOI:https://doi.org/10.1145/3242969.3242985

# The SWELL Knowledge Work Dataset for Stress and User Modeling Research
# Koldijk, S., Sappelli, M., Verberne, S., Neerincx, M., & Kraaij, W. (2014). 
# The SWELL Knowledge Work Dataset for Stress and User Modeling Research.
# To appear in: Proceedings of the 16th ACM International Conference on Multimodal Interaction (ICMI 2014) (Istanbul, Turkey, 12-16 November 2014). 
# The dataset can be accessed medio 2015 here: http://persistent-identifier.nl/?identifier=urn:nbn:nl:ui:13-kwrv-3e.

# Non-EEG Dataset for Assessment of Neurological Status
# Birjandtalab, Javad, Diana Cogan, Maziyar Baran Pouyan, and Mehrdad Nourani, 
# A Non-EEG Biosignals Dataset for Assessment and Visualization of Neurological Status, 
# 2016 IEEE International Workshop on Signal Processing Systems (SiPS), Dallas, TX, 2016, pp. 110-114. doi: 10.1109/SiPS.2016.27

# Toadstool: A Dataset for Training Emotional Intelligent Machines Playing Super Mario Bros
# Svoren, H., Thambawita, V., Halvorsen, P., Jakobsen, P., Garcia-Ceja, E., Noori, F. M., … Hicks, S. (2020, February 28). 
# https://doi.org/10.31219/osf.io/4v9mp

# The UBFC-Phys dataset is a public multimodal dataset dedicated to psychophysiological studies
# Meziati Sabour, Y. Benezeth, P. De Oliveira, J. Chappé, F. Yang. "UBFC-Phys: A Multimodal Database For Psychophysiological Studies Of Social Stress",
# IEEE Transactions on Affective Computing, 2021.

# A Wearable Exam Stress Dataset for Predicting Cognitive Performance in Real-World Settings
# Amin, M. R., Wickramasuriya, D., & Faghih, R. T. (2022). A Wearable Exam Stress Dataset for Predicting Cognitive Performance in Real-World Settings (version 1.0.0). 
# PhysioNet. https://doi.org/10.13026/kvkb-aj90.


library(dplyr)
library(caret)
library(xgboost)
library(zoo)
library(stresshelpers)
library(randomForest)

options(scipen=999)

set.seed(123)

#########################################################################################################################################################
# Merge four datasets for training no feature engineering
#########################################################################################################################################################
data_neuro <- stresshelpers::make_neuro_data('NEURO', feature_engineering = FALSE)
data_swell <- stresshelpers::make_swell_data('SWELL', feature_engineering = FALSE)
data_wesad <- stresshelpers::make_wesad_data('WESAD', feature_engineering = FALSE)
data_ubfc <-  stresshelpers::make_ubfc_data('UBFC',   feature_engineering = FALSE)
data <- rbind(data_neuro, data_swell, data_wesad, data_ubfc) # 99 subjects

rm(data_neuro, data_ubfc, data_swell, data_wesad)
gc()

#########################################################################################################################################################
# Model training - Random Forest LOSO no feature engineering
#########################################################################################################################################################

subjects <- unique(data$Subject)
index <- 1
results <- NULL

for (subject in subjects)
{
  print(subject)
  val <- data[data$Subject == subject,]
  temp <- data[!(data$Subject == subject),]
  
  val$metric <- as.factor(val$metric)
  val$Subject <- NULL
  temp$metric <- as.factor(temp$metric)
  temp$Subject <- NULL
  
  if (length(levels(val$metric)) == 2)
  {
    model_rf = randomForest(x = temp[,1:2], y = temp$metric, ntree = 200, random_state = 123)
    y_hat = predict(model_rf, newdata = val[,1:2])
    acc <- sum(val$metric == y_hat)/nrow(val)
    precision <- posPredValue(y_hat, val$metric, positive="1")
    recall <- sensitivity(y_hat, val$metric, positive="1")
    F1 <- (2 * precision * recall) / (precision + recall)
    res <- cbind(subject, acc, precision, recall, F1)
    res <- as.data.frame(res)
    names(res) <- c("SUBJECT","ACC", "PRECISION", "RECALL", "F1")
    results <- rbind(results, res)
  }
}
print(mean(as.numeric(results$ACC))) # 0.4840103
write.csv(results, "Ex3_RF.csv", row.names = FALSE)



#########################################################################################################################################################
# Parameter Search
#########################################################################################################################################################
hyper_grid <- expand.grid(
  max_depth = c(6,8), 
  eta = c(0.08, 0.1,0.5),
  subsample = c(0.35, 0.5, 0.7),
  colsample_bytree = c(0.4, 0.6, 0.8),
  rmse = 0,
  trees = 0
)

# grid-search with 10 folds
for (i in seq_len(nrow(hyper_grid))) {
  set.seed(123)
  m <- xgb.cv(
    data = as.matrix(data[,1:2]),
    label = data$metric,
    nrounds = 100,
    early_stopping_rounds = 3, 
    objective = "reg:logistic",
    nfold = 10,
    verbose = 0,
    params = list( 
      eta = hyper_grid$eta[i], 
      max_depth = hyper_grid$max_depth[i],
      subsample = hyper_grid$subsample[i],
      colsample_bytree = hyper_grid$colsample_bytree[i]
    ) 
  )
  hyper_grid$rmse[i] <- min(m$evaluation_log$test_rmse_mean)
  hyper_grid$trees[i] <- m$best_iteration
}

# display best parameters
hyper_grid %>%
  filter(rmse > 0) %>%
  arrange(rmse) %>%
  glimpse()


#########################################################################################################################################################
# Model training - xgboost using optimal parameters validation using LOSO
#########################################################################################################################################################
subjects <- unique(data$Subject)
index <- 1
results <- NULL

# found using hyper parameter search
params <- list(
  eta = 0.50, 
  max_depth = 8, 
  subsample = 0.70,
  colsample_bytree = 0.4
)

for (subject in subjects)
{
  val <- data[data$Subject == subject,]
  temp <- data[!(data$Subject == subject),]
  
  train.index <- createDataPartition(temp$metric, p = .7, list = FALSE) # 70/30 train/test split along subject
  train <- temp[train.index,]
  test <- temp[-train.index,]
  
  # class balancing
  scale_pos_weight = nrow(train[train$metric==0,])/nrow(train[train$metric==1,])
  
  dtrain <- xgb.DMatrix(data = as.matrix(train[,1:2]), label = train$metric)
  dtest <- xgb.DMatrix(data = as.matrix(test[,1:2]), label = test$metric)
  watchlist <- list(train = dtrain, test = dtest)
  
  model_xgb <- xgb.train(
    params = params,
    data = dtrain,
    objective = "reg:logistic",
    watchlist = watchlist,
    nrounds = 500,
    early_stopping_rounds = 3,
    scale_pos_weight = scale_pos_weight,
    verbose = 0
  )
  
  x_val <- val[,1:2]
  yhat_xgb <- predict(model_xgb, as.matrix(x_val))
  yhat_xgb <- round(yhat_xgb)
  acc_xgb <- sum(as.numeric(val$metric == yhat_xgb))/nrow(val)
  
  # precision, recall, F1 score
  precision <- posPredValue(factor(yhat_xgb, levels=c(0,1)), factor(val$metric, levels=c(0,1)), positive="1")
  recall <- sensitivity(factor(yhat_xgb, levels=c(0,1)), factor(val$metric, levels=c(0,1)), positive="1")
  F1 <- (2 * precision * recall) / (precision + recall)
  
  res <- cbind(subject, acc_xgb, precision, recall, F1)
  res <- as.data.frame(res)
  names(res) <- c("SUBJECT","XGB", "PRECISION", "RECALL", "F1")
  results <- rbind(results, res)
}

results$XGB <- as.numeric(results$XGB)
results$PRECISION <- as.numeric(results$PRECISION)
results$RECALL <- as.numeric(results$RECALL)
results$F1 <- as.numeric(results$F1)
print(mean(results$XGB, na.rm=TRUE)) # 0.6460696
print(mean(results$PRECISION, na.rm=TRUE)) # 0.422624
print(mean(results$RECALL, na.rm=TRUE)) # 0.6811702
print(mean(results$F1, na.rm=TRUE)) # 0.4758952

