# Experiment 8

# Investigating Wearable Sensor Biomarkers for Chronic Stress Measurement and Analysis
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

library(ggplot2)
library(dplyr)
library(ggsci)
library(caret)
library(xgboost)
library(zoo)
library(e1071)
library(stresshelpers)
library(keras)
library(TTR)

options(scipen=999)
set.seed(123)
tensorflow::set_random_seed(123)

#########################################################################################################################################################
# Load and Prep SWELL for Training
#########################################################################################################################################################
data <- stresshelpers::make_swell_data('SWELL', feature_engineering = FALSE)
gc()

#########################################################################################################################################################
# Model training - xgboost using optimal parameters
#########################################################################################################################################################
train.index <- createDataPartition(data$Subject, p = .7, list = FALSE) # 70/30 train/test split along subjects
train <- data[train.index,]
test <- data[-train.index,]

# class balancing
scale_pos_weight = nrow(train[train$metric==0,])/nrow(train[train$metric==1,])

# found using hyper parameter search
params <- list(
  eta = 0.1, 
  max_depth = 8, 
  subsample = 0.5,
  colsample_bytree = 0.2
)
dtrain <- xgb.DMatrix(data = as.matrix(train[,1:2]), label = train$metric)
dtest <- xgb.DMatrix(data = as.matrix(test[,1:2]), label = test$metric)
watchlist <- list(train = dtrain, test = dtest)

model_xgb <- xgb.train(
  params = params,
  data = dtrain,
  objective = "reg:logistic",
  watchlist = watchlist,
  nrounds = 5000,
  early_stopping_rounds = 3,
  verbose = 1
)

# 216/216 [==============================] - 0s 2ms/step - loss: 0.2039 - val_loss: 0.2047

#########################################################################################################################################################
# Build Neural Network Model
#########################################################################################################################################################
x_train <- train[,1:2]
y_train <- train$metric
x_test <- test[,1:2]
y_test <- test$metric

# scale
x_train <- scale(x_train)
x_test <- scale(x_test, center = attr(x_train, "scaled:center") , scale = attr(x_train, "scaled:scale"))

model_nn <- keras_model_sequential()

model_nn %>% 
  layer_dense(
    units              = 2, 
    kernel_initializer = "normal", 
    activation         = "relu", 
    input_shape        = ncol(x_train)) %>% 
  
  layer_dense(
    units              = 5, 
    kernel_initializer = "normal", 
    activation         = "relu") %>% 
  
  layer_dense(
    units              = 1, 
    kernel_initializer = "normal",
    activation         = "linear") %>%
  
  compile(
    loss = "mse",
    optimizer = optimizer_adamax()
  )

history <- fit(
  object           = model_nn, 
  x                = x_train, 
  y                = y_train,
  batch_size       = 512, 
  epochs           = 120,
  validation_data  = list(x_test, y_test),
  shuffle          = TRUE,
  callbacks        = list(callback_early_stopping(monitor = "val_loss", patience = 5, restore_best_weights = TRUE))
)

# 216/216 [==============================] - 0s 2ms/step - loss: 0.1518 - val_loss: 0.1516

#########################################################################################################################################################
# Test on unseen NEURO data
#########################################################################################################################################################
data_neuro <- stresshelpers::make_neuro_data('NEURO', feature_engineering = FALSE)

weighted <- function(xgb, ann) (xgb*0.7) + (ann*0.3)

results <- NULL
subjects <- unique(data_neuro$Subject)
for (subject in subjects)
{
  val <- data_neuro[data_neuro$Subject == subject,]
  x_val <- val[,1:2]
  yhat_xgb <- predict(model_xgb, as.matrix(x_val))
  x_val <- scale(x_val, center = attr(x_train, "scaled:center") , scale = attr(x_train, "scaled:scale"))
  yhat_nn <- as.data.frame(predict(model_nn, x_val))
  yhat_nn <- yhat_nn[,1]
  yhat_nn[yhat_nn > 1] <- 1
  yhat_xgb <- round(yhat_xgb)
  yhat_nn <- round(yhat_nn)
  
  yhat_ens <- weighted(yhat_xgb, yhat_nn)
  
  yhat_ens <- round(yhat_ens)
  acc_xgb <- sum(as.numeric(val$metric == yhat_xgb))/nrow(val)
  acc_ann <- sum(as.numeric(val$metric == yhat_nn))/nrow(val)
  acc_ens <- sum(as.numeric(val$metric == yhat_ens))/nrow(val)
  
  # precision, recall, F1 score
  precision <- posPredValue(factor(yhat_ens, levels=c(0,1)), factor(val$metric, levels=c(0,1)), positive="1")
  recall <- sensitivity(factor(yhat_ens, levels=c(0,1)), factor(val$metric, levels=c(0,1)), positive="1")
  F1 <- (2 * precision * recall) / (precision + recall)
  
  res <- cbind(subject, acc_xgb, acc_ann, acc_ens, precision, recall, F1)
  res <- as.data.frame(res)
  names(res) <- c("SUBJECT","XGB","ANN","ENS", "PRECISION", "RECALL", "F1")
  results <- rbind(results, res)
}

results$XGB <- as.numeric(results$XGB)
results$ANN <- as.numeric(results$ANN)
results$ENS <- as.numeric(results$ENS)
results$PRECISION <- as.numeric(results$PRECISION)
results$RECALL <- as.numeric(results$RECALL)
results$F1 <- as.numeric(results$F1)

print(mean(results$XGB, na.rm=TRUE)) # 0.51
print(mean(results$ANN, na.rm=TRUE)) # 0.41
print(mean(results$ENS, na.rm=TRUE)) # 0.51
print(mean(results$PRECISION, na.rm=TRUE)) # 0.67
print(mean(results$RECALL, na.rm=TRUE)) # 0.30
print(mean(results$F1, na.rm=TRUE)) # 0.38


#########################################################################################################################################################
# Test on unseen WESAD data
#########################################################################################################################################################
data_wesad <- stresshelpers::make_wesad_data('WESAD', feature_engineering = FALSE)

results <- NULL
subjects <- unique(data_wesad$Subject)
for (subject in subjects)
{
  val <- data_wesad[data_wesad$Subject == subject,]
  x_val <- val[,1:2]
  yhat_xgb <- predict(model_xgb, as.matrix(x_val))
  x_val <- scale(x_val, center = attr(x_train, "scaled:center") , scale = attr(x_train, "scaled:scale"))
  yhat_nn <- as.data.frame(predict(model_nn, x_val))
  yhat_nn <- yhat_nn[,1]
  yhat_nn[yhat_nn > 1] <- 1
  yhat_xgb <- round(yhat_xgb)
  yhat_nn <- round(yhat_nn)
  
  yhat_ens <- weighted(yhat_xgb, yhat_nn)
  
  yhat_ens <- round(yhat_ens)
  acc_xgb <- sum(as.numeric(val$metric == yhat_xgb))/nrow(val)
  acc_ann <- sum(as.numeric(val$metric == yhat_nn))/nrow(val)
  acc_ens <- sum(as.numeric(val$metric == yhat_ens))/nrow(val)
  
  # precision, recall, F1 score
  precision <- posPredValue(factor(yhat_ens, levels=c(0,1)), factor(val$metric, levels=c(0,1)), positive="1")
  recall <- sensitivity(factor(yhat_ens, levels=c(0,1)), factor(val$metric, levels=c(0,1)), positive="1")
  F1 <- (2 * precision * recall) / (precision + recall)
  
  res <- cbind(subject, acc_xgb, acc_ann, acc_ens, precision, recall, F1)
  res <- as.data.frame(res)
  names(res) <- c("SUBJECT","XGB","ANN","ENS", "PRECISION", "RECALL", "F1")
  results <- rbind(results, res)
}

results$XGB <- as.numeric(results$XGB)
results$ANN <- as.numeric(results$ANN)
results$ENS <- as.numeric(results$ENS)
results$PRECISION <- as.numeric(results$PRECISION)
results$RECALL <- as.numeric(results$RECALL)
results$F1 <- as.numeric(results$F1)

print(mean(results$XGB, na.rm=TRUE)) # 0.65
print(mean(results$ANN, na.rm=TRUE)) # 0.69
print(mean(results$ENS, na.rm=TRUE)) # 0.65
print(mean(results$PRECISION, na.rm=TRUE)) # 0.59
print(mean(results$RECALL, na.rm=TRUE)) # 0.19
print(mean(results$F1, na.rm=TRUE)) # 0.31
