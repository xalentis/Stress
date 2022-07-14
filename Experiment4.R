# Experiment 4

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
library(dplyr)
library(caret)
library(xgboost)
library(zoo)
library(e1071)
library(stresshelpers)

options(scipen=999)

set.seed(123)

#########################################################################################################################################################
# Load and Prep Datasets for Training - WITH FEATURE ENGINEERING
#########################################################################################################################################################
data_neuro <- stresshelpers::make_neuro_data('NEURO', feature_engineering = TRUE)
data_swell <- stresshelpers::make_swell_data('SWELL', feature_engineering = TRUE)
data_wesad <- stresshelpers::make_wesad_data('WESAD', feature_engineering = TRUE)
data_ubfc  <-  stresshelpers::make_ubfc_data('UBFC',  feature_engineering = TRUE)
data <- rbind(data_neuro, data_swell, data_wesad, data_ubfc) # 99 subjects

data <- data %>% select(hrmax,hrmin,hrstd,hrmedian,hr,edaskew,edastd,hrmean,eda,edarange, Subject, metric)

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
    data = as.matrix(data[,1:10]),
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
# Model training - xgboost using optimal parameters
#########################################################################################################################################################
split.index <- createDataPartition(data$Subject, p = .9, list = FALSE) # 90% of StressData for training
train <- data[split.index,]
val <- data[-split.index,] # 10% for validation


train.index <- createDataPartition(train$Subject, p = .7, list = FALSE) # 70/30 train/test split along subjects
train <- train[train.index,]
test <- train[-train.index,]

# class balancing
scale_pos_weight = nrow(train[train$metric==0,])/nrow(train[train$metric==1,])

# found using hyper parameter search
params <- list(
  eta = 0.5, 
  max_depth = 8, 
  subsample = 0.35,
  colsample_bytree = 0.8
)

dtrain <- xgb.DMatrix(data = as.matrix(train[,1:10]), label = train$metric)
dtest <- xgb.DMatrix(data = as.matrix(test[,1:10]), label = test$metric)
watchlist <- list(train = dtrain, test = dtest)

model <- xgb.train(
  params = params,
  data = dtrain,
  objective = "reg:logistic",
  watchlist = watchlist,
  nrounds = 5000,
  early_stopping_rounds = 3,
  verbose = 1,
  scale_pos_weight = scale_pos_weight
)

# Best iteration:
# [256]	train-rmse:0.005864	test-rmse:0.005726

pred <- predict(model, as.matrix(val[,1:10]))
pred <- round(pred) # round to 0/1 for binary classification (no stress vs. stress)
print(sum(as.numeric(val$metric == pred))/nrow(val)) # 99.99% on hold-out set

# precision, recall, F1 score
precision <- posPredValue(as.factor(pred), as.factor(val$metric), positive="1") # 0.99
recall <- sensitivity(as.factor(pred), as.factor(val$metric), positive="1") # 0.99
F1 <- (2 * precision * recall) / (precision + recall) # 0.99

#########################################################################################################################################################