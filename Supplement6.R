# Experiment 6

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

library(ggplot2)
library(dplyr)
library(ggsci)
library(caret)
library(xgboost)
library(zoo)
library(e1071)
library(stresshelpers)
library(TTR)
library(randomForest)

options(scipen=999)
set.seed(123)
tensorflow::set_random_seed(123)

#########################################################################################################################################################
# Train on WESAD, test on SWELL and NEURO
#########################################################################################################################################################
data <- stresshelpers::make_wesad_data('WESAD', feature_engineering = TRUE)

data <- data %>% select(hrrange, hrvar, hrstd, hrmin, edarange, edastd, edavar, hrkurt, edamin, hrmax, Subject, metric)
gc()

#########################################################################################################################################################
# Model training - random forest
#########################################################################################################################################################
metric <- data$metric
data$metric <- as.factor(data$metric)
model_rf = randomForest(x = data[,1:10], y = data$metric, ntree = 100, random_state = 123)
data$metric <- metric

#########################################################################################################################################################
# Test on unseen NEURO data
#########################################################################################################################################################
data_neuro <- stresshelpers::make_neuro_data('NEURO', feature_engineering = TRUE)
data_neuro <- data_neuro %>% select(hrrange, hrvar, hrstd, hrmin, edarange, edastd, edavar, hrkurt, edamin, hrmax, Subject, metric)

results <- NULL
subjects <- unique(data_neuro$Subject)
for (subject in subjects)
{
  val <- data_neuro[data_neuro$Subject == subject,]
  val$metric <- as.factor(val$metric)
  yhat_rf <- predict(model_rf, val[,1:10])
  acc_rf <- sum(val$metric == yhat_rf)/nrow(val)
  res <- cbind(subject, acc_rf)
  res <- as.data.frame(res)
  names(res) <- c("SUBJECT","RF")
  results <- rbind(results, res)
}

results$RF <- as.numeric(results$RF)
print(mean(results$RF, na.rm=TRUE)) # 0.4729185

#########################################################################################################################################################
# Test on unseen SWELL data
#########################################################################################################################################################
data_swell <- stresshelpers::make_swell_data('SWELL', feature_engineering = TRUE)
data_swell <- data_swell %>% select(hrrange, hrvar, hrstd, hrmin, edarange, edastd, edavar, hrkurt, edamin, hrmax, Subject, metric)

results <- NULL
subjects <- unique(data_swell$Subject)
for (subject in subjects)
{
  val <- data_swell[data_swell$Subject == subject,]
  val$metric <- as.factor(val$metric)
  yhat_rf <- predict(model_rf, val[,1:10])
  acc_rf <- sum(val$metric == yhat_rf)/nrow(val)
  res <- cbind(subject, acc_rf)
  res <- as.data.frame(res)
  names(res) <- c("SUBJECT","RF")
  results <- rbind(results, res)
}

results$RF <- as.numeric(results$RF)
print(mean(results$RF, na.rm=TRUE)) # 0.6049544


#########################################################################################################################################################
# Train on NEURO, test on SWELL and WESAD
#########################################################################################################################################################
data <- stresshelpers::make_neuro_data('NEURO', feature_engineering = TRUE)

data <- data %>% select(hrrange, hrvar, hrstd, hrmin, edarange, edastd, edavar, hrkurt, edamin, hrmax, Subject, metric)
gc()

#########################################################################################################################################################
# Model training - random forest
#########################################################################################################################################################
metric <- data$metric
data$metric <- as.factor(data$metric)
model_rf = randomForest(x = data[,1:10], y = data$metric, ntree = 100, random_state = 123)
data$metric <- metric

#########################################################################################################################################################
# Test on unseen WESAD data
#########################################################################################################################################################
data_wesad <- stresshelpers::make_wesad_data('WESAD', feature_engineering = TRUE)
data_wesad <- data_wesad %>% select(hrrange, hrvar, hrstd, hrmin, edarange, edastd, edavar, hrkurt, edamin, hrmax, Subject, metric)

results <- NULL
subjects <- unique(data_wesad$Subject)
for (subject in subjects)
{
  val <- data_wesad[data_wesad$Subject == subject,]
  val$metric <- as.factor(val$metric)
  yhat_rf <- predict(model_rf, val[,1:10])
  acc_rf <- sum(val$metric == yhat_rf)/nrow(val)
  res <- cbind(subject, acc_rf)
  res <- as.data.frame(res)
  names(res) <- c("SUBJECT","RF")
  results <- rbind(results, res)
}

results$RF <- as.numeric(results$RF)
print(mean(results$RF, na.rm=TRUE)) # 0.4822473

#########################################################################################################################################################
# Test on unseen SWELL data
#########################################################################################################################################################
data_swell <- stresshelpers::make_swell_data('SWELL', feature_engineering = TRUE)
data_swell <- data_swell %>% select(hrrange, hrvar, hrstd, hrmin, edarange, edastd, edavar, hrkurt, edamin, hrmax, Subject, metric)

results <- NULL
subjects <- unique(data_swell$Subject)
for (subject in subjects)
{
  val <- data_swell[data_swell$Subject == subject,]
  val$metric <- as.factor(val$metric)
  yhat_rf <- predict(model_rf, val[,1:10])
  acc_rf <- sum(val$metric == yhat_rf)/nrow(val)
  res <- cbind(subject, acc_rf)
  res <- as.data.frame(res)
  names(res) <- c("SUBJECT","RF")
  results <- rbind(results, res)
}

results$RF <- as.numeric(results$RF)
print(mean(results$RF, na.rm=TRUE)) # 0.4905855

