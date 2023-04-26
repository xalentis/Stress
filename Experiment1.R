# Experiment 1

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
library(ggsci)
library(e1071)
library(randomForest)

options(scipen=999)
set.seed(123)


#########################################################################################################################################################
# Load data, no feature-engineering
#########################################################################################################################################################
data_neuro <- stresshelpers::make_neuro_data('NEURO', feature_engineering = FALSE)
data_swell <- stresshelpers::make_swell_data('SWELL', feature_engineering = FALSE)
data_wesad <- stresshelpers::make_wesad_data('WESAD', feature_engineering = FALSE)

#########################################################################################################################################################
# Model training - SVM. LOSO on SWELL, NEURO, WESAD
#########################################################################################################################################################

# SWELL
subjects <- unique(data_swell$Subject)
index <- 1
results <- NULL

for (subject in subjects)
{
  print(subject)
  val <- data_swell[data_swell$Subject == subject,]
  temp <- data_swell[!(data_swell$Subject == subject),]
  
  val$metric <- as.factor(val$metric)
  val$Subject <- NULL
  temp$metric <- as.factor(temp$metric)
  temp$Subject <- NULL
  
  model_svm = svm(x = temp[,1:2], y = temp$metric, kernel = "radial", cost = 5, scale = FALSE)
  y_hat = predict(model_svm, newdata = val[,1:2])
  acc <- sum(val$metric == y_hat)/nrow(val)
  precision <- posPredValue(y_hat, val$metric, positive="1")
  recall <- sensitivity(y_hat, val$metric, positive="1")
  F1 <- (2 * precision * recall) / (precision + recall)
  
  res <- cbind(subject, acc, precision, recall, F1)
  res <- as.data.frame(res)
  names(res) <- c("SUBJECT","ACC", "PRECISION", "RECALL", "F1")
  results <- rbind(results, res)
}
print(mean(as.numeric(results$ACC))) # 58.9%
write.csv(results, "Ex1_SVM_Swell.csv", row.names = FALSE)

# NEURO
subjects <- unique(data_neuro$Subject)
index <- 1
results <- NULL

for (subject in subjects)
{
  print(subject)
  val <- data_neuro[data_neuro$Subject == subject,]
  temp <- data_neuro[!(data_neuro$Subject == subject),]
  
  val$metric <- as.factor(val$metric)
  val$Subject <- NULL
  temp$metric <- as.factor(temp$metric)
  temp$Subject <- NULL
  
  model_svm = svm(x = temp[,1:2], y = temp$metric, kernel = "radial", cost = 5, scale = FALSE)
  y_hat = predict(model_svm, newdata = val[,1:2])
  acc <- sum(val$metric == y_hat)/nrow(val)
  precision <- posPredValue(y_hat, val$metric, positive="1")
  recall <- sensitivity(y_hat, val$metric, positive="1")
  F1 <- (2 * precision * recall) / (precision + recall)
  
  res <- cbind(subject, acc, precision, recall, F1)
  res <- as.data.frame(res)
  names(res) <- c("SUBJECT","ACC", "PRECISION", "RECALL", "F1")
  results <- rbind(results, res)
}
print(mean(as.numeric(results$ACC))) # 54.3%
write.csv(results, "Ex1_SVM_Neuro.csv", row.names = FALSE)

# WESAD
subjects <- unique(data_wesad$Subject)
index <- 1
results <- NULL

for (subject in subjects)
{
  print(subject)
  val <- data_wesad[data_wesad$Subject == subject,]
  temp <- data_wesad[!(data_wesad$Subject == subject),]
  
  val$metric <- as.factor(val$metric)
  val$Subject <- NULL
  temp$metric <- as.factor(temp$metric)
  temp$Subject <- NULL
  
  model_svm = svm(x = temp[,1:2], y = temp$metric, kernel = "radial", cost = 5, scale = FALSE)
  y_hat = predict(model_svm, newdata = val[,1:2])
  acc <- sum(val$metric == y_hat)/nrow(val)
  precision <- posPredValue(y_hat, val$metric, positive="1")
  recall <- sensitivity(y_hat, val$metric, positive="1")
  F1 <- (2 * precision * recall) / (precision + recall)
  
  res <- cbind(subject, acc, precision, recall, F1)
  res <- as.data.frame(res)
  names(res) <- c("SUBJECT","ACC", "PRECISION", "RECALL", "F1")
  results <- rbind(results, res)
}
print(mean(as.numeric(results$ACC))) # 60.22%
write.csv(results, "Ex1_SVM_Wesad.csv", row.names = FALSE)

#########################################################################################################################################################
# Model training - Random Forest LOSO on SWELL, NEURO, WESAD
#########################################################################################################################################################

# SWELL
subjects <- unique(data_swell$Subject)
index <- 1
results <- NULL

for (subject in subjects)
{
  print(subject)
  val <- data_swell[data_swell$Subject == subject,]
  temp <- data_swell[!(data_swell$Subject == subject),]
  
  val$metric <- as.factor(val$metric)
  val$Subject <- NULL
  temp$metric <- as.factor(temp$metric)
  temp$Subject <- NULL
  
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
print(mean(as.numeric(results$ACC))) # 58.07%
write.csv(results, "Ex1_RF_Swell.csv", row.names = FALSE)

# NEURO
subjects <- unique(data_neuro$Subject)
index <- 1
results <- NULL

for (subject in subjects)
{
  print(subject)
  val <- data_neuro[data_neuro$Subject == subject,]
  temp <- data_neuro[!(data_neuro$Subject == subject),]
  
  val$metric <- as.factor(val$metric)
  val$Subject <- NULL
  temp$metric <- as.factor(temp$metric)
  temp$Subject <- NULL
  
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
print(mean(as.numeric(results$ACC))) # 51.09%
write.csv(results, "Ex1_RF_Neuro.csv", row.names = FALSE)

# WESAD
subjects <- unique(data_wesad$Subject)
index <- 1
results <- NULL

for (subject in subjects)
{
  print(subject)
  val <- data_wesad[data_wesad$Subject == subject,]
  temp <- data_wesad[!(data_wesad$Subject == subject),]
  
  val$metric <- as.factor(val$metric)
  val$Subject <- NULL
  temp$metric <- as.factor(temp$metric)
  temp$Subject <- NULL
  
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
print(mean(as.numeric(results$ACC))) # 57.36%
write.csv(results, "Ex1_RF_Wesad.csv", row.names = FALSE)

#########################################################################################################################################################
# Model training - SVM. Train on swell, predict on NEURO and WESAD
#########################################################################################################################################################
data_swell$y <- as.factor(data_swell$metric)
data_neuro$y <- as.factor(data_neuro$metric)
data_wesad$y <- as.factor(data_wesad$metric)

model_svm = svm(x = data_swell[,1:2], y = data_swell$y, kernel = "radial", cost = 5, scale = FALSE)
y_hat = predict(model_svm, newdata = data_neuro[,1:2])
print(sum(as.numeric(data_neuro$y == y_hat))/nrow(data_neuro)) # 47%
# precision, recall, F1 score
precision <- posPredValue(y_hat, data_neuro$y, positive="1") # 52.49
recall <- sensitivity(y_hat, data_neuro$y, positive="1") # 43.17
F1 <- (2 * precision * recall) / (precision + recall) # 47.37

colors<-c("white","lightsteelblue1","dodgerblue3")
text.scl<-5
actual = as.data.frame(table(data_neuro$y))
names(actual) = c("Actual","ActualFreq")
confusion = as.data.frame(table(data_neuro$y, y_hat))
names(confusion) = c("Actual","Predicted","Freq")
confusion = merge(confusion, actual, by=c('Actual','Actual'))
confusion$Percent = confusion$Freq/confusion$ActualFreq*100
confusion$ColorScale<-confusion$Percent*-1
confusion[which(confusion$Actual==confusion$Predicted),]$ColorScale<-confusion[which(confusion$Actual==confusion$Predicted),]$ColorScale*-1
confusion$Label<-paste(round(confusion$Percent,0),"%",sep="")
ggplot() +
  geom_tile(aes(x=Actual, y=Predicted,fill=ColorScale),data=confusion, color="black",size=0.1) +
  labs(x="Actual",y="Predicted") +
  geom_text(aes(x=Actual,y=Predicted, label=Label),data=confusion, size=text.scl, colour="black") +
  scale_fill_gradient2(low=colors[2],high=colors[3],mid=colors[1],midpoint = 0,guide='none') +
  theme_classic() + 
  theme(axis.title = element_text(size = 20, family="Times New Roman",face="bold")) +
  theme(axis.text=element_text(size=12, family="Times New Roman",face="bold"))

y_hat = predict(model_svm, newdata = data_wesad[,1:2])
print(sum(as.numeric(data_wesad$y == y_hat))/nrow(data_wesad)) # 50.47%
# precision, recall, F1 score
precision <- posPredValue(y_hat, data_wesad$y, positive="1") # 31.58
recall <- sensitivity(y_hat, data_wesad$y, positive="1") # 54.09
F1 <- (2 * precision * recall) / (precision + recall) # 39.88

colors<-c("white","lightsteelblue1","dodgerblue3")
text.scl<-5
actual = as.data.frame(table(data_wesad$y))
names(actual) = c("Actual","ActualFreq")
confusion = as.data.frame(table(data_wesad$y, y_hat))
names(confusion) = c("Actual","Predicted","Freq")
confusion = merge(confusion, actual, by=c('Actual','Actual'))
confusion$Percent = confusion$Freq/confusion$ActualFreq*100
confusion$ColorScale<-confusion$Percent*-1
confusion[which(confusion$Actual==confusion$Predicted),]$ColorScale<-confusion[which(confusion$Actual==confusion$Predicted),]$ColorScale*-1
confusion$Label<-paste(round(confusion$Percent,0),"%",sep="")
ggplot() +
  geom_tile(aes(x=Actual, y=Predicted,fill=ColorScale),data=confusion, color="black",size=0.1) +
  labs(x="Actual",y="Predicted") +
  geom_text(aes(x=Actual,y=Predicted, label=Label),data=confusion, size=text.scl, colour="black") +
  scale_fill_gradient2(low=colors[2],high=colors[3],mid=colors[1],midpoint = 0,guide='none') +
  theme_classic() + 
  theme(axis.title = element_text(size = 20, family="Times New Roman",face="bold")) +
  theme(axis.text=element_text(size=12, family="Times New Roman",face="bold"))


#########################################################################################################################################################
# Model training - Random Forest. Train on swell, predict on NEURO and WESAD
#########################################################################################################################################################
model_rf = randomForest(x = data_swell[,1:2], y = data_swell$y, ntree = 200, random_state = 123)

y_hat = predict(model_rf, newdata = data_neuro[,1:2])
print(sum(as.numeric(data_neuro$y == y_hat))/nrow(data_neuro)) # 48.86%
# precision, recall, F1 score
precision <- posPredValue(y_hat, data_neuro$y, positive="1") # 55.67
recall <- sensitivity(y_hat, data_neuro$y, positive="1") # 34.4
F1 <- (2 * precision * recall) / (precision + recall) # 42.53

colors<-c("white","lightsteelblue1","dodgerblue3")
text.scl<-5
actual = as.data.frame(table(data_neuro$y))
names(actual) = c("Actual","ActualFreq")
confusion = as.data.frame(table(data_neuro$y, y_hat))
names(confusion) = c("Actual","Predicted","Freq")
confusion = merge(confusion, actual, by=c('Actual','Actual'))
confusion$Percent = confusion$Freq/confusion$ActualFreq*100
confusion$ColorScale<-confusion$Percent*-1
confusion[which(confusion$Actual==confusion$Predicted),]$ColorScale<-confusion[which(confusion$Actual==confusion$Predicted),]$ColorScale*-1
confusion$Label<-paste(round(confusion$Percent,0),"%",sep="")
ggplot() +
  geom_tile(aes(x=Actual, y=Predicted,fill=ColorScale),data=confusion, color="black",size=0.1) +
  labs(x="Actual",y="Predicted") +
  geom_text(aes(x=Actual,y=Predicted, label=Label),data=confusion, size=text.scl, colour="black") +
  scale_fill_gradient2(low=colors[2],high=colors[3],mid=colors[1],midpoint = 0,guide='none') +
  theme_classic() + 
  theme(axis.title = element_text(size = 20, family="Times New Roman",face="bold")) +
  theme(axis.text=element_text(size=12, family="Times New Roman",face="bold"))

y_hat = predict(model_rf, newdata = data_wesad[,1:2])
print(sum(as.numeric(data_wesad$y == y_hat))/nrow(data_wesad)) # 62.6%
# precision, recall, F1 score
precision <- posPredValue(y_hat, data_wesad$y, positive="1") # 37.09
recall <- sensitivity(y_hat, data_wesad$y, positive="1") # 29.93
F1 <- (2 * precision * recall) / (precision + recall) # 33.13

colors<-c("white","lightsteelblue1","dodgerblue3")
text.scl<-5
actual = as.data.frame(table(data_wesad$y))
names(actual) = c("Actual","ActualFreq")
confusion = as.data.frame(table(data_wesad$y, y_hat))
names(confusion) = c("Actual","Predicted","Freq")
confusion = merge(confusion, actual, by=c('Actual','Actual'))
confusion$Percent = confusion$Freq/confusion$ActualFreq*100
confusion$ColorScale<-confusion$Percent*-1
confusion[which(confusion$Actual==confusion$Predicted),]$ColorScale<-confusion[which(confusion$Actual==confusion$Predicted),]$ColorScale*-1
confusion$Label<-paste(round(confusion$Percent,0),"%",sep="")
ggplot() +
  geom_tile(aes(x=Actual, y=Predicted,fill=ColorScale),data=confusion, color="black",size=0.1) +
  labs(x="Actual",y="Predicted") +
  geom_text(aes(x=Actual,y=Predicted, label=Label),data=confusion, size=text.scl, colour="black") +
  scale_fill_gradient2(low=colors[2],high=colors[3],mid=colors[1],midpoint = 0,guide='none') +
  theme_classic() +
  theme(axis.title = element_text(size = 20, family="Times New Roman",face="bold")) +
  theme(axis.text=element_text(size=12, family="Times New Roman",face="bold"))

data_swell$y <- NULL
data_neuro$y <- NULL
data_wesad$y <- NULL

#########################################################################################################################################################
# Parameter Search
#########################################################################################################################################################
hyper_grid <- expand.grid(
  max_depth = c(6, 8, 10), 
  eta = c(0.01, 0.1),
  subsample = c(0.25, 0.35, 0.5),
  colsample_bytree = c(0.2, 0.4, 0.6, 0.8),
  rmse = 0,
  trees = 0
)

# grid-search with 10 folds
for (i in seq_len(nrow(hyper_grid))) {
  set.seed(123)
  m <- xgb.cv(
    data = as.matrix(data_swell[,1:2]),
    label = data_swell[,3],
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
# Model training - xgboost using optimal parameters. Train on swell, predict on NEURO and WESAD
#########################################################################################################################################################
train.index <- createDataPartition(data_swell$Subject, p = .7, list = FALSE) # 70/30 split along subjects
train <- data_swell[train.index,]
test <- data_swell[-train.index,]

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

model <- xgb.train(
  params = params,
  data = dtrain,
  watchlist = watchlist,
  objective = "reg:logistic",
  nrounds = 5000,
  early_stopping_rounds = 3,
  verbose = 1,
  scale_pos_weight = scale_pos_weight
)

# Best iteration:
# [72]	train-rmse:0.365152	test-rmse:0.372248



importance_matrix <- xgb.importance(model = model)
importance_matrix$Feature <- toupper(importance_matrix$Feature)
xgb.ggplt <- xgb.ggplot.importance(importance_matrix = importance_matrix, top_n = 10)
xgb.ggplt + theme(text = element_text(size = 20),
                  axis.text.x = element_text(size = 20, angle = 45, hjust = 1)) + 
  theme_classic() +  scale_color_lancet() +  scale_fill_lancet() +
  theme(axis.title = element_text(size = 20, family="Times New Roman",face="bold")) +
  theme(axis.text=element_text(size=20, family="Times New Roman",face="bold")) +
  theme(plot.title = element_text(family="Times New Roman",face="bold")) +
  theme(legend.text = element_text(family="Times New Roman",face="bold", size=14)) +
  theme(legend.title =  element_text(family="Times New Roman",face="bold", size=14))

# now validate against neuro
pred <- predict(model, as.matrix(data_neuro[,1:2]))
pred <- round(pred) # round to 0/1 for binary classification (no stress vs. stress)
print(sum(as.numeric(data_neuro$metric == pred))/nrow(data_neuro)) # 50%

# precision, recall, F1 score
precision <- posPredValue(as.factor(pred), as.factor(data_neuro$metric), positive="1") # 0.59
recall <- sensitivity(as.factor(pred), as.factor(data_neuro$metric), positive="1") # 0.36
F1 <- (2 * precision * recall) / (precision + recall) # 0.45

colors<-c("white","lightsteelblue1","dodgerblue3")
text.scl<-5
actual = as.data.frame(table(as.factor(data_neuro$metric)))
names(actual) = c("Actual","ActualFreq")
confusion = as.data.frame(table(as.factor(data_neuro$metric), as.factor(pred)))
names(confusion) = c("Actual","Predicted","Freq")
confusion = merge(confusion, actual, by=c('Actual','Actual'))
confusion$Percent = confusion$Freq/confusion$ActualFreq*100
confusion$ColorScale<-confusion$Percent*-1
confusion[which(confusion$Actual==confusion$Predicted),]$ColorScale<-confusion[which(confusion$Actual==confusion$Predicted),]$ColorScale*-1
confusion$Label<-paste(round(confusion$Percent,0),"%",sep="")
ggplot() +
  geom_tile(aes(x=Actual, y=Predicted,fill=ColorScale),data=confusion, color="black",size=0.1) +
  labs(x="Actual",y="Predicted") +
  geom_text(aes(x=Actual,y=Predicted, label=Label),data=confusion, size=text.scl, colour="black") +
  scale_fill_gradient2(low=colors[2],high=colors[3],mid=colors[1],midpoint = 0,guide='none') +
  theme_classic() +
  theme(axis.title = element_text(size = 20, family="Times New Roman",face="bold")) +
  theme(axis.text=element_text(size=12, family="Times New Roman",face="bold"))

# validate against wesad
pred <- predict(model, as.matrix(data_wesad[,1:2]))
pred <- round(pred) # round to 0/1 for binary classification (no stress vs. stress)
print(sum(as.numeric(data_wesad$metric == pred))/nrow(data_wesad)) # 66%

# precision, recall, F1 score
precision <- posPredValue(as.factor(pred), as.factor(data_wesad$metric), positive="1") # 0.27
recall <- sensitivity(as.factor(pred), as.factor(data_wesad$metric), positive="1") # 0.29
F1 <- (2 * precision * recall) / (precision + recall) # 0.28

colors<-c("white","lightsteelblue1","dodgerblue3")
text.scl<-5
actual = as.data.frame(table(as.factor(data_wesad$metric)))
names(actual) = c("Actual","ActualFreq")
confusion = as.data.frame(table(as.factor(data_wesad$metric), as.factor(pred)))
names(confusion) = c("Actual","Predicted","Freq")
confusion = merge(confusion, actual, by=c('Actual','Actual'))
confusion$Percent = confusion$Freq/confusion$ActualFreq*100
confusion$ColorScale<-confusion$Percent*-1
confusion[which(confusion$Actual==confusion$Predicted),]$ColorScale<-confusion[which(confusion$Actual==confusion$Predicted),]$ColorScale*-1
confusion$Label<-paste(round(confusion$Percent,0),"%",sep="")
ggplot() +
  geom_tile(aes(x=Actual, y=Predicted,fill=ColorScale),data=confusion, color="black",size=0.1) +
  labs(x="Actual",y="Predicted") +
  geom_text(aes(x=Actual,y=Predicted, label=Label),data=confusion, size=text.scl, colour="black") +
  scale_fill_gradient2(low=colors[2],high=colors[3],mid=colors[1],midpoint = 0,guide='none') +
  theme_classic() +
  theme(axis.title = element_text(size = 20, family="Times New Roman",face="bold")) +
  theme(axis.text=element_text(size=12, family="Times New Roman",face="bold"))