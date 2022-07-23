# Experiment 6

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

options(scipen=999)
set.seed(123)

#########################################################################################################################################################
# Load and Prep StressData for Training
#########################################################################################################################################################
data_neuro <- stresshelpers::make_neuro_data('NEURO', feature_engineering = TRUE)
data_swell <- stresshelpers::make_swell_data('SWELL', feature_engineering = TRUE)
data_wesad <- stresshelpers::make_wesad_data('WESAD', feature_engineering = TRUE)
data_ubfc  <-  stresshelpers::make_ubfc_data('UBFC',  feature_engineering = TRUE)

# balancing across data sources: XGB
data_neuro$Balance <- 1
data_swell$Balance <- 1
data_wesad$Balance <- 1
data_ubfc$Balance <- 0

data <- rbind(data_neuro, data_swell, data_wesad, data_ubfc) # 99 subjects
data <- data %>% select(hrrange, hrvar, hrstd, hrmin, edarange, edastd, edavar, hrkurt, edamin, hrmax, Subject, metric)

rm(data_neuro, data_swell, data_wesad, data_ubfc)
gc()

#########################################################################################################################################################
# Model training - xgboost using optimal parameters
#########################################################################################################################################################
train.index <- createDataPartition(data$Subject, p = .7, list = FALSE) # 70/30 train/test split along subjects
train <- data[train.index,]
test <- data[-train.index,]

# class balancing
scale_pos_weight = nrow(train[train$Balance==0,])/nrow(train[train$Balance==1,])

# found using hyper parameter search
params <- list(
  eta = 0.5, 
  max_depth = 8, 
  subsample = 0.70,
  colsample_bytree = 0.8
)
dtrain <- xgb.DMatrix(data = as.matrix(train[,1:10]), label = train$metric)
dtest <- xgb.DMatrix(data = as.matrix(test[,1:10]), label = test$metric)
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

# [151]	train-rmse:0.025898	test-rmse:0.026284

#########################################################################################################################################################
# Build Neural Network Model
#########################################################################################################################################################
x_train <- train[,1:10]
y_train <- train$metric
x_test <- test[,1:10]
y_test <- test$metric

# scale
x_train <- scale(x_train)
x_test <- scale(x_test, center = attr(x_train, "scaled:center") , scale = attr(x_train, "scaled:scale"))

model_nn <- keras_model_sequential()

model_nn %>% 
  layer_dense(
    units              = 10, 
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

# 360/360 [==============================] - 1s 2ms/step - loss: 0.1220 - val_loss: 0.1217

rm(data, history, params, test, train, train.index, watchlist, x_test, dtest, dtrain, scale_pos_weight, y_test, y_train)
gc()

#########################################################################################################################################################
# Test on unseen EXAM data
#########################################################################################################################################################

exam_data <- stresshelpers::make_exam_data('EXAM')
exam_data <- exam_data %>% select(hrrange, hrvar, hrstd, hrmin, edarange, edastd, edavar, hrkurt, edamin, hrmax, Subject)
exam_data <- exam_data[exam_data$Subject %in% c("S1_Final", "S10_Final"),]
gc()

# ensemble weighting
weighted <- function(xgb, ann) (xgb*0.7) + (ann*0.3)

temp <- exam_data[exam_data$Subject=='S1_Final',]
x_val <- temp[,1:10]
yhat_xgb <- predict(model_xgb, as.matrix(x_val))
x_val <- scale(x_val, center = attr(x_train, "scaled:center") , scale = attr(x_train, "scaled:scale"))
yhat_nn <- as.data.frame(predict(model_nn, x_val))
yhat_nn <- yhat_nn[,1]
yhat_nn <- (yhat_nn - min(yhat_nn)) / (max(yhat_nn) - min(yhat_nn))
yhat_ens <- weighted(yhat_xgb, yhat_nn)
temp <- cbind(yhat_xgb, yhat_nn, yhat_ens)
temp <- as.data.frame(temp)
names(temp) <- c("xgb","ann","ens")
temp$ID <- seq.int(nrow(temp))
ggplot(temp, aes(x=ID)) + 
  geom_line(aes(y = ens, colour="ENS"),  size=1,  alpha=0.4) + 
  scale_color_manual(values=c("#0080ff","#FF6666")) + 
  scale_fill_manual(values=c("#0080ff","#FF6666")) + 
  labs(colour="Model") + 
  guides(color = guide_legend(override.aes = list(fill="white", size=5))) + 
  theme_classic() + ylab('Stress - S1 (Final)') + xlab('Time (seconds)') + 
  scale_x_continuous(breaks=seq(0,nrow(temp)+1200,1200)) +
  scale_y_continuous(limits=c(0,1))+
  theme(axis.title = element_text(size = 20, family="Times New Roman",face="bold")) +
  theme(axis.text=element_text(size=14, family="Times New Roman",face="bold")) +
  theme(plot.title = element_text(family="Times New Roman",face="bold")) +
  theme(legend.text = element_text(family="Times New Roman",face="bold", size=14)) +
  theme(legend.title =  element_text(family="Times New Roman",face="bold", size=14))


temp <- exam_data[exam_data$Subject=='S10_Final',]
x_val <- temp[,1:10]
yhat_xgb <- predict(model_xgb, as.matrix(x_val))
x_val <- scale(x_val, center = attr(x_train, "scaled:center") , scale = attr(x_train, "scaled:scale"))
yhat_nn <- as.data.frame(predict(model_nn, x_val))
yhat_nn <- yhat_nn[,1]
yhat_nn <- (yhat_nn - min(yhat_nn)) / (max(yhat_nn) - min(yhat_nn))
yhat_ens <- weighted(yhat_xgb, yhat_nn)
temp <- cbind(yhat_xgb, yhat_nn, yhat_ens)
temp <- as.data.frame(temp)
names(temp) <- c("xgb","ann","ens")
temp$ID <- seq.int(nrow(temp))
ggplot(temp, aes(x=ID)) + 
  geom_line(aes(y = ens, colour="ENS"),  size=1,  alpha=0.4) + 
  scale_color_manual(values=c("#0080ff","#FF6666")) + 
  scale_fill_manual(values=c("#0080ff","#FF6666")) + 
  labs(colour="Model") + 
  guides(color = guide_legend(override.aes = list(fill="white", size=5))) + 
  theme_classic() + ylab('Stress - S10 (Final)') + xlab('Time (seconds)') + 
  scale_x_continuous(breaks=seq(0,nrow(temp)+1200,1200)) +
  scale_y_continuous(limits=c(0,1))+
  theme(axis.title = element_text(size = 20, family="Times New Roman",face="bold")) +
  theme(axis.text=element_text(size=14, family="Times New Roman",face="bold")) +
  theme(plot.title = element_text(family="Times New Roman",face="bold")) +
  theme(legend.text = element_text(family="Times New Roman",face="bold", size=14)) +
  theme(legend.title =  element_text(family="Times New Roman",face="bold", size=14))
