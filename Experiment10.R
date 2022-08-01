# Experiment 10

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
library(tensorflow)
library(TTR)

options(scipen=999)
set.seed(123)
tensorflow::set_random_seed(123)

make_exam_data <- function(folder) {
  subjects <- c("S1","S2","S3","S4","S5","S6","S7","S8","S9","S10")
  data <- NULL
  
  for (subject in subjects)
  {
    subject_file <- stresshelpers::read.empatica(paste(folder, "/",subject,"/Midterm 1",sep=""))
    hr <- subject_file$signal$hr$data
    eda <- subject_file$signal$eda$data
    eda <- eda[2:(length(eda)-1)]
    hr <- hr[2:(length(hr)-1)]
    eda <- stresshelpers::downsample(eda, round(length(eda) / length(hr)))
    shortest <- min(length(eda), length(hr))
    hr <- hr[1:shortest]
    eda <- eda[1:shortest]
    dataset <- cbind(eda, hr)
    dataset <- as.data.frame(dataset)
    dataset$Subject <- paste(subject,"Midterm1",sep='_')
    data <- rbind(data, dataset)
    
    subject_file <- stresshelpers::read.empatica(paste(folder, "/",subject,"/Midterm 2",sep=""))
    hr <- subject_file$signal$hr$data
    eda <- subject_file$signal$eda$data
    eda <- eda[2:(length(eda)-1)]
    hr <- hr[2:(length(hr)-1)]
    eda <- stresshelpers::downsample(eda, round(length(eda) / length(hr)))
    shortest <- min(length(eda), length(hr))
    hr <- hr[1:shortest]
    eda <- eda[1:shortest]
    dataset <- cbind(eda, hr)
    dataset <- as.data.frame(dataset)
    dataset$Subject <- paste(subject,"Midterm2",sep='_')
    data <- rbind(data, dataset)
    
    subject_file <- stresshelpers::read.empatica(paste(folder, "/",subject,"/Final",sep=""))
    hr <- subject_file$signal$hr$data
    eda <- subject_file$signal$eda$data
    eda <- eda[2:(length(eda)-1)]
    hr <- hr[2:(length(hr)-1)]
    eda <- stresshelpers::downsample(eda, round(length(eda) / length(hr)))
    shortest <- min(length(eda), length(hr))
    hr <- hr[1:shortest]
    eda <- eda[1:shortest]
    dataset <- cbind(eda, hr)
    dataset <- as.data.frame(dataset)
    dataset$Subject <- paste(subject,"Final",sep='_')
    data <- rbind(data, dataset)
  }
  data$metric <- 1
  return (data)
}


#########################################################################################################################################################
# Generate 3 minute blocks of stressed/non-stressed samples
#########################################################################################################################################################
data_neuro <- stresshelpers::make_neuro_data('NEURO', feature_engineering = FALSE)
data_swell <- stresshelpers::make_swell_data('SWELL', feature_engineering = FALSE)
data_wesad <- stresshelpers::make_wesad_data('WESAD', feature_engineering = FALSE)
data_ubfc  <- stresshelpers::make_ubfc_data('UBFC',  feature_engineering = FALSE)
data_exam  <- make_exam_data('EXAM')

data <- rbind(data_neuro, data_swell, data_wesad, data_ubfc, data_exam)
data <- data %>% select(hr, eda, Subject, metric)

rm(data_neuro, data_swell, data_wesad, data_ubfc, data_exam)
gc()

# split subjects into stressed/non-stressed
split_subject_data <- function(data)
{
  subject <- unique(data$Subject)
  data_list <- split(data, f = data$metric)
  index <- 1
  merged <- NULL
  for (df in data_list)
  {
    df$Subject <- paste(subject, '_', index, sep='')
    merged <- rbind(merged, df)
    index <- index + 1
  }
  return (merged)
}

split_subjects <- NULL
subjects <- unique(data$Subject)
for (subject in subjects)
{
  subject_data <- data[data$Subject == subject,]
  split_data <- split_subject_data(subject_data)
  split_subjects <- rbind(split_subjects, split_data)
}
data <- split_subjects

rm(split_data, split_subjects, subject_data, subject, subjects)
gc()

# split these blocks into 3-min samples
subjects <- unique(data$Subject)
newdata <- NULL
counter <- 1
for (subject in subjects)
{
  temp <- data[data$Subject == subject,]
  metric <- unique(temp$metric)
  size <- nrow(temp)
  n <- 180
  nr <- nrow(temp)
  subsets <- split(temp, rep(1:ceiling(nr/n), each=n, length.out=nr))
  for (subset in subsets)
  {
    if (nrow(subset) == 180)
    {
      subset$Subject <- paste('X_',metric, '_', counter,sep='')
      newdata <- rbind(newdata, subset)
      counter <- counter + 1
    }
  }
}
data <- newdata

rm(newdata, subset, subsets, temp, counter, metric, n, nr, size, subject, subjects)
gc()

# now we have x groupings of 3 minute samples of stressed or non-stressed
# break it all into 2 groups: stressed and non-stressed
stressed <- data[data$metric==1,]
nonstressed <- data[data$metric==0,]

stressed_subjects <- unique(stressed$Subject)
nonstressed_subjects <- unique(nonstressed$Subject)

rm(data)
gc()

newdata <- NULL
synthesize <- function(subject_count)
{
  # 6 mins of stress, then 6 mins of no-stress
  for (index in seq(1:subject_count))
  {
    sample_stressed <- sample(stressed_subjects, 4)
    sample_nonstressed <- sample(nonstressed_subjects, 4)
    temp <- NULL
    temp <- rbind(temp, nonstressed[nonstressed$Subject==sample_nonstressed[[1]],])
    temp <- rbind(temp, nonstressed[nonstressed$Subject==sample_nonstressed[[2]],])
    temp <- rbind(temp, nonstressed[nonstressed$Subject==sample_nonstressed[[3]],])
    temp <- rbind(temp, nonstressed[nonstressed$Subject==sample_nonstressed[[4]],])
    temp <- rbind(temp, stressed[stressed$Subject==sample_stressed[[1]],])
    temp <- rbind(temp, stressed[stressed$Subject==sample_stressed[[2]],])
    temp <- rbind(temp, stressed[stressed$Subject==sample_stressed[[3]],])
    temp <- rbind(temp, stressed[stressed$Subject==sample_stressed[[4]],])

    temp$Subject <- paste('X', index, sep='')
    newdata <- rbind(newdata, temp)
  }
  return (newdata)
}

data <- synthesize(200)
data <- stresshelpers::rolling_features(data, 25)
data <- data %>% select(hrrange, hrvar, hrstd, hrmin, edarange, edastd, edavar, hrkurt, edamin, hrmax, Subject, metric)

rm(nonstressed, stressed, newdata, nonstressed_subjects, stressed_subjects)
gc()

#########################################################################################################################################################
# Model training - xgboost using optimal parameters with LOSO
#########################################################################################################################################################
subjects <- unique(data$Subject) # 200
index <- 1
results <- NULL

weighted <- function(xgb, ann) (xgb*0.5) + (ann*0.5)

# found using hyper parameter search
params <- list(
  eta = 0.5, 
  max_depth = 8, 
  subsample = 0.70,
  colsample_bytree = 0.8
)

for (subject in subjects)
{
  print(index)
  val <- data[data$Subject == subject,]
  temp <- data[!(data$Subject == subject),]
  
  train.index <- createDataPartition(temp$Subject, p = .7, list = FALSE) # 70/30 train/test split along metric
  train <- temp[train.index,]
  test <- temp[-train.index,]
  
  dtrain <- xgb.DMatrix(data = as.matrix(train[,1:10]), label = train$metric)
  dtest <- xgb.DMatrix(data = as.matrix(test[,1:10]), label = test$metric)
  watchlist <- list(train = dtrain, test = dtest)
  
  model_xgb <- xgb.train(
    params = params,
    data = dtrain,
    objective = "reg:logistic",
    watchlist = watchlist,
    nrounds = 500,
    early_stopping_rounds = 3,
    verbose = 0
  )
  
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
      units              = 4, 
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
    callbacks        = list(callback_early_stopping(monitor = "val_loss", patience = 3, restore_best_weights = TRUE)),
    verbose          = 0
  )
  
  x_val <- val[,1:10]
  yhat_xgb <- predict(model_xgb, as.matrix(x_val))
  x_val <- scale(x_val, center = attr(x_train, "scaled:center") , scale = attr(x_train, "scaled:scale"))
  yhat_nn <- as.data.frame(predict(model_nn, x_val))
  yhat_nn <- yhat_nn[,1]
  yhat_nn <- (yhat_nn - min(yhat_nn)) / (max(yhat_nn) - min(yhat_nn))
  yhat_ens <- weighted(yhat_xgb, yhat_nn)
  yhat_xgb <- round(yhat_xgb)
  yhat_nn <- round(yhat_nn)
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
  write.csv(results, "Results.csv",row.names = FALSE)
  index <- index + 1
}

results$XGB <- as.numeric(results$XGB)
results$ANN <- as.numeric(results$ANN)
results$ENS <- as.numeric(results$ENS)
results$PRECISION <- as.numeric(results$PRECISION)
results$RECALL <- as.numeric(results$RECALL)
results$F1 <- as.numeric(results$F1)

print(mean(results$XGB, na.rm=TRUE)) # 0.8672
print(mean(results$ANN, na.rm=TRUE)) # 0.8658
print(mean(results$ENS, na.rm=TRUE)) # 0.8710
print(mean(results$PRECISION, na.rm=TRUE)) # 0.89
print(mean(results$RECALL, na.rm=TRUE)) # 0.85
print(mean(results$F1, na.rm=TRUE)) # 0.86

#########################################################################################################################################################
# Test
#########################################################################################################################################################

temp <- data[data$Subject=='X59',]
x_val <- temp[,1:10]
yhat_xgb <- predict(model_xgb, as.matrix(x_val))
x_val <- scale(x_val, center = attr(x_train, "scaled:center") , scale = attr(x_train, "scaled:scale"))
yhat_nn <- as.data.frame(predict(model_nn, x_val))
yhat_nn <- yhat_nn[,1]
yhat_nn <- (yhat_nn - min(yhat_nn)) / (max(yhat_nn) - min(yhat_nn))
yhat_ens <- weighted(yhat_xgb, yhat_nn)
temp <- cbind(yhat_xgb, yhat_nn, yhat_ens, temp$metric)
temp <- as.data.frame(temp)
names(temp) <- c("xgb","ann","ens","metric")
temp$ID <- seq.int(nrow(temp))
ggplot(temp, aes(x=ID)) + 
  geom_area(aes(y = metric, colour="STRESS"), fill="#FF6666",  alpha=0.4, size=0.5) + 
  geom_line(aes(y = runMean(ens, 120) , colour="ENS"), size=0.5) + 
  scale_color_lancet() + scale_fill_lancet() +
  labs(colour="Model") + 
  guides(color = guide_legend(override.aes = list(fill="white", size=5))) + 
  theme_classic() + ylab('Synthetic - X59') + xlab('Time (seconds)') + 
  scale_x_continuous(breaks=seq(0,nrow(temp)+120,120)) +
  theme(axis.title = element_text(size = 22, family="Times New Roman",face="bold")) +
  theme(axis.text=element_text(size=18, family="Times New Roman",face="bold")) +
  theme(plot.title = element_text(family="Times New Roman",face="bold")) +
  theme(legend.text = element_text(family="Times New Roman",face="bold", size=18)) +
  theme(legend.title =  element_text(family="Times New Roman",face="bold", size=18)) +
  theme(
    axis.title.y = element_text(vjust = +1),
    axis.title.x = element_text(vjust = -0.8)
  ) 

temp <- data[data$Subject=='X200',]
x_val <- temp[,1:10]
yhat_xgb <- predict(model_xgb, as.matrix(x_val))
x_val <- scale(x_val, center = attr(x_train, "scaled:center") , scale = attr(x_train, "scaled:scale"))
yhat_nn <- as.data.frame(predict(model_nn, x_val))
yhat_nn <- yhat_nn[,1]
yhat_nn <- (yhat_nn - min(yhat_nn)) / (max(yhat_nn) - min(yhat_nn))
yhat_ens <- weighted(yhat_xgb, yhat_nn)
temp <- cbind(yhat_xgb, yhat_nn, yhat_ens, temp$metric)
temp <- as.data.frame(temp)
names(temp) <- c("xgb","ann","ens","metric")
temp$ID <- seq.int(nrow(temp))
ggplot(temp, aes(x=ID)) + 
  geom_area(aes(y = metric, colour="STRESS"), fill="#FF6666",  alpha=0.4, size=0.5) + 
  geom_line(aes(y = runMean(ens, 60) , colour="ENS"), size=0.5) + 
  scale_color_lancet() + scale_fill_lancet() +
  labs(colour="Model") + 
  guides(color = guide_legend(override.aes = list(fill="white", size=5))) + 
  theme_classic() + ylab('Syntheic - X200') + xlab('Time (seconds)') + 
  scale_x_continuous(breaks=seq(0,nrow(temp)+120,120)) +
  theme(axis.title = element_text(size = 22, family="Times New Roman",face="bold")) +
  theme(axis.text=element_text(size=18, family="Times New Roman",face="bold")) +
  theme(plot.title = element_text(family="Times New Roman",face="bold")) +
  theme(legend.text = element_text(family="Times New Roman",face="bold", size=18)) +
  theme(legend.title =  element_text(family="Times New Roman",face="bold", size=18)) +
  theme(
    axis.title.y = element_text(vjust = +1),
    axis.title.x = element_text(vjust = -0.8)
  ) 

