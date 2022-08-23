# Generate SynthesizedStressData

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

#devtools::install_github("gideonvosjcu/stresshelpers", force=TRUE)
library(stresshelpers)
library(dplyr)

options(scipen=999)

#########################################################################################################################################################
# Generate 3 minute blocks of stressed/non-stressed samples
#########################################################################################################################################################
data_neuro <- stresshelpers::make_neuro_data('NEURO', feature_engineering = FALSE)
data_swell <- stresshelpers::make_swell_data('SWELL', feature_engineering = FALSE)
data_wesad <- stresshelpers::make_wesad_data('WESAD', feature_engineering = FALSE)
data_ubfc  <- stresshelpers::make_ubfc_data('UBFC',  feature_engineering = FALSE)
data_exam  <- make_exam_data('EXAM')
data_exam <- data_exam %>% select(eda, hr, Subject)
data_exam$metric <- 1

data <- rbind(data_neuro, data_swell, data_wesad, data_ubfc, data_exam)
data <- data %>% select(hr, eda, Subject, metric)

rm(data_neuro, data_swell, data_ubfc, data_exam, data_wesad)
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
  # 12mins of no-stress, then 12 mins of stress
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

Q <- quantile(data$eda, probs=c(.25, .75), na.rm = FALSE)
iqr <- IQR(data$eda)
up <-  Q[2]+1.5*iqr # Upper Range  
low<- Q[1]-1.5*iqr # Lower Range﻿
data <- subset(data, data$eda > (Q[1] - 1.5*iqr) & data$eda < (Q[2]+1.5*iqr))

Q <- quantile(data$hr, probs=c(.25, .75), na.rm = FALSE)
iqr <- IQR(data$hr)
up <-  Q[2]+1.5*iqr # Upper Range  
low<- Q[1]-1.5*iqr # Lower Range﻿
data <- subset(data, data$hr > (Q[1] - 1.5*iqr) & data$hr < (Q[2]+1.5*iqr))

data <- stresshelpers::rolling_features(data, 25)
data <- data %>% select(hrrange, hrvar, hrstd, hrmin, edarange, edastd, edavar, hrkurt, edamin, hrmax, Subject, metric)
gc()

write.csv(data, "SynthesizedStressData.csv", row.names = FALSE, quote = FALSE)
