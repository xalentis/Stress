# Exploratory Data Analysis

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
library(ggplot2)
library(corrplot)
library(WebPower)
library(summarytools)
library(e1071)
library(ggsci)
library(ggpubr)
#devtools::install_github("gideonvosjcu/stresshelpers", force=TRUE)
library(stresshelpers)

options(scipen=999)

#########################################################################################################################################################
# Explore Part 1 - No Log Transform, NO FEATURE ENGINEERING
#########################################################################################################################################################
data_swell <- stresshelpers::make_swell_data('SWELL', feature_engineering = FALSE)   # 157739 observations, 9 subjects
data_wesad <- stresshelpers::make_wesad_data('WESAD', feature_engineering = FALSE)   # 26385 observations, 14 subjects
data_neuro <- stresshelpers::make_neuro_data('NEURO', feature_engineering = FALSE)   # 20060 observations, 20 subjects
data_ubfc <-  stresshelpers::make_ubfc_data('UBFC', feature_engineering = FALSE)     # 40215 observations, 54 subjects

# subject summary
data_subjects <- data.frame()
data_subjects <- rbind(data_subjects, c('NEURO',length(unique(data_neuro$Subject))))
data_subjects <- rbind(data_subjects, c('SWELL',length(unique(data_swell$Subject))))
data_subjects <- rbind(data_subjects, c('WESAD',length(unique(data_wesad$Subject))))
data_subjects <- rbind(data_subjects, c('UBFC',length(unique(data_ubfc$Subject))))
names(data_subjects) <- c("Dataset", "Subjects")
data_subjects$Subjects <- as.numeric(data_subjects$Subjects)
ggplot(data = data_subjects, aes(Dataset)) + 
  geom_col(aes(y = Subjects, fill = Dataset), colour = "black") + 
  scale_fill_nejm() + theme_light() + 
  labs(y = "Number of subjects in each dataset\n") + 
  labs(x = "\nDataset") +
  theme(legend.position="none") +
  theme(axis.title = element_text(size = 20, family="Times New Roman",face="bold")) +
  theme(axis.text=element_text(size=14, family="Times New Roman",face="bold"))

# histograms
hist_neuro_eda <- ggplot(data_neuro, aes(x=eda)) + 
  geom_histogram(aes(y=..density..), colour="black", fill="white")+
  scale_x_continuous(lim = c(0, 14)) + 
  scale_y_continuous(lim = c(0, 2.5)) + 
  geom_density(alpha=.2, fill="skyblue")  + labs(y="") + labs(x="EDA") + theme_classic() +
  theme(axis.title = element_text(size = 20, family="Times New Roman",face="bold")) +
  theme(axis.text=element_text(size=12, family="Times New Roman",face="bold"))

hist_swell_eda <- ggplot(data_swell, aes(x=eda)) + 
  geom_histogram(aes(y=..density..), colour="black", fill="white")+
  scale_x_continuous(lim = c(0, 14)) + 
  scale_y_continuous(lim = c(0, 2.5)) + 
  geom_density(alpha=.2, fill="skyblue")  + labs(y="") + labs(x="EDA") + theme_classic() +
  theme(axis.title = element_text(size = 20, family="Times New Roman",face="bold")) +
  theme(axis.text=element_text(size=12, family="Times New Roman",face="bold"))

hist_ubfc_eda <- ggplot(data_ubfc, aes(x=eda)) + 
  geom_histogram(aes(y=..density..), colour="black", fill="white")+
  scale_x_continuous(lim = c(0, 14)) + 
  scale_y_continuous(lim = c(0, 2.5)) + 
  geom_density(alpha=.2, fill="skyblue")  + labs(y="") + labs(x="EDA") + theme_classic() +
  theme(axis.title = element_text(size = 20, family="Times New Roman",face="bold")) +
  theme(axis.text=element_text(size=12, family="Times New Roman",face="bold"))

hist_wesad_eda <- ggplot(data_wesad, aes(x=eda)) + 
  geom_histogram(aes(y=..density..), colour="black", fill="white")+
  scale_x_continuous(lim = c(0, 14)) + 
  scale_y_continuous(lim = c(0, 2.5)) + 
  geom_density(alpha=.2, fill="skyblue")  + labs(y="") + labs(x="EDA") + theme_classic() +
  theme(axis.title = element_text(size = 20, family="Times New Roman",face="bold")) +
  theme(axis.text=element_text(size=12, family="Times New Roman",face="bold"))

ggarrange(hist_neuro_eda, hist_swell_eda, hist_ubfc_eda, hist_wesad_eda, ncol =4, nrow = 1)


hist_neuro_hr <- ggplot(data_neuro, aes(x=hr)) + 
  geom_histogram(aes(y=..density..), colour="black", fill="white")+
  scale_x_continuous(lim = c(40, 180)) + 
  geom_density(alpha=.2, fill="skyblue")  + labs(y="") + labs(x="HR") + theme_classic() +
  theme(axis.title = element_text(size = 20, family="Times New Roman",face="bold")) +
  theme(axis.text=element_text(size=12, family="Times New Roman",face="bold"))

hist_swell_hr <- ggplot(data_swell, aes(x=hr)) + 
  geom_histogram(aes(y=..density..), colour="black", fill="white")+
  scale_x_continuous(lim = c(40, 180)) + 
  geom_density(alpha=.2, fill="skyblue")  + labs(y="") + labs(x="HR") + theme_classic() +
  theme(axis.title = element_text(size = 20, family="Times New Roman",face="bold")) +
  theme(axis.text=element_text(size=12, family="Times New Roman",face="bold"))

hist_ubfc_hr <- ggplot(data_ubfc, aes(x=hr)) + 
  geom_histogram(aes(y=..density..), colour="black", fill="white")+
  scale_x_continuous(lim = c(40, 180)) + 
  geom_density(alpha=.2, fill="skyblue")  + labs(y="") + labs(x="HR") + theme_classic() +
  theme(axis.title = element_text(size = 20, family="Times New Roman",face="bold")) +
  theme(axis.text=element_text(size=12, family="Times New Roman",face="bold"))

hist_wesad_hr <- ggplot(data_wesad, aes(x=hr)) + 
  geom_histogram(aes(y=..density..), colour="black", fill="white")+
  scale_x_continuous(lim = c(40, 180)) + 
  geom_density(alpha=.2, fill="skyblue")  + labs(y="") + labs(x="HR") + theme_classic() +
  theme(axis.title = element_text(size = 20, family="Times New Roman",face="bold")) +
  theme(axis.text=element_text(size=12, family="Times New Roman",face="bold"))

ggarrange(hist_neuro_hr, hist_swell_hr, hist_ubfc_hr, hist_wesad_hr, ncol =4, nrow = 1)

# correlation
names(data_neuro) <- c("EDA","HR","STRESS","Subject")
names(data_ubfc) <- c("EDA","HR","STRESS","Subject")
names(data_swell) <- c("EDA","HR","STRESS","Subject")
names(data_wesad) <- c("EDA","HR","STRESS","Subject")
col <- colorRampPalette(c("red", "white","navy"))(10)

par(family="Times New Roman")
corrplot(cor(data_swell[,c("EDA","HR","STRESS")], method="spearman"), method="color", title="",addCoef.col = 'black', number.cex = 1.7,tl.cex=1.5,tl.col = "black", col=col, cl.cex=1.5, cl.ratio = 0.5)
corrplot(cor(data_wesad[,c("EDA","HR","STRESS")], method="spearman"), method="color", title="",addCoef.col = 'black', number.cex = 1.7,tl.cex=1.5,tl.col = "black", col=col, cl.cex=1.5, cl.ratio = 0.5)
corrplot(cor(data_ubfc[,c("EDA","HR","STRESS")], method="spearman"), method="color", title="",addCoef.col = 'black', number.cex = 1.7,tl.cex=1.5,tl.col = "black", col=col, cl.cex=1.5, cl.ratio = 0.5)
corrplot(cor(data_neuro[,c("EDA","HR","STRESS")], method="spearman"), method="color", title="",addCoef.col = 'black', number.cex = 1.7,tl.cex=1.5,tl.col = "black", col=col, cl.cex=1.5, cl.ratio = 0.5)

# merge and box plots
data <- rbind(data_neuro, data_swell, data_wesad, data_ubfc)
data$Dataset <- substr(data$Subject,1,1)
data$Group <- paste(substr(data$Subject,1,1), '\n',substr(data$Subject,2,3),sep='')
data <- data %>% arrange(Subject)
data %>%
  ggplot(aes(x=Dataset, y=EDA, fill=Dataset)) +
  geom_boxplot() +
  scale_fill_nejm() + 
  theme_light() +
  labs(y = "EDA\n") + 
  labs(x = "\nSubjects") +
  theme(legend.position="none") +
  theme(axis.title = element_text(size = 20, family="Times New Roman",face="bold")) +
  theme(axis.text=element_text(size=12, family="Times New Roman",face="bold")) +
  scale_x_discrete(breaks=data$Dataset,label=data$Dataset)

data %>%
  ggplot(aes(x=Dataset, y=HR, fill=Dataset)) +
  geom_boxplot() +
  scale_fill_nejm() + 
  theme_light() +
  labs(y = "HR\n") + 
  labs(x = "\nSubjects") +
  theme(legend.position="none") +
  theme(axis.title = element_text(size = 20, family="Times New Roman",face="bold")) +
  theme(axis.text=element_text(size=12, family="Times New Roman",face="bold")) +
  scale_x_discrete(breaks=data$Dataset,label=data$Dataset)

# metric class balance
hist(data$STRESS, xlab = "Metric", col = "skyblue")

# Power Analysis
# We want to examine the relationship between stress and eda, hr.
# Based on plots correlation is around 0.30 on average.
# We have a total of 99 subjects, so what is the power?
wp.correlation(n=99, r=0.3) # 86% power
power <- wp.correlation(n=seq(50,150,10), r=0.30, alternative = "two.sided") 
plot(power, type='b') # we will need at least 85 subjects or more

#########################################################################################################################################################
# Explore Part 2 - WITH FEATURE ENGINEERING (rolling windows of 25 seconds, log-transform of eda and hr)
#########################################################################################################################################################
neuro_expanded <- stresshelpers::make_neuro_data('NEURO', feature_engineering = TRUE)
swell_expanded <- stresshelpers::make_swell_data('SWELL', feature_engineering = TRUE)
wesad_expanded <- stresshelpers::make_wesad_data('WESAD', feature_engineering = TRUE)
ubfc_expanded <-  stresshelpers::make_ubfc_data('UBFC', feature_engineering = TRUE)

neuro_expanded$Subject<- NULL
ubfc_expanded$Subject<- NULL
swell_expanded$Subject<- NULL
wesad_expanded$Subject<- NULL
corrplot(cor(neuro_expanded, method="spearman"), method="color", title="",addCoef.col = 'black', number.cex = 1,tl.cex=1,tl.col = "black", col=col, cl.cex=1, cl.ratio = 0.5)
corrplot(cor(ubfc_expanded, method="spearman"), method="color", title="",addCoef.col = 'black', number.cex = 1,tl.cex=1,tl.col = "black", col=col, cl.cex=1, cl.ratio = 0.5)
corrplot(cor(swell_expanded, method="spearman"), method="color", title="",addCoef.col = 'black', number.cex = 1,tl.cex=1,tl.col = "black", col=col, cl.cex=1, cl.ratio = 0.5)
corrplot(cor(wesad_expanded, method="spearman"), method="color", title="",addCoef.col = 'black', number.cex = 1,tl.cex=1,tl.col = "black", col=col, cl.cex=1, cl.ratio = 0.5)

# reload data for boxplots to fix names
neuro_expanded <- stresshelpers::make_neuro_data('NEURO', feature_engineering = TRUE)
swell_expanded <- stresshelpers::make_swell_data('SWELL', feature_engineering = TRUE)
wesad_expanded <- stresshelpers::make_wesad_data('WESAD', feature_engineering = TRUE)
ubfc_expanded <-  stresshelpers::make_ubfc_data('UBFC', feature_engineering = TRUE)

# merge and box plots
data <- rbind(neuro_expanded, swell_expanded, wesad_expanded, ubfc_expanded)
data$Dataset <- substr(data$Subject,1,1)
data$Group <- paste(substr(data$Subject,1,1), '\n',substr(data$Subject,2,3),sep='')
data <- data %>% arrange(Subject)
data %>%
  ggplot(aes(x=Subject, y=eda, fill=Dataset)) +
  geom_boxplot() +
  scale_fill_nejm() + 
  theme_light() +
  labs(y = "EDA\n") + 
  labs(x = "\nSubjects") +
  theme(legend.position="none") +
  theme(axis.title = element_text(size = 20, family="Times New Roman",face="bold")) +
  theme(axis.text=element_text(size=12, family="Times New Roman",face="bold")) +
  scale_x_discrete(breaks=data$Subject,label=data$Group)

data %>%
  ggplot(aes(x=Subject, y=hr, fill=Dataset)) +
  geom_boxplot() +
  scale_fill_nejm() + 
  theme_light() +
  labs(y = "HR\n") + 
  labs(x = "\nSubjects") +
  theme(legend.position="none") +
  theme(axis.title = element_text(size = 20, family="Times New Roman",face="bold")) +
  theme(axis.text=element_text(size=12, family="Times New Roman",face="bold")) +
  scale_x_discrete(breaks=data$Subject,label=data$Group)

# remove outliers at 25th and 75th percentile of eda and hr signals
Q <- quantile(data$eda, probs=c(.25, .75), na.rm = FALSE)
iqr <- IQR(data$eda)
data<- subset(data, data$eda > (Q[1] - 1.5*iqr) & data$eda < (Q[2]+1.5*iqr))
Q <- quantile(data$hr, probs=c(.25, .75), na.rm = FALSE)
iqr <- IQR(data$hr)
data<- subset(data, data$hr > (Q[1] - 1.5*iqr) & data$hr < (Q[2]+1.5*iqr))

# Compute the analysis of variance
res.aov <- aov(edarange + hrmax ~ metric, data = data)
summary(res.aov) # p-value = 0.339

# correlation across merged data sets
corrplot(cor(data[,1:22], method="spearman"), method="color", title="",addCoef.col = 'black', number.cex = 0.8,tl.cex=1.5,tl.col = "black", col=col, cl.cex=1.5, cl.ratio = 0.5)
# correlation in merged data is now around 0.20
wp.correlation(n=99, r=0.30) # that gives us 86.17% power if we split across subjects

#########################################################################################################################################################

