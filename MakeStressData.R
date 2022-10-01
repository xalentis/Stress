# Generate StressData

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

#devtools::install_github("xalentis/Stress/stresshelpers", force=TRUE)
library(stresshelpers)
library(dplyr)

options(scipen=999)

data_neuro <- stresshelpers::make_neuro_data('NEURO', feature_engineering = TRUE)
data_swell <- stresshelpers::make_swell_data('SWELL', feature_engineering = TRUE)
data_wesad <- stresshelpers::make_wesad_data('WESAD', feature_engineering = TRUE)
data_ubfc  <- stresshelpers::make_ubfc_data('UBFC', feature_engineering = TRUE)

data_final <- rbind(data_neuro, data_swell, data_wesad, data_ubfc) # 99 subjects, 242760 observations
data_final <- data_final %>% select(hrrange, hrvar, hrstd, hrmin, edarange, edastd, edavar, hrkurt, edamin, hrmax, Subject, metric)

rm(data_neuro, data_swell, data_wesad, data_ubfc)
gc()

write.csv(data_final, "StressData.csv", row.names = FALSE, quote = FALSE)
