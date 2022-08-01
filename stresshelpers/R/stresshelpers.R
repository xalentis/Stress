# Investigating Wearable Sensor Biomarkers for Chronic Stress Measurement and Analysis
# Gideon Vos, Master of Philosophy, James Cook University
# 14 September 2021

# Citations:

# WESAD (Wearable Stress and Affect Detection)
# Philip Schmidt, Attila Reiss, Robert Duerichen, Claus Marberger, and Kristof Van Laerhoven. 2018.
# Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection.
# In Proceedings of the 20th ACM International Conference on Multimodal Interaction (ICMI '18).
# Association for Computing Machinery, New York, NY, USA, 400–408. DOI:https://doi.org/10.1145/3242969.3242985

# AffectiveROAD:
# Neska El Haouij, Jean-Michel Poggi, Sylvie Sevestre-Ghalila, Raja Ghozi, and Mériem Jaïdane.
# 2018. AffectiveROAD system and database to assess driver's attention. In Proceedings of the 33rd Annual ACM Symposium on Applied Computing (SAC '18).
# ACM, New York, NY, USA, 800-803. DOI: https://doi.org/10.1145/3167132.3167395. https://dam-prod.media.mit.edu/x/2021/06/14/AffectiveROAD_Data_w1dqSB9.zip

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
# Amin, M. R., Wickramasuriya, D., & Faghih, R. T. (2022). A Wearable Exam Stress Dataset for Predicting Cognitive Performance in Real-World Settings (version 1.0.0). PhysioNet. https://doi.org/10.13026/kvkb-aj90.

#########################################################################################################################################################
# Empatica E4 data reader - courtesy of https://github.com/bwrc/empatica-r
#########################################################################################################################################################
organise_data <- function(data, samplingrate = NULL) {
  out <- vector(mode = "list", length = ncol(data))
  if (!is.null(samplingrate))
  {
    time_vector <- seq.int(0, (nrow(data) - 1)) / samplingrate
  }
  else
  {
    time_vector <- NULL
  }
  for (i in seq.int(ncol(data))) {
    signal_name <- colnames(data)[i]
    out[[i]]$data <- data[,i]
    out[[i]]$t <- time_vector
    out[[i]]$samplingrate <- samplingrate
  }
  names(out) <- colnames(data)
  return (out)
}

read_empatica_ibi <- function(filename, signal_names = c("time", "ibi")) {
  t_start <- scan(filename, nmax = 1, what = numeric(), nlines = 1, skip = 0, sep = ",", quiet = TRUE)
  data <- read.csv(filename, header = FALSE, sep = ",", skip = 1)
  if (!is.null(signal_names))
  {
    colnames(data) <- signal_names
  }
  out <- organise_data(data)
  out[["ibi"]][["t"]] <- out[["time"]][["data"]]
  out[["ibi"]][["samplingrate"]] <- NULL
  out[["time"]] <- NULL
  return(out)
}

read_header <- function(filename) {
  scan(filename, nmax = 1, what = numeric(), nlines = 1, skip = 0, sep = ",", quiet = TRUE)
}

read_empatica_events <- function(f) {
  tmp <- scan(f, quiet = TRUE)
  if (length(tmp) > 0)
  {
    return (data.frame("id" = seq.int(length(tmp)), "time_raw"  = tmp,
                       "timestamp" = as.POSIXct(tmp, tz = "GMT", origin = "1970-01-01"), "timedelta" = 0))
  }
  else
  {
    return (data.frame("id" = numeric(), "time_raw" = numeric(), "timestamp" = as.POSIXct(character()), "timedelta" = numeric()))
  }
}

read_empatica_gen <- function(filename, signal_names = NULL) {
  t_start <- scan(filename, nmax = 1, what = numeric(), nlines = 1, skip = 0, sep = ",", quiet = TRUE)
  samplingrate <- scan(filename, nmax = 1, what = numeric(), nlines = 1, skip = 1, sep = ",", quiet = TRUE)
  data <- read.csv(filename, header = FALSE, sep = ",", skip = 2)

  if (!is.null(signal_names))
  {
    colnames(data) <- signal_names
  }
  return (organise_data(data, samplingrate))
}

new_recording <- function() {
  recording <- list()
  recording$properties <- list()
  recording$signal <- list()
  recording$events <- list()
  recording$properties$header <- list()
  recording$properties$time.start.raw <- NA
  recording$properties$time.start <- NA
  recording$properties$time.stop.raw <- NA
  recording$properties$time.stop <- NA
  recording$properties$subject <- NA
  recording$properties$format <- NA
  recording$properties$format.long <- NA
  recording$properties$device.type <- NA
  recording$properties$device.version <- NA
  recording$properties$length <- NA
  return (recording)
}

#' Reads an Empatica E4 folder of data into a data frame
#'
#' @param exdir The folder containing the E4 data
#' @param header.only Header only, T or F
#' @return A data frame containing the data
#' @export
read.empatica <- function(exdir, header.only = FALSE) {
  recording <- new_recording()
  filelist <- list.files(exdir, pattern = "*.csv", full.names = TRUE)

  out <- list()
  for (f in filelist) {
    signal_name <- tolower(gsub(".csv", "", basename(f)))
    signal_names <- NULL
    if (signal_name == "acc")
      signal_names <- c("acc_x", "acc_y", "acc_z")
    if (signal_name == "bvp")
      signal_names <- c("bvp")
    if (signal_name == "eda")
      signal_names <- c("eda")
    if (signal_name == "hr")
      signal_names <- c("hr")
    if (signal_name == "temp")
      signal_names <- c("temp")
    if (signal_name == "ibi") {
      signal_names <- c("ibi")
      out <- c(out, read_empatica_ibi(f))
      signal_names <- NULL
    }
    if (signal_name == "tags") {
      recording$events <- read_empatica_events(f)
      signal_names <- NULL
    }

    if (! is.null(signal_names))
      out <- c(out, read_empatica_gen(f, signal_names))
  }
  header <- read_header(filelist[[1]])
  recording$signal <- out
  if (length(names(recording$signal)) > 1)
  {
    recording$properties$length <- rev(recording$signal[[names(recording$signal)[1]]]$t)[1]
  }
  else
  {
    recording$properties$length <- 0
  }
  recording$properties$time.start.raw <- header
  recording$properties$time.start <- as.POSIXct(header, tz = "GMT", origin = "1970-01-01")
  recording$properties$time.stop.raw <- header + recording$properties$length
  recording$properties$time.stop <- as.POSIXct(header, tz = "GMT", origin = "1970-01-01") + as.difftime(recording$properties$length, units = "secs")
  if (length(recording$events) > 0)
    recording$events$timedelta <- as.numeric(difftime(recording$events$timestamp, recording$properties$time.start, units = "secs"))
  recording$properties$device.type <- "Empatica"
  recording$properties$device.version <- "E4"
  return (recording)
}

#########################################################################################################################################################
# PhysioNet file reader - courtesy of https://rdrr.io/github/Absox/wfdb.R/f/
#########################################################################################################################################################
wfdbdesc <- function(header.filename, read.datetime = TRUE) {
  # Get root directory of header file, as we may need to read other files using that path
  root.dir <- ""
  if (grepl("/", header.filename)) {
    root.dir <- substr(header.filename, 1, regexpr("/[^/]*$", header.filename))
    header.filename <- substr(header.filename, regexpr("/[^/]*$", header.filename)+1, nchar(header.filename))
  }
  # If filename lacks extension, add extension
  if (!grepl(".hea$", header.filename)) {
    header.filename <- sprintf("%s.hea", header.filename)
  }

  # Read in header file for parsing
  header.filepath <- sprintf("%s%s", root.dir, header.filename)
  header.file <- file(header.filepath)
  header.data <- trimws(strsplit(readChar(header.file,file.size(header.filepath)),"\n")[[1]])
  close(header.file)

  # Process the first line of the header (record line):
  record.line <- strsplit(header.data[1]," ")[[1]]
  # Determine what type of record this is: single segment or multi-segment, number of signals, sampling frequency,
  # and record length
  record.name <- record.line[1]
  num.signals <- as.numeric(record.line[2])
  if (grepl("/", record.line[3])) {
    sample.frequency <- as.numeric(substr(record.line[3],1,regexpr("/",record.line[3])-1))
  } else {
    sample.frequency <- as.numeric(record.line[3])
  }
  record.length <- as.numeric(record.line[4])

  # Single segment datetimes
  if (read.datetime) {
    base.datetime <- lubridate::dmy(record.line[6]) + lubridate::hms(record.line[5])
  } else {
    base.datetime <- NA
  }

  if (grepl("/", record.name)) {
    # Multi segment file
    record.type <- "multi"
    num.segments <- as.numeric(substr(record.name, regexpr("/",record.name)+1,nchar(record.name)))
    record.name <- substr(record.name, 1, regexpr("/",record.name)-1)
    # Read segment names and segment lengths
    segments <- strsplit(header.data[2:(num.segments+1)], " ")
    segment.lengths <- sapply(segments, function(x) as.numeric(x[2]))
    segment.names <- sapply(segments, function(x) x[1])

    # From first valid segment, read descriptions, adc gains, and adc zeros
    first.segment <- segment.names[min(which(segment.names != "~"))]
    layout.filepath <- sprintf("%s%s.hea", root.dir, first.segment)
    layout.file <- file(layout.filepath)
    layout.data <- trimws(strsplit(readChar(layout.file, file.size(layout.filepath)), "\n")[[1]])
    close(layout.file)

    signal.lines <- strsplit(layout.data[2:(num.signals+1)], " ") # TODO
    signal.format <- NULL
  } else {
    # Single segment, can proceed to read signal specification from current file
    record.type <- "single"
    signal.lines <- strsplit(header.data[2:(num.signals+1)], " ")
    segment.names <- signal.lines[[1]][1];
    segment.lengths <- record.length
    signal.format <- as.numeric(signal.lines[[1]][2])
  }

  # Read signal descriptions, adc gains, units of measure, and zeros from signal specification
  descriptions <- sapply(signal.lines, function(x) {
    if (length(x) > 9) {
      return(paste(x[-(1:8)],collapse=" "))
    } else {
      return(x[9])
    }
  })
  # This extra code determines if there is a baseline
  adc.gains <- sapply(signal.lines, function(x) {
    if (grepl("\\(.*\\)", x[3])) {
      return(as.numeric(substr(x[3], 1, regexpr("\\(", x[3])-1)))
    } else {
      return(as.numeric(substr(x[3], 1, regexpr("/", x[3])-1)))
    }

  })
  adc.units <- sapply(signal.lines, function(x) substr(x[3], regexpr("/", x[3])+1, nchar(x[3])))
  # If no baseline is present, zero is set to ADC zero. Otherwise it is equal to the baseline.
  adc.zeros <- sapply(signal.lines, function(x) {
    if (grepl("\\(.*\\)", x[3])) {
      return(as.numeric(substr(x[3], regexpr("\\(", x[3])+1, regexpr("\\)", x[3])-1)))
    } else {
      return(as.numeric(x[5]))
    }
  })

  return(list(root.dir = root.dir, record.name = record.name, record.type = record.type,
              sample.frequency = sample.frequency,  record.length = record.length, base.datetime = base.datetime,
              signal.format = signal.format, num.signals = num.signals, descriptions = descriptions, adc.gains = adc.gains,
              adc.zeros = adc.zeros, adc.units = adc.units, segment.names = segment.names,
              segment.lengths = segment.lengths))
}

#' Reads a a PhysioNET file
#'
#' @param header.filename The header file
#' @return A data frame containing the data
#' @export
rdsamp <- function(header.filename) {
  siginfo <- wfdbdesc(header.filename, FALSE)

  read.data <- function(data.filename, signal.format, num.signals, signal.length, adc.gains, adc.zeros) {
    data.file = file(data.filename, "rb")

    if (signal.format == 16) {
      signal.data <- t((array(readBin(data.file, integer(), n = file.size(data.filename)/2, size = 2,
                                      endian = "little"), dim=c(num.signals, signal.length)) - adc.zeros)/adc.gains)
    } else if (signal.format == 80) {
      raw.data <- readBin(data.file, integer(), n = file.size(data.filename), size = 1, endian = "little", signed = FALSE)
      signal.data <- t((array(raw.data - 128, dim=c(num.signals, signal.length)) - adc.zeros)/adc.gains)
    }

    close(data.file)
    return(signal.data)
  }

  if (siginfo$record.type == "single") {
    # If single-segment file, we can simply read and return the first segment.
    return (read.data(data.filename = sprintf("%s%s", siginfo$root.dir, siginfo$segment.names),
                      signal.format = siginfo$signal.format, num.signals = siginfo$num.signals,
                      signal.length = siginfo$record.length, adc.gains = siginfo$adc.gains, adc.zeros = siginfo$adc.zeros))
  } else {
    # We have to allocate a table for the unified data, read each segment, and place it into its proper place
    signal.data <- array(NA, dim = c(siginfo$record.length, siginfo$num.signals))
    segment.starts <- c(cumsum(c(1,siginfo$segment.lengths[1:(length(siginfo$segment.lengths)-1)])))
    segment.ends <- c(segment.starts[-1]-1, siginfo$record.length)
    segment.starts <- segment.starts[siginfo$segment.names != "~" & siginfo$segment.lengths > 0]
    segment.ends <- segment.ends[siginfo$segment.names != "~" & siginfo$segment.lengths > 0]
    # Recursively read data from segments
    valid.segments = sapply(siginfo$segment.names[siginfo$segment.names != "~" & siginfo$segment.lengths > 0],
                            function(x) sprintf("%s%s", siginfo$root.dir, x), USE.NAMES = FALSE)

    # Need to check that there are any valid segments
    if(length(valid.segments) > 0) {
      segment.headers <- lapply(valid.segments, function(x) wfdbdesc(x, FALSE))
      segment.data <- lapply(valid.segments, rdsamp)
      # Yo dawg, I heard you like lambdas, so I put a lambda in my lambda so you can lambda while you lambda
      indices <- lapply(segment.headers, function(x) sapply(x$descriptions,
                                                            function (y) which(siginfo$descriptions == y)))
      # Unify the data
      for (c in 1:length(segment.data)) {
        signal.data[segment.starts[c]:segment.ends[c],indices[[c]]] = segment.data[[c]]
      }
    }

    return(signal.data)
  }
}

#########################################################################################################################################################
# Feature engineering routines
#########################################################################################################################################################

#' Downsample a series to match size of another series given by rate
#'
#' @param data A vector of sensor data
#' @param rate The sampling rate to downsize to
#' @return A downsampled vector
#' @export
downsample <- function(data, rate)
{
  result <- c()
  for(i in seq(1, length(data), by = rate))
  {
    result <- c(result, mean(data[i:(i+rate)]))
  }
  return (result)
}


#' Creates new features from Empatica E4 data
#'
#' @param data A data frame of EDA and HR sensor data
#' @param window_size Window size to use for rolling windows
#' @return An expanded data frame of features
#' @export
rolling_features <- function(data, window_size)
{
  for (row in seq(1,nrow(data), by=window_size))
  {
    subset <- data[row:(row+(window_size-1)),]
    data[row:(row+(window_size-1)), "edamean"] <- mean(subset$eda)
    data[row:(row+(window_size-1)), "edamedian"] <- median(subset$eda)
    data[row:(row+(window_size-1)), "edastd"] <- sd(subset$eda)
    data[row:(row+(window_size-1)), "edavar"] <- var(subset$eda)
    data[row:(row+(window_size-1)), "edamin"] <- min(subset$eda)
    data[row:(row+(window_size-1)), "edamax"] <- max(subset$eda)
    data[row:(row+(window_size-1)), "edaskew"] <- e1071::skewness(subset$eda)
    data[row:(row+(window_size-1)), "edakurt"] <- e1071::kurtosis(subset$eda)
    data[row:(row+(window_size-1)), "edarange"] <- max(subset$eda) - min(subset$eda)
    data[row:(row+(window_size-1)), "hrmean"] <- mean(subset$hr)
    data[row:(row+(window_size-1)), "hrmedian"] <- median(subset$hr)
    data[row:(row+(window_size-1)), "hrstd"] <- sd(subset$hr)
    data[row:(row+(window_size-1)), "hrmin"] <- min(subset$hr)
    data[row:(row+(window_size-1)), "hrmax"] <- max(subset$hr)
    data[row:(row+(window_size-1)), "hrvar"] <- var(subset$hr)
    data[row:(row+(window_size-1)), "hrskew"] <- e1071::skewness(subset$hr)
    data[row:(row+(window_size-1)), "hrkurt"] <- e1071::kurtosis(subset$hr)
    data[row:(row+(window_size-1)), "hrrange"] <- max(subset$hr) - min(subset$hr)
    data[row:(row+(window_size-1)), "cov1"] <- cov(subset$hr, subset$eda)
  }
  data[is.nan(data$edakurt),"edakurt"] <-0
  data[is.nan(data$edaskew),"edaskew"] <-0
  data <- na.omit(data)
  # move metric to be last column
  metric <- data$metric
  data$metric <- NULL
  data$metric <- metric
  return (data)
}

#########################################################################################################################################################
# Dataframe generation routines
#########################################################################################################################################################
#' Loads Exam Data Set
#'
#' @param folder Folder containing the data
#' @return Data frame of EXAM data set
#' @export
make_exam_data <- function(folder) {
  subjects <- c("S1","S2","S3","S4","S5","S6","S7","S8","S9","S10")
  data <- NULL

  for (subject in subjects)
  {
    subject_file <- stresshelpers::read.empatica(paste(folder, "/",subject,"/Final",sep=""))
    hr <- subject_file$signal$hr$data
    eda <- subject_file$signal$eda$data
    eda <- eda[2:(length(eda)-1)]
    hr <- hr[2:(length(hr)-1)]
    eda <- eda[1:10800] # 3 hour exam
    hr <- hr[1:10800]# 3 hour exam
    eda <- stresshelpers::downsample(eda, round(length(eda) / length(hr)))
    shortest <- min(length(eda), length(hr))
    hr <- hr[1:shortest]
    eda <- eda[1:shortest]
    dataset <- cbind(eda, hr)
    dataset <- as.data.frame(dataset)
    dataset <- stresshelpers::rolling_features(dataset, 25)
    dataset$Subject <- paste(subject,"Final",sep='_')
    data <- rbind(data, dataset)
  }
  return (data)
}

#' Loads Exam Data Set with all exams
#'
#' @param folder Folder containing the data
#' @return Data frame of EXAM data set
#' @export
make_exam_data_full <- function(folder) {
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


#' Loads UBFC Data Set
#'
#' @param folder Folder containing the data
#' @param feature_engineering If TRUE, generates rolling features
#' @return Data frame of UBFC data set
#' @export
make_ubfc_data <- function(folder, feature_engineering = FALSE) {
  data <- NULL
  indexes <- c(1,4,6,9,11,15,18,19,27,29,30,31,32,33,34,37,39,44,45,47,48,49,50,51,53,55)
  for (subject in 1:56)
  {
    hr <- read.csv(paste(folder, '/s', subject, '_hr.csv', sep=''))
    hr <- hr[,1]
    eda <- read.csv(paste(folder, '/s', subject, '_eda.csv', sep=''))
    eda <- eda[,1:2]
    metric <- eda$metric
    eda <- eda$eda
    eda <- downsample(eda, round(length(eda) / length(hr)))
    metric <- downsample(metric, round(length(metric) / length(hr)))
    shortest <- min(length(eda), length(hr))
    hr <- hr[1:shortest]
    eda <- eda[1:shortest]
    metric <- metric[1:shortest]
    temp <- cbind(eda, hr, metric)
    temp <- as.data.frame(na.omit(temp))
    if (feature_engineering == TRUE)
    {
      temp <- rolling_features(temp, 50) # larger window for this data set
    }
    temp$Subject <- paste('U', subject, sep='')
    data <- rbind(data, temp)
  }
  range <- function(x){(x-min(x))/(max(x)-min(x))}
  data$metric <- range(data$metric)
  data$metric <- 1
  # intro is 5 min, remove
  data[1:300,"metric"] <- 5
  # rest 10 min
  data[301:901,"metric"] <- 0
  # remove ending of 3 min
  data[(nrow(data)-180):nrow(data),"metric"] <- 5
  data <- data[data$metric <= 1,]
  return (data)
}



#' Loads NEURO E4 Data Set
#'
#' @param folder Folder containing the E4 data
#' @param feature_engineering If TRUE, generates rolling features
#' @return Data frame of NEURO data set
#' @export
make_neuro_data <- function(folder, feature_engineering = FALSE)
{
  data <- NULL
  for (subject in 1:20)
  {
    file1 <- rdsamp(paste(folder,'/Subject', subject, '_AccTempEDA.hea', sep=''))
    file1 <- as.data.frame(file1)
    file2 <- rdsamp(paste(folder,'/Subject', subject, '_SpO2HR.hea', sep=''))
    file2 <- as.data.frame(file2)
    eda <- file1$V5
    hr <- file2$V2
    eda_sampling_rate <- round(length(eda) / length(hr))
    eda <- downsample(eda, eda_sampling_rate)
    shortest <- min(length(eda), length(hr))
    hr <- hr[1:shortest]
    eda <- eda[1:shortest]
    temp <- cbind(eda, hr)
    names(temp) <- c("eda","hr")
    temp <- as.data.frame(temp)
    # first 5 mins is relaxation
    temp[1:300,"metric"] <- 0
    # next 5 mins is exercise - ignore (300)
    temp[301:601,"metric"] <- 1
    # second relaxation 5 minutes (300) - ignore so can cool/calm down
    temp[602:902,"metric"] <- 0
    # 40 seconds - mini stress - (40)
    temp[903:943,"metric"] <- 5
    # 3 mins - stress (300)
    temp[944:1124,"metric"] <- 1
    # 5 mins relax (300)
    temp[1125:1425,"metric"] <- 0
    # 1 min anticipation (60)
    temp[1426:1486,"metric"] <- 0
    # 5 minutes stress
    temp[1487:1787,"metric"] <- 1
    # relax for 5 mins
    temp[1907:nrow(temp),"metric"] <- 5
    temp[1:120,"metric"] <- 5 # 2 mins to adjust to experiment
    temp[602:780,"metric"] <- 5 # cooldown after exercise
    temp[1020:1140,"metric"] <- 5 # cooldown after stress
    temp[1380:1500,"metric"] <- 5 # stress anticipation
    temp <- as.data.frame(temp)
    temp <- na.omit(temp)
    if (feature_engineering == TRUE)
    {
      temp <- rolling_features(temp, 25)
    }
    temp$Subject <- paste('N',subject,sep='')
    temp <- temp[temp$metric <= 1,]
    temp <- na.omit(temp) # remove entries we want to ignore
    data <- rbind(data, temp)
  }
  return (data)
}


#' Loads WESAD E4 Data Set
#'
#' @param folder Folder containing the E4 data
#' @param feature_engineering If TRUE, generates rolling features
#' @return Data frame of WESAD data set
#' @export
make_wesad_data <- function(folder, feature_engineering = FALSE)
{
  data <- NULL
  indexes <- c(3,4,5,6,7,8,9,10,11,13,14,15,16,17)
  for (subject in indexes)
  {
    wesad <- read.empatica(paste(folder, '/S', subject, sep=''))
    eda_sampling_rate <- wesad$signal$eda$samplingrate
    metrics <- read.csv(paste(folder,'/Metrics/S', subject, '_quest.csv', sep=''), sep=';')
    metrics <- metrics[1:3,2:9]
    names(metrics) <- metrics[1,]
    metrics <- metrics[2:3,]
    metric <- rep(NA, wesad$properties$length) # NA=transient/not defined
    for (index in 1:ncol(metrics))
    {
      temp <- strsplit(metrics[1,index],'.', TRUE)
      start <- (as.numeric(temp[[1]][1])*60)
      if (length(temp[[1]]) == 2)
      {
        start <- start + as.numeric(temp[[1]][2])
      }
      temp <- strsplit(metrics[2,index],'.', TRUE)
      end <- (as.numeric(temp[[1]][1])*60)
      if (length(temp[[1]]) == 2)
      {
        end <- end + as.numeric(temp[[1]][2])
      }
      if (substr(names(metrics)[index],1,4) == 'Base') metric[start:end] <- 0
      if (substr(names(metrics)[index],1,4) == 'TSST') metric[start:end] <- 1
      if (substr(names(metrics)[index],1,3) == 'Fun') metric[start:end] <- 0
      if (substr(names(metrics)[index],1,4) == 'Medi') metric[start:end] <- NA
    }
    hr <- wesad$signal$hr$data # 1Hz, each entry is 10 seconds
    eda <- downsample(wesad$signal$eda$data, eda_sampling_rate)
    shortest_sample <- min(length(hr), length(eda), length(metric))
    hr <- hr[1:shortest_sample]
    eda <- eda[1:shortest_sample]
    metric <- metric[1:shortest_sample]
    temp <- cbind(eda, hr, metric)
    names(temp) <- c("eda", "hr", "metric")
    temp <- as.data.frame(temp)
    temp <- na.omit(temp)
    if (feature_engineering == TRUE)
    {
      temp <- rolling_features(temp, 25)
    }
    temp$Subject <- paste('W', subject, sep='')
    data <- rbind(data, temp)
  }
  return(data)
}


#' Loads SWELL E4 Data Set
#'
#' @param folder Folder containing the E4 data
#' @param feature_engineering If TRUE, generates rolling features
#' @return Data frame of SWELL data set
#' @export
make_swell_data <- function(folder, feature_engineering = FALSE)
{
  data <- NULL
  indexes <- c(1,2,3,9,13,16,17,20,24)
  for (subject in indexes)
  {
    temp <- read.csv(paste(folder, '/p', subject, '.csv', sep=''))
    if (feature_engineering == TRUE)
    {
      temp <- rolling_features(temp, 25)
    }
    temp$Subject <- paste('S', subject, sep='')
    data <- rbind(data, temp)
  }
  data$metric <- data$metric - 1
  return (data)
}

#' Loads Toadstool E4 Data Set
#'
#' @param folder Folder containing the E4 data
#' @return Data frame of Toadstool data set
#' @export
make_toadstool_data <- function(folder)
{
  data <- NULL
  indexes <- c(1,2,3,4,5,6,7,9,10)
  for (subject in indexes)
  {
    toad <- read.empatica(paste(folder,'/P',subject,sep=''))
    eda_sampling_rate <- toad$signal$eda$samplingrate
    eda <- downsample(toad$signal$eda$data, eda_sampling_rate)
    hr <- toad$signal$hr$data
    shortest_sample <- min(length(hr), length(eda))
    hr <- hr[1:shortest_sample]
    eda <- eda[1:shortest_sample]
    temp <- cbind(eda, hr)
    temp <- as.data.frame(temp)
    temp <- na.omit(temp)
    names(temp) <- c("eda","hr")
    temp <- as.data.frame(temp)
    temp <- rolling_features(temp, 25)
    temp$Subject <- paste('T',subject,sep='')
    # ignore first 5 and last 5 seconds
    temp <- temp[5:(nrow(temp)-10),]
    data <- rbind(data, temp)
  }
  return(data)
}

#' Loads AffectiveROAD E4 Data Set
#'
#' @param folder Folder containing the E4 data
#' @return Data frame of AffectiveROAD data set
#' @export
make_drive_data <- function(folder)
{
  data <- NULL
  indexes <- c(1,2,3,4,5,6,7,8,9)
  for (subject in indexes)
  {
    # using annotations, extract driving sections
    annotations <- read.csv(paste(folder, '/Metrics/Annot_Subjective_metric.csv',sep=''))
    annotations <- annotations[annotations$Drive_id==paste('Drv',1,sep=''),]
    left <- read.empatica(paste(folder,'/E4/',subject,'-E4-Drv',subject,'/Left',sep='')) # left hand only
    hr <- left$signal$hr$data
    eda_sampling_rate <- left$signal$eda$samplingrate
    metrics <- read.csv(paste(folder, '/Metrics/SM_Drv',subject,'.csv',sep=''))
    metrics <- metrics$x
    metrics1 <- metrics[annotations$City1_Start:annotations$City1_End]
    metrics2 <- metrics[annotations$Hwy_Start:annotations$Hwy_End]
    metrics3 <- metrics[annotations$City2_Start:annotations$City2_End]
    metrics4 <- metrics[annotations$City2_Start.1:annotations$City2_End.1]
    metrics5 <- metrics[annotations$Hwy_Start.1:annotations$Hwy_End.1]
    metrics6 <- metrics[annotations$City1_Start.1:annotations$City1_End.1]
    metrics <- c(metrics1, metrics2, metrics3, metrics4, metrics5, metrics6)
    metrics_sampling_rate <- round(length(metrics) / length(hr))
    eda <- downsample(left$signal$eda$data, eda_sampling_rate)
    metric <- downsample(metrics, metrics_sampling_rate)
    shortest_sample <- min(length(hr), length(eda), length(metric))
    temp <- data.frame(eda[1:shortest_sample], hr[1:shortest_sample], metric[1:shortest_sample])
    temp <- as.data.frame(temp)
    names(temp) <- c( "eda","hr","metric")
    subset1 <- temp[annotations$City1_Start:annotations$City1_End,]
    subset2 <- temp[annotations$Hwy_Start:annotations$Hwy_End,]
    subset3 <- temp[annotations$City2_Start:annotations$City2_End,]
    subset4 <- temp[annotations$City2_Start.1:annotations$City2_End.1,]
    subset5 <- temp[annotations$Hwy_Start.1:annotations$Hwy_End.1,]
    subset6 <- temp[annotations$City1_Start.1:annotations$City1_End.1,]
    subset1$Test <- 'City1'
    subset3$Test <- 'City2'
    subset4$Test <- 'City3'
    subset6$Test <- 'City4'
    subset2$Test <- 'Highway1'
    subset5$Test <- 'Highway2'
    temp <- rbind(subset1, subset2, subset3, subset4, subset5, subset6)
    temp <- na.omit(temp)
    temp <- rolling_features(temp, 25)
    temp$Subject <- paste('D', subject,sep='')
    temp <- temp[temp$metric > 0,]
    data <- rbind(data,temp)
  }
  return(data)
}

