library(fpp2)
options(digits=4)

#get file path from cmdline
args <- commandArgs(trailingOnly = TRUE)

#read in the data (assume csv)
#rawData = subset(read.csv(head(args), TRUE, ","), select = c("R3"))
rawData = subset(read.csv("./SI_archive/Target_Rain_ENI_regional_ave_time_series.csv", TRUE, ","), select = c("R3"))

#split the data into seasonal head, body, tail parts to avoid overlap; assume the original data starts on April 1993 (they all do)
seasonalBody = rawData[seq(1, nrow(rawData), 3), ]
seasonalTail = rawData[seq(2, nrow(rawData), 3), ]
seasonalHead = rawData[seq(3, nrow(rawData), 3), ]

#creating time series objects from the datasets.
#We shall define yearly quarters as seasons of the year (Q1 = Winter [Dec, Jan, Feb], Q2 = Spring [Mar, Apr, May], Q3 = Summer [Jun, Jul, Aug], Q4 = Autumn [Sep, Oct, Nov])
#This means that body, tail start on Q2 as they begin in Apr, May which is in spring, but head starts in Q3 as it starts in June which is Summer
tshead = ts(data = seasonalHead, start = c(1993, 3), frequency = 4)
tsbody = ts(data = seasonalBody, start = c(1993, 2), frequency = 4)
tstail = ts(data = seasonalTail, start = c(1993, 2), frequency = 4)


#The unmodified data is monthly instead of quarterly, and starts on April i.e. the 4th month
tsfull = ts(data = rawData, start = c(1993, 4), frequency = 12)

#creating data subsets for hold-out testing, leaving out the last 4 points (12 for the unmodified data) so there is a year of data for testing
trainhead = head(tshead, -4)
trainbody = head(tsbody, -4)
traintail = head(tstail, -4)
trainfull = head(tsfull, -12);
testhead = tail(tshead, 4)
testbody = tail(tshead, 4)
testtail = tail(tshead, 4)
testfull = tail(tsfull, 12)

#Training our models on the held-out training data
#first the trivial model outputting just the average
meanhead = meanf(trainhead, h = 4)
meanbody = meanf(trainbody, h = 4)
meantail = meanf(traintail, h = 4)
meanfull = meanf(trainfull, h = 12)

#Seasonal naive model:
naivehead = snaive(trainhead, h = 4)
naivebody = snaive(trainbody, h = 4)
naivetail = snaive(traintail, h = 4)
naivefull = snaive(trainfull, h = 12)

#Holt winters additive:
hwahead = hw(trainhead, h = 4, seasonal="additive")
hwabody = hw(trainbody, h = 4, seasonal="additive")
hwatail = hw(traintail, h = 4, seasonal="additive")
hwafull = hw(trainfull, h = 12, seasonal="additive")

#Holt winters multiplicative
hwmhead = hw(trainhead, h = 4, seasonal="multiplicative")
hwmbody = hw(trainbody, h = 4, seasonal="multiplicative")
hwmtail = hw(traintail, h = 4, seasonal="multiplicative")
hwmfull = hw(trainfull, h = 12, seasonal="multiplicative")

cat("Errors for mean model:\n")
cat("head:\n")
print(accuracy(meanhead, testhead),2)
cat("\nbody:\n")
accuracy(meanbody, testbody)
cat("\ntail:\n")
accuracy(meantail, testtail)
cat("\nfull:\n")
accuracy(meanfull, testfull)

cat("\n\nErrors for seasonal naive model:\n")
cat("head:\n")
accuracy(naivehead, testhead)
cat("\nbody:\n")
accuracy(naivebody, testbody)
cat("\ntail:\n")
accuracy(naivetail, testtail)
cat("\nfull:\n")
accuracy(naivefull, testfull)

cat("\n\nErrors for holt-winters additive:\n")
cat("head:\n")
accuracy(hwahead, testhead)
cat("\nbody:\n")
accuracy(hwabody, testbody)
cat("\ntail:\n")
accuracy(hwatail, testtail)
cat("\nfull:\n")
accuracy(hwafull, testfull)

cat("\n\nErrors for holt-winters multiplicative:\n")
cat("head:\n")
accuracy(hwmhead, testhead)
cat("\nbody:\n")
accuracy(hwmbody, testbody)
cat("\ntail:\n")
accuracy(hwmtail, testtail)
cat("\nfull:\n")
accuracy(hwmfull, testfull)