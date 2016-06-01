# coding: utf-8
#__author__ = 'Ravi: https://kaggle.com/company'


# Required Libraries
library(data.table)
library(readr)
library(caret)
library(stringdist)

# Read in data
location <- fread("../input/Location.csv")
itemPairsTest <- fread("../input/ItemPairs_test.csv")
itemPairsTrain <- fread("../input/ItemPairs_train.csv")
itemInfoTest <- read_csv("../input/ItemInfo_test.csv")
itemInfoTrain <- read_csv("../input/ItemInfo_train.csv")
itemInfoTest <- data.table(itemInfoTest)
itemInfoTrain <- data.table(itemInfoTrain)

setkey(location, locationID)
setkey(itemInfoTrain, itemID)
setkey(itemInfoTest, itemID)

# Drop unused factors
dropAndNumChar <- function(itemInfo){
  itemInfo[, ':=' (ncharTitle = nchar(title),
                   ncharDescription = nchar(description),
                   description = NULL,
                   images_array = NULL,
                   attrsJSON = NULL)]
}

dropAndNumChar(itemInfoTest)
dropAndNumChar(itemInfoTrain)


# Merge
mergeInfo <- function(itemPairs, itemInfo){
  # merge on itemID_1
  setkey(itemPairs, itemID_1)
  itemPairs <- itemInfo[itemPairs]
  setnames(itemPairs, names(itemInfo), paste0(names(itemInfo), "_1"))
  # merge on itemID_2
  setkey(itemPairs, itemID_2)
  itemPairs <- itemInfo[itemPairs]
  setnames(itemPairs, names(itemInfo), paste0(names(itemInfo), "_2"))
  # merge on locationID_1
  setkey(itemPairs, locationID_1)
  itemPairs <- location[itemPairs]
  setnames(itemPairs, names(location), paste0(names(location), "_1"))
  # merge on locationID_2
  setkey(itemPairs, locationID_2)
  itemPairs <- location[itemPairs]
  setnames(itemPairs, names(location), paste0(names(location), "_2"))
  return(itemPairs)
}

itemPairsTrain <- mergeInfo(itemPairsTrain, itemInfoTrain)
itemPairsTest <- mergeInfo(itemPairsTest, itemInfoTest)

rm(list=c("itemInfoTest", "itemInfoTrain", "location"))

# Create features
matchPair <- function(x, y){
  ifelse(is.na(x), ifelse(is.na(y), 3, 2), ifelse(is.na(y), 2, ifelse(x==y, 1, 4)))
}

createFeatures <- function(itemPairs){
  itemPairs[, ':=' (locationMatch = matchPair(locationID_1, locationID_2),
                    locationID_1 = NULL,
                    locationID_2 = NULL,
                    regionMatch = matchPair(regionID_1, regionID_2),
                    regionID_1 = NULL,
                    regionID_2 = NULL,
                    metroMatch = matchPair(metroID_1, metroID_2),
                    metroID_1 = NULL,
                    metroID_2 = NULL,
                    categoryID_1 = NULL,
                    categoryID_2 = NULL,
                    priceMatch = matchPair(price_1, price_2),
                    priceDiff = pmax(price_1/price_2, price_2/price_1),
                    priceMin = pmin(price_1, price_2, na.rm=TRUE),
                    priceMax = pmax(price_1, price_2, na.rm=TRUE),
                    price_1 = NULL,
                    price_2 = NULL,
                    titleStringDist = stringdist(title_1, title_2, method = "jw"),
                    titleStringDist2 = (stringdist(title_1, title_2, method = "lcs") / 
                        pmax(ncharTitle_1, ncharTitle_2, na.rm=TRUE)),
                    title_1 = NULL,
                    title_2 = NULL,
                    titleCharDiff = pmax(ncharTitle_1/ncharTitle_2, ncharTitle_2/ncharTitle_1),
                    titleCharMin = pmin(ncharTitle_1, ncharTitle_2, na.rm=TRUE),
                    titleCharMax = pmax(ncharTitle_1, ncharTitle_2, na.rm=TRUE),
                    ncharTitle_1 = NULL,
                    ncharTitle_2 = NULL,
                    descriptionCharDiff = pmax(ncharDescription_1/ncharDescription_2, ncharDescription_2/ncharDescription_1),
                    descriptionCharMin = pmin(ncharDescription_1, ncharDescription_2, na.rm=TRUE),
                    descriptionCharMax = pmax(ncharDescription_1, ncharDescription_2, na.rm=TRUE),
                    ncharDescription_1 = NULL,
                    ncharDescription_2 = NULL,
                    distance = sqrt((lat_1-lat_2)^2+(lon_1-lon_2)^2),
                    lat_1 = NULL,
                    lat_2 = NULL,
                    lon_1 = NULL,
                    lon_2 = NULL,
                    itemID_1 = NULL,
                    itemID_2 = NULL)]
  
  itemPairs[, ':=' (priceDiff = ifelse(is.na(priceDiff), 0, priceDiff),
                    priceMin = ifelse(is.na(priceMin), 0, priceMin),
                    priceMax = ifelse(is.na(priceMax), 0, priceMax),
                    titleStringDist = ifelse(is.na(titleStringDist), 0, titleStringDist),
                    titleStringDist2 = ifelse(is.na(titleStringDist2) | titleStringDist2 == Inf, 0, titleStringDist2))]
}

createFeatures(itemPairsTest)
createFeatures(itemPairsTrain)


library(xgboost)

maxTrees <- 100
shrinkage <- 0.1
gamma <- 2
depth <- 8
minChildWeight <- 60
colSample <- 0.9
subSample <- 0.9
earlyStopRound <- 2

modelVars <- names(itemPairsTrain)[which(!(names(itemPairsTrain) %in% c("isDuplicate", "generationMethod", "foldId")))]

itemPairsTest <- data.frame(itemPairsTest)
itemPairsTrain <- data.frame(itemPairsTrain)
set.seed(1984)
itemPairsTrain <- itemPairsTrain[sample(nrow(itemPairsTrain), 50000), ]

# Matrix
dtrain <- xgb.DMatrix(as.matrix(itemPairsTrain[, modelVars]), label=itemPairsTrain$isDuplicate)
dtest <- xgb.DMatrix(as.matrix(itemPairsTest[, modelVars]))

# xgboost cross-validated
set.seed(1984)
xgbCV <- xgb.cv(params=list(max_depth=depth,
                            eta=shrinkage,
                            gamma=gamma,
                            colsample_bytree=colSample,
                            min_child_weight=minChildWeight,
                            subsample=subSample,
                            objective="binary:logistic"),
                data=dtrain,
                nrounds=maxTrees,
                eval_metric ="auc",
                nfold=4,
                stratified=TRUE,
                early.stop.round=earlyStopRound)

numTrees <- min(which(xgbCV$test.auc.mean==max(xgbCV$test.auc.mean)))

xgbResult <- xgboost(params=list(max_depth=depth,
                                 eta=shrinkage,
                                 gamma=gamma,
                                 colsample_bytree=colSample,
                                 min_child_weight=minChildWeight),
                     data=dtrain,
                     nrounds=numTrees,
                     objective="binary:logistic",
                     eval_metric="auc")

testPreds <- predict(xgbResult, dtest)

submission <- data.frame(id=itemPairsTest$id, probability=testPreds)
write.csv(submission, file="submission.csv",row.names=FALSE)

