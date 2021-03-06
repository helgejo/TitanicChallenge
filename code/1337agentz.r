library(Amelia); library(Hmisc); library(corrgram); library(caret); library(vcd); library(stringr);library(plyr);library(stringr)

readData <- function(path.name, file.name, column.types, missing.types) {
        read.csv( paste( path.name, file.name, sep="" ), 
                  colClasses= column.types,
                  na.strings= missing.types )
}

Titanic.path <- "data/"
train.data.file <- "train.csv"
test.data.file <- "test.csv"
missing.types <- c("NA", "")
train.column.types <- c('integer',   # PassengerId
                        'factor',    # Survived 
                        'factor',    # Pclass
                        'character', # Name
                        'factor',    # Sex
                        'numeric',   # Age
                        'integer',   # SibSp
                        'integer',   # Parch
                        'character', # Ticket
                        'numeric',   # Fare
                        'character', # Cabin
                        'factor'     # Embarked
)
test.column.types <- train.column.types[-2]     # # no Survived column in test.csv

train.raw <- readData(Titanic.path, train.data.file, 
                      train.column.types, missing.types)
df.train <- train.raw

test.raw <- readData(Titanic.path, test.data.file, 
                     test.column.types, missing.types)
df.infer <- test.raw   

#Time to tackle those missing ages.
#By the late 19th century, etiquette dictated that men be addressed as Mister, and boys as Master
df.train$Title <- sapply(df.train$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
df.train$Title <- sub(' ', '', df.train$Title)
#unique(df.train$Title)

options(digits=2)
#bystats(df.train$Age, df.train$Title, 
#        fun=function(x)c(Mean=mean(x),Median=median(x)))


## list of titles with missing Age value(s) requiring imputation
titles.na.train <- c("Dr", "Master", "Mrs", "Miss", "Mr")

imputeMedian <- function(impute.var, filter.var, var.levels) {
        for (v in var.levels) {
                impute.var[ which( filter.var == v)] <- impute(impute.var[ 
                        which( filter.var == v)])
        }
        return (impute.var)
}

#df.train$Age[which(df.train$Title=="Dr")]

df.train$Age <- imputeMedian(df.train$Age, df.train$Title,titles.na.train)

#df.train$Age[which(df.train$Title=="Dr")]


df.train$Title[df.train$Title %in% c('Capt', 'Don', 'Major', 'Sir', 'Col', 'Dr', 'Rev')] <- 'Sir'
df.train$Title[df.train$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer', 'Ms')] <- 'Lady'
df.train$Title[df.train$Title %in% c('Mme', 'Mlle')] <- 'Miss'
df.train$Title <- factor(df.train$Title)

# impute embarked
df.train$Embarked[which(is.na(df.train$Embarked))] <- 'S'


## impute missings on Fare feature with median fare by Pclass
df.train$Fare[ which( df.train$Fare == 0 )] <- NA
df.train$Fare <- imputeMedian(df.train$Fare, df.train$Pclass, 
                              as.numeric(levels(df.train$Pclass)))


require(plyr)     # for the revalue function 
require(stringr)  # for the str_sub function

## test a character as an EVEN single digit
isEven <- function(x) x %in% c("0","2","4","6","8") 
## test a character as an ODD single digit
isOdd <- function(x) x %in% c("1","3","5","7","9") 

## function to add features to training or test data frames
featureEngrg <- function(data) {
        ## Using Fate ILO Survived because term is shorter and just sounds good
        data$Fate <- data$Survived
        ## Revaluing Fate factor to ease assessment of confusion matrices later
        data$Fate <- revalue(data$Fate, c("1" = "Survived", "0" = "Perished"))
        ## Boat.dibs attempts to capture the "women and children first"
        ## policy in one feature.  Assuming all females plus males under 15
        ## got "dibs' on access to a lifeboat
        data$Boat.dibs <- "No"
        data$Boat.dibs[which(data$Sex == "female" | data$Age < 15)] <- "Yes"
        data$Boat.dibs <- as.factor(data$Boat.dibs)
        ## Family consolidates siblings and spouses (SibSp) plus
        ## parents and children (Parch) into one feature
        data$Family <- data$SibSp + data$Parch
        ## Fare.pp attempts to adjust group purchases by size of family
        data$Fare.pp <- data$Fare/(data$Family + 1)
        ## Giving the traveling class feature a new look
        data$Class <- data$Pclass
        data$Class <- revalue(data$Class, 
                              c("1"="First", "2"="Second", "3"="Third"))
        ## First character in Cabin number represents the Deck 
        data$Deck <- substring(data$Cabin, 1, 1)
        data$Deck[ which( is.na(data$Deck ))] <- "UNK"
        data$Deck <- as.factor(data$Deck)
        ## Odd-numbered cabins were reportedly on the port side of the ship
        ## Even-numbered cabins assigned Side="starboard"
        data$cabin.last.digit <- str_sub(data$Cabin, -1)
        data$Side <- "UNK"
        data$Side[which(isEven(data$cabin.last.digit))] <- "port"
        data$Side[which(isOdd(data$cabin.last.digit))] <- "starboard"
        data$Side <- as.factor(data$Side)
        data$cabin.last.digit <- NULL
        return (data)
}

## add remaining features to training data frame
df.train <- featureEngrg(df.train)

train.keeps <- c("Fate", "Sex", "Boat.dibs", "Age", "Title", 
                 "Class", "Deck", "Side", "Fare", "Fare.pp", 
                 "Embarked", "Family")
df.train.munged <- df.train[train.keeps]

## split training data into train batch and test batch
set.seed(23)
training.rows <- createDataPartition(df.train.munged$Fate,p = 0.8, list = FALSE)
train.batch <- df.train.munged[training.rows, ]
test.batch <- df.train.munged[-training.rows, ]


## Define control function to handle optional arguments for train function
## Models to be assessed based on largest absolute area under ROC curve
cv.ctrl <- trainControl(method = "repeatedcv", repeats = 3,
                        summaryFunction = twoClassSummary,
                        classProbs = TRUE)    

set.seed(35)
glm.tune.1 <- train(Fate ~ Sex + Class + Age + Family + Embarked,
                    data = train.batch,
                    method = "glm",
                    metric = "ROC",
                    trControl = cv.ctrl)

prediction1 <- predict(glm.tune.1, newdata = test.batch )
confusionMatrix(prediction1, test.batch[,1])

glm.tune.2 <- train(Fate ~ Sex + Class + Age + Family + I(Embarked=="S"),
                    data = train.batch,
                    method = "glm",
                    metric = "ROC",
                    trControl = cv.ctrl)

prediction2 <- predict(glm.tune.2, newdata = test.batch )
confusionMatrix(prediction2, test.batch[,1])

glm.tune.3 <- train(Fate ~ Sex + Class + Title + Age + Family + I(Embarked=="S"),
                    data = train.batch,
                    method = "glm",
                    metric = "ROC",
                    trControl = cv.ctrl)

prediction3 <- predict(glm.tune.3, newdata = test.batch )
confusionMatrix(prediction3, test.batch[,1])

glm.tune.5 <- train(Fate ~ Class + I(Title=="Mr") + I(Title=="Sir") + Age + Family + I(Embarked=="S") + I(Title=="Mr"&Class=="Third"),
                    data = train.batch,
                    method = "glm",
                    metric = "ROC",
                    trControl = cv.ctrl)

prediction5 <- predict(glm.tune.5, newdata = test.batch )
confusionMatrix(prediction5, test.batch[,1])


rf.tune <- train(Fate~., data = train.batch, method = "rf")
rf.prediction <- predict(rf.tune, newdata = test.batch )
confusionMatrix(rf.prediction, test.batch[,1])


## Boosted model (GREEN curve)
ada.grid <- expand.grid(.iter = c(50, 100),
                        .maxdepth = c(4, 8),
                        .nu = c(0.1, 1))

ada.tune <- train(Fate ~ Sex + Class + Age + Family + Embarked, 
                  data = train.batch,
                  method = "ada",
                  metric = "ROC",
                  tuneGrid = ada.grid,
                  trControl = cv.ctrl)


ada.probs <- predict(ada.tune, test.batch, type = "prob")
confusionMatrix(ada.probs, test.batch[,1])


## SVM model (BLUE curve)
svm.probs <- predict(svm.tune, test.batch, type = "prob")
svm.ROC <- roc(response = test.batch$Fate,
               predictor = svm.probs$Survived,
               levels = levels(test.batch$Fate))
plot(svm.ROC, add=TRUE, col="blue")
## Area under the curve: 0.8077

glm.tune.4 <- train(Fate ~ Class + I(Title=="Mr") + I(Title=="Noble") 
                    + Age + Family + I(Embarked=="S"), 
                    data = train.batch, method = "glm",
                    metric = "ROC", trControl = cv.ctrl)

summary(glm.tune.4)

df.infer$Title <- sapply(df.infer$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
df.infer$Title <- sub(' ', '', df.infer$Title)
options(digits=2)
titles.na.test <- c("Master", "Mrs", "Miss", "Mr")
df.infer$Age <- imputeMedian(df.infer$Age, df.infer$Title,titles.na.test)

df.infer$Title[df.infer$Title %in% c('Capt', 'Don', 'Major', 'Sir', 'Col', 'Dr', 'Rev')] <- 'Sir'
df.infer$Title[df.infer$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer', 'Ms')] <- 'Lady'
df.infer$Title[df.infer$Title %in% c('Mme', 'Mlle')] <- 'Miss'
df.infer$Title <- factor(df.infer$Title)

## impute missings on Fare feature with median fare by Pclass
df.infer$Fare[ which( df.infer$Fare == 0 )] <- NA
df.infer$Fare <- imputeMedian(df.infer$Fare, df.infer$Pclass, 
                              as.numeric(levels(df.infer$Pclass)))

# add the other features
df.infer <- featureEngrg(df.infer)

# data prepped for casting predictions
test.keeps <- train.keeps[-1]
pred.these <- df.infer[test.keeps]



# use the logistic regression model to generate predictions
Survived <- predict(glm.tune.3, newdata = pred.these)



# reformat predictions to 0 or 1 and link to PassengerId in a data frame
Survived <- revalue(Survived, c("Survived" = 1, "Perished" = 0))
predictions <- as.data.frame(Survived)
predictions$PassengerId <- df.infer$PassengerId
#summary(df.infer$PassengerId)
predictions
# write predictions to csv file for submission to Kaggle
write.csv(predictions[,c("PassengerId", "Survived")], 
          file="output/Titanic_predictions.csv", row.names=FALSE, quote=FALSE)