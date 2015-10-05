library(caret); library(plyr)

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

head(df.train)
str(df.train)

set.seed(35)

#Split into training and test set from training set
training.rows <- createDataPartition(df.train$Survived, p = 0.8, list = FALSE)
train.batch <- df.train[training.rows, ]
test.batch <- df.train[-training.rows, ]

#Create model
glm.tune <- train(Survived ~ Sex + Pclass + SibSp + Parch + Embarked, data = train.batch, method = "glm")

# use the logistic regression model to generate predictions
Survived <- predict(glm.tune, newdata = test.batch)

result <- confusionMatrix(Survived, test.batch[,2])
print(result)

#Create model
output.model <- train(Survived ~ Sex + Pclass + SibSp + Parch + Embarked, data = df.train, method = "glm")

# use the logistic regression model to generate predictions
Survived <- predict(output.model, newdata = df.infer)

# link to PassengerId in a data frame
predictions <- as.data.frame(Survived)
#colnames(predictions) <- "Survived"
predictions$PassengerId <- df.infer$PassengerId


# write predictions to csv file for submission to Kaggle
write.csv(predictions[,c("PassengerId", "Survived")], 
          file="output/Titanic_predictions.csv", row.names=FALSE, quote=FALSE)
