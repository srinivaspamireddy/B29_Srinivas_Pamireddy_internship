# read the dataset
getwd()
library(ggplot2)
# set the current working directory 
setwd("C:\\Users\\Srini\\Desktop\\Intern\\epileptic seizure")

getwd()
#read the epileptic seizure dataset

epilep_data<-read.csv("data.csv",sep =",",header = TRUE)
# to view all the columns using utils function

utils::View(epilep_data)

# observe the characteristics of the data atrributes by using str function

str(epilep_data)

# get the attribute characteristics for the 99 to 180 columns.
str(epilep_data[,99:180])
# expect first attribute,all the attributes are in numeric nature


# check the mean median, standard deviation and na's of the data by using the summary function

summary(epilep_data)
# check class distribution of the target variable.

table(epilep_data$y)

hist((epilep_data$y))

# remove the first column X as it is a just text data

epilep_data$X <- NULL

# check for missing values.

sum(is.na(epilep_data))

# there are no missing values in the data.


# convert the target variable into factor

epilep_data$y <-as.factor(as.character(epilep_data$y))
str(epilep_data$y)

# split the data into train and test.

set.seed(1234)

library(caret)
train_rows <- createDataPartition(epilep_data$y,p=.7,list = FALSE)
train_data_ep<- epilep_data[train_rows,]
test_data_ep <- epilep_data[-train_rows,]

# standardize the data 

std_method_ep<-preProcess(train_data_ep[,!names(train_data_ep) %in% c("y")],method = c("center","scale"))
train_data_ep[,!names(train_data_ep) %in% c("y")] <- predict(std_method_ep,train_data_ep[,!names(train_data_ep) %in% c("y")])

test_data_ep[,!names(test_data_ep) %in% c("y")] <-predict(std_method_ep,test_data_ep[,!names(test_data_ep) %in% c("y")])
str(train_data_ep$y)
test_data_ep$y
utils::View(train_data_ep)



# seperate the target variable from the test and train data for applying the KNN algorithm on it.

train_data_withoutclass_ep <- subset(train_data_ep,select = -c(y))
test_data_withoutclass_ep <-subset(test_data_ep,select = -c(y))



# apply the knn model on the data 

library(class)


pred_knn <- knn(train_data_withoutclass_ep,test_data_withoutclass_ep,train_data_ep$y,k=7)

pred_knn

library(caret)
confusionMatrix(pred_knn,test_data_ep$y)
