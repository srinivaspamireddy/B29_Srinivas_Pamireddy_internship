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
# except first attribute,all the attributes are in numeric nature

#check for any attributes with near zero variance.

zero_var<-nearZeroVar(epilep_data)

zero_var
# there are no near zero variance attributes in this dataset.

# check the mean median, standard deviation and na's of the data by using the summary function

summary(epilep_data)
# check class distribution of the target variable.

table(epilep_data$y)

hist((epilep_data$y))

# preprocessing steps

# remove the first column X as it is a just text data

epilep_data$X <- NULL

# check for missing values.

sum(is.na(epilep_data))

# there are no missing values in the data.


# as the target varaible has 5 levels of data with values(1,2,3,4,5), and value 1 corresponds to epileptic 
#seizure and rest of the values corresponds to normal state. Convert the target into 2 level variable with 
# 0 and 1.




epilep_data$target<-ifelse(epilep_data$y == "1","1","0")

epilep_data$target<-as.factor(as.character(epilep_data$target))

# remove the original target variable as it becomes reduntant.

epilep_data$y <- NULL
#check the class distribution after converting the target variable into 2 levels.

table(epilep_data$target)
  
utils::View(epilep_data)

histogram(epilep_data$target)

# convert the target variable into factor
epilep_data$target <-as.factor(as.character(epilep_data$target))


str(epilep_data$target)

# corroplot

library(corrplot)

corrplot(cor(epilep_data[,!names(epilep_data) %in% c("target")]))

library(GGally)

ggpairs(epilep_data[,0:10])
# plots
## Basic scatterplot with 2 dimensions, x and y
ggplot(data=epilep_data, aes(x=X1, y=target))+ geom_point() + 
  xlab("X1")+ylab("target") + 
  ggtitle("x1 Vs. target")


## Basic scatterplot with 2 dimensions, x and y
ggplot(data=epilep_data, aes(x=X2, y=target))+ geom_point() + 
  xlab("X1")+ylab("target") + 
  ggtitle("x2 Vs. target")

# split the data into train and test.

set.seed(1234)

library(caret)
train_rows <- createDataPartition(epilep_data$y,p=.7,list = FALSE)
train_data_ep<- epilep_data[train_rows,]
test_data_ep <- epilep_data[-train_rows,]

# standardize the data 

std_method_ep<-preProcess(train_data_ep[,!names(train_data_ep) %in% c("target")],method = c("center","scale"))
train_data_ep[,!names(train_data_ep) %in% c("target")] <- predict(std_method_ep,train_data_ep[,!names(train_data_ep) %in% c("target")])

test_data_ep[,!names(test_data_ep) %in% c("target")] <-predict(std_method_ep,test_data_ep[,!names(test_data_ep) %in% c("target")])
str(train_data_ep$target)
test_data_ep$target
utils::View(train_data_ep)



# seperate the target variable from the test and train data for applying the KNN algorithm on it.

train_data_withoutclass_ep <- subset(train_data_ep,select = -c(target))
test_data_withoutclass_ep <-subset(test_data_ep,select = -c(target))



# apply the knn model on the data 

library(class)


pred_knn <- knn(train_data_withoutclass_ep,test_data_withoutclass_ep,train_data_ep$target,k=7)

pred_knn

library(caret)
confusionMatrix(pred_knn,test_data_ep$target,positive = "1")

# apply the pca on the data to reduce the dimensions.

pca_data<- prcomp(train_data_ep[,!names(train_data_ep) %in% c("target")])

summary(pca_data)

plot(pca_data$sdev)

plot(pca_data,type = "l")
#

# 55 variables contributing to the 99% of variation in data

# convert the train and test data into PCA form

train_data_ep_pca<-as.data.frame(predict(pca_data, train_data_ep[,!names(train_data_ep) %in% c("target")]))[1:55]

test_data_ep_pca<-as.data.frame(predict(pca_data,test_data_ep[,!names(test_data_ep) %in% c("target")]))[1:55]

#binding the target to train and test in pca form.


train_data_ep_pca<-cbind(train_data_ep_pca,train_data_ep$target)
test_data_ep_pca <-cbind(test_data_ep_pca,test_data_ep$target)

colnames(train_data_ep_pca)[56]<-"target"
colnames(test_data_ep_pca) [56]<-"target"

# apply knn on the PCA data

pred_knn_pca<- knn(train_data_ep_pca,test_data_ep_pca,train_data_ep$target,k = 7)

pred_knn_pca

confusionMatrix(pred_knn_pca,test_data_ep$target,positive = "1")

# apply cross validation on the KNN model

# create the control parameter using trainControl function.
ctrl_knn <- trainControl(method = "repeatedcv",repeats = 5)

model_knn_cv<- train(target~.,data = train_data_ep_pca,method = "knn",
                     trControl = ctrl_knn)
model_knn_cv

# predict on the test data

pred_knn_pca_cv<- predict(model_knn_cv,test_data_ep_pca)

confusionMatrix(pred_knn_pca_cv,test_data_ep_pca$target,positive = "1")

# sensitivity got improved from  0.5739 to .6000 after applying the cross validation on the knn model.

# apply the decision tree.

library(rpart)
library(rpart.plot)

model_dt <- rpart(target~.,train_data_ep_pca)

pred_dt<-predict(model_dt,test_data_ep_pca)

pred_dt_tree<-ifelse(pred_dt[,1]> pred_dt[,2],0,1)

confusionMatrix(pred_dt_tree,test_data_ep_pca$target,positive = "1")


rpart.plot(model_dt)


# the sensitivity has increased from .60 to .8913 in decision tree.

# apply the cross validation on the  decision tree.

ctrl_dt <-trainControl("repeatedcv",5) # for doing the cross validation and number of times to be repeated
rpart.grid<-expand.grid(.cp = seq(.01,.2,.02)) # determines the different grid values to chosen for cp.


model_dt_cv<-train(target~.,data = train_data_ep_pca,method = "rpart",trControl = ctrl_dt,tuneLength=9,tuneGrid=rpart.grid)


                   

model_dt_cv

# at .cp=0.01 kappa and accurracy values are high
pred_dt_cv<-predict(model_dt_cv,test_data_ep_pca)

confusionMatrix(pred_dt_cv,test_data_ep_pca$target,positive = "1")
# sensitivity got increased to .8913 by using the decision tree model.

# apply the SVM model on the data
library(e1071)
model_svm<-svm(target~.,train_data_ep_pca,kernel = "linear")
pred_svm<-predict(model_svm,test_data_ep_pca)
confusionMatrix(pred_svm,test_data_ep_pca$target,positive = "1")
# sensitivity is very low with SVM model, it's just .06

# apply svm with kernal as radial basis
library(kernlab)

model_svm_rb<- ksvm(target~.,train_data_ep_pca,kernel = "rbfdot")

pred_svm_rb <- predict(model_svm_rb,test_data_ep_pca)

confusionMatrix(pred_svm_rb,test_data_ep_pca$target, positive = "1")

# apply tune.svm 

library(mlr)

train_pca_task<-makeClassifTask(data = train_data_ep_pca,target = "target")
ps = makeParamSet(makeDiscreteParam("C",values = 2^(-2:2)),makeDiscreteParam("sigma",values = 2^(-2:2)),
                  makeDiscreteParam("kernel",values = c("rbfdot")))
ctrl_ksvm<-makeTuneControlGrid()
rdesc = makeResampleDesc("CV",iters = 3L)
model_svm_tune = tuneParams("classif.ksvm",task = train_pca_task,resampling = rdesc ,par.set = ps,
                            control = ctrl_ksvm )

model_svm
model_svm_rb_tune<- ksvm(target~.,train_data_ep_pca,kernel = "rbfdot",C=2,kpar =list(sigma=0.25))

pred_svm_rb_tune<-predict(model_svm_rb_tune,test_data_ep_pca)
confusionMatrix(pred_svm_rb_tune,test_data_ep_pca$target,positive = "1")


# apply the logistic model

model_log<-glm(target~.,train_data_ep,family = "binomial")

summary(model_log)

model_log_aic <-stepAIC(model_log)
# predict on train 
prob_train <-predict(model_log,train_data_ep, type = "response")


# apply the prediction function on the prob_train

pred_log_tr<-prediction(prob_train,train_data_ep$target)

library(ROCR)
library(MASS)

perf_log_tr<-performance(pred_log_tr,measure = "tpr",x.measure = "fpr")

plot(perf_log_tr,col = rainbow(10),colorize = T,print.cutoffs.at = seq(0,1,0.05))
