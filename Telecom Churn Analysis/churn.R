#Setting
library(corrplot)
library(car)
library(e1071)
library(caret)
library(pROC)
library(randomForest)

#read in the dataset
df<-read.csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
str(df)
summary(df)

#Search if any missing value
colSums(is.na(df))
#delete rows with missing value
df<-na.omit(df)

#convert SeniorCitizen to factor
df$SeniorCitizen=factor(ifelse(df$SeniorCitizen==0,"No","Yes"))

#Exploratory Data Analysis
par(mfrow=c(3,3))
plot(df$gender,df$Churn,xlab="Gender",ylab="Churn")
plot(df$SeniorCitizen,df$Churn,xlab="SeniorCitizen",ylab="Churn")
plot(df$Partner,df$Churn,xlab="Partner",ylab="Churn")
plot(df$Dependents,df$Churn,xlab="Dependents",ylab="Churn")
plot(df$PhoneService,df$Churn,xlab="PhoneService",ylab="Churn")
plot(df$MultipleLines,df$Churn,xlab="MultipleLines",ylab="Churn")
plot(df$InternetService,df$Churn,xlab="InternetService",ylab="Churn")
plot(df$OnlineSecurity,df$Churn,xlab="OnlineSecurity",ylab="Churn")
plot(df$OnlineBackup,df$Churn,xlab="OnlineBackup",ylab="Churn")
par(mfrow=c(2,3))
plot(df$DeviceProtection,df$Churn,xlab="DeviceProtection",ylab="Churn")
plot(df$TechSupport,df$Churn,xlab="TechSupport",ylab="Churn")
plot(df$StreamingTV,df$Churn,xlab="StreamingTV",ylab="Churn")
plot(df$StreamingMovies,df$Churn,xlab="StreamingMovies",ylab="Churn")
plot(df$Contract,df$Churn,xlab="Contract",ylab="Churn")
plot(df$PaperlessBilling,df$Churn,xlab="PaperlessBilling",ylab="Churn")
par(mfrow=c(2,2))
plot(df$PaymentMethod,df$Churn,xlab="PaymentMethod",ylab="Churn")
plot(df$Churn,df$tenure,xlab="Churn",ylab="tenure")
plot(df$Churn,df$MonthlyCharges,xlab="Churn",ylab="MonthlyCharges")
plot(df$Churn,df$TotalCharges,xlab="Churn",ylab="TotalCharges")

#check correlation between tenure,MonthlyCharges,TotalCharges
corrplot(cor(df[sapply(df,is.numeric)]))

#Data preprocessing
#convert numerical variable: tenure,MonthlyCharges to category variable
df$tenure.0.to.1yr<-factor(ifelse(df$tenure<=12,"Yes","No"))
df$tenure.1.to.2yr<-factor(ifelse(df$tenure>12 & df$tenure<=24,"Yes","No"))
df$tenure.2.to.3yr<-factor(ifelse(df$tenure>24 & df$tenure<=36,"Yes","No"))
df$tenure.3.to.4yr<-factor(ifelse(df$tenure>36 & df$tenure<=48,"Yes","No"))
df$tenure.4.to.5yr<-factor(ifelse(df$tenure>48 & df$tenure<=60,"Yes","No"))
df$tenure.5.to.6yr<-factor(ifelse(df$tenure>60 & df$tenure<=72,"Yes","No"))
summary(df$MonthlyCharges)
df$MonthlyCharges_c1<-factor(ifelse(df$MonthlyCharges<=35,"Yes","No"))
df$MonthlyCharges_c2<-factor(ifelse(df$MonthlyCharges>35 & df$MonthlyCharges<=60,"Yes","No"))
df$MonthlyCharges_c3<-factor(ifelse(df$MonthlyCharges>60 & df$MonthlyCharges<=80,"Yes","No"))
df$MonthlyCharges_c4<-factor(ifelse(df$MonthlyCharges>80,"Yes","No"))
#Combine "No Phone Service" with "No" & "No Internet Service" with "No"
df<- data.frame(lapply(df, function(x) { gsub("No phone service", "No", x)}))
df<- data.frame(lapply(df, function(x) { gsub("No internet service", "No", x)}))


#convert multicategorical variable to dummy variable
summary(df$InternetService)
df$InternetService.DSL=factor(ifelse(df$InternetService=="DSL",1,0))
df$InternetService.FO=factor(ifelse(df$InternetService=="Fiber optic",1,0))
df$InternetService.No=factor(ifelse(df$InternetService=="No",1,0))
summary(df$Contract)
df$Contract.MTM=factor(ifelse(df$Contract=="Month-to-month",1,0))
df$Contract.1Y=factor(ifelse(df$Contract=="One year",1,0))
df$Contract.2Y=factor(ifelse(df$Contract=="Two year",1,0))
summary(df$PaymentMethod)
df$PaymentMethod.BT=factor(ifelse(df$PaymentMethod=="Bank transfer (automatic)",1,0))
df$PaymentMethod.CC=factor(ifelse(df$PaymentMethod=="Credit card (automatic)",1,0))
df$PaymentMethod.EC=factor(ifelse(df$PaymentMethod=="Electronic check",1,0))
df$PaymentMethod.MC=factor(ifelse(df$PaymentMethod=="Mailed check",1,0))

#divide into training and testing set
set.seed(1)
TelcoCustomer.flag<-sample(1:nrow(df),0.7*nrow(df),replace=F)
TelcoCustomer.train<-df[TelcoCustomer.flag,]
TelcoCustomer.test<-df[-TelcoCustomer.flag,]

#function to update the model formula
updateModel<-function(varlist){
  model<-as.formula(paste("Churn",paste(varlist,collapse = "+"),sep="~"))
  New.logit<-glm(model,family=binomial,data=TelcoCustomer.train)
  print (summary(New.logit))
  print(vif(New.logit))
  return (model)
}

#train model with stepwise regression
varlist<-names(TelcoCustomer.train)
excluded<-c("Churn","customerID","tenure","InternetService","Contract","PaymentMethod","MonthlyCharges","TotalCharges" )
varlist<-varlist[!varlist %in% excluded]
model0 <- as.formula(paste("Churn",paste(varlist,collapse = "+"),sep="~"))
logit<-glm(model0,family=binomial,data=TelcoCustomer.train)
summary(logit)
step(logit,direction="both")
varlist.new<-c("SeniorCitizen","Dependents","PhoneService","MultipleLines","OnlineSecurity","OnlineBackup","TechSupport", 
                     "StreamingTV","StreamingMovies","PaperlessBilling","tenure.0.to.1yr", 
                     "tenure.1.to.2yr","tenure.2.to.3yr","tenure.3.to.4yr","MonthlyCharges_c1", 
                     "MonthlyCharges_c2","MonthlyCharges_c3","InternetService.DSL","InternetService.FO","Contract.MTM","Contract.1Y","PaymentMethod.EC")
model1<-updateModel(varlist.new)
varlist.new<-varlist.new[varlist.new!="Dependents"]
model2<-updateModel(varlist.new)
varlist.new<-varlist.new[varlist.new!="MonthlyCharges_c1"]
model3<-updateModel(varlist.new)
varlist.new<-varlist.new[varlist.new!="MonthlyCharges_c3"]
model4<-updateModel(varlist.new)
varlist.new<-varlist.new[varlist.new!="InternetService.DSL"]
model5<-updateModel(varlist.new)
varlist.new<-varlist.new[varlist.new!="Contract.1Y"]
model6<-updateModel(varlist.new)
varlist.new<-varlist.new[varlist.new!="tenure.1.to.2yr"]
model7<-updateModel(varlist.new)
varlist.new<-varlist.new[varlist.new!="InternetService.FO"]
model8<-updateModel(varlist.new)
varlist.new<-varlist.new[varlist.new!="tenure.2.to.3yr"]
model9<-updateModel(varlist.new)
varlist.new<-varlist.new[varlist.new!="tenure.3.to.4yr"]
model10<-updateModel(varlist.new)
varlist.new<-varlist.new[varlist.new!="PhoneService"]
model11<-updateModel(varlist.new)
varlist.new<-varlist.new[varlist.new!="MonthlyCharges_c2"]
model12<-updateModel(varlist.new)
Final.logit<-glm(model12,family=binomial,data=TelcoCustomer.train)
#Using Final Model:Model12 to predict the probability of a customer will be churn 
TelcoCustomer.train$Pred_prob<-predict(Final.logit,TelcoCustomer.train,type=c("response"))

#Find the optimal threshold that minimize the difference between Sensitivity and Specificity
myRoc <- roc(response = (TelcoCustomer.train$Churn), predictor = (TelcoCustomer.train$Pred_prob), positive = 'Yes')
#Area Under Curve
myRoc$auc
result<-data.frame(myRoc$sensitivities, myRoc$specificities, myRoc$thresholds)
optimalCutOff=0
minDiff=1000;
for (i in 1:nrow(result)){
  output<-c (result[i,1],result[i,2],result[i,3])
  if (result[i,1]>result[i,2] & abs(result[i,1]-result[i,2])<minDiff){
    minDiff<-abs(result[i,1]-result[i,2])
    optimalCutOff<-result[i,3]
  }
}
optimalCutOff
TelcoCustomer.train$Pred_Churn<-ifelse(TelcoCustomer.train$Pred_prob>optimalCutOff,"Yes","No")
confusionMatrix(data=factor(TelcoCustomer.train$Pred_Churn),reference = factor(TelcoCustomer.train$Churn),positive = 'Yes')
par(mfrow=c(1,1))
plot(myRoc)
#Apply to test data 
TelcoCustomer.test$Pred_prob<-predict(Final.logit,TelcoCustomer.test,type=c("response"))
TelcoCustomer.test$Pred_Churn<-ifelse(TelcoCustomer.test$Pred_prob>optimalCutOff,"Yes","No")
confusionMatrix(data=factor(TelcoCustomer.test$Pred_Churn),reference = factor(TelcoCustomer.test$Churn),positive = 'Yes')


#Random Forest
TelcoCustomer.rf<-randomForest(model0,TelcoCustomer.train,ntree=500,importance=T)
plot(TelcoCustomer.rf)
#Variable Importance 
varImpPlot(TelcoCustomer.rf,sort=T,main="Variable Importance",n.var = 10)
var.imp.test<-data.frame(importance(TelcoCustomer.rf,type=2))
var.imp.test$Variables<-row.names(var.imp.test)
var.imp.test[order(var.imp.test$MeanDecreaseGini,decreasing=T),]
#Predicting response Variable
TelcoCustomer.train$rf.predict<-predict(TelcoCustomer.rf,TelcoCustomer.train)
TelcoCustomer.train$rf.predict.value<-predict(TelcoCustomer.rf,TelcoCustomer.train,type="prob")
confusionMatrix(data=factor(TelcoCustomer.train$rf.predict),reference=factor(TelcoCustomer.train$Churn),positive="Yes")
myRoc <- roc(response = (TelcoCustomer.train$Churn), predictor = (TelcoCustomer.train$rf.predict.value), positive = 'Yes')
#Area Under Curve
myRoc$auc
result<-data.frame(myRoc$sensitivities, myRoc$specificities, myRoc$thresholds)
rf.optimalCutOff=0
minDiff=1000;
for (i in 1:nrow(result)){
  output<-c (result[i,1],result[i,2],result[i,3])
  if (result[i,1]>result[i,2] & abs(result[i,1]-result[i,2])<minDiff){
    minDiff<-abs(result[i,1]-result[i,2])
    rf.optimalCutOff<-result[i,3]
  }
}
rf.optimalCutOff



#Predict of testing data
TelcoCustomer.test$rf.Pred.prob<-predict(TelcoCustomer.rf,TelcoCustomer.test,type="prob")[,2] #the Probabilty of Yes
TelcoCustomer.test$rf.predict<-ifelse(TelcoCustomer.test$rf.Pred.prob>rf.optimalCutOff,"Yes","No") 
confusionMatrix(data=factor(TelcoCustomer.test$rf.predict),reference=factor(TelcoCustomer.test$Churn),positive="Yes")
myRoc <- roc(response = (TelcoCustomer.test$Churn), predictor = (TelcoCustomer.test$rf.Pred.prob), positive = 'Yes')
#Area Under Curve
myRoc$auc
