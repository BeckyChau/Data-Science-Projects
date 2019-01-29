#import libraries
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,roc_curve,auc
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# =============================================================================
# Data Overview
# =============================================================================
#read in the dataset
df=pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv",na_values=[" "])
#check the basic information
print (df.head(5))
print ("Rows:",df.shape[0]," Columns:",df.shape[1])
print (df.describe())
print (df.info())

def get_unique(x):
    """
    Print the unique value if the number of unqiue value less than 20
    Input: 
        x: DataFrame
    """
    print ("Column,dType,UniqueNo,UniqueValues")
    for column in x:
        if x[column].nunique()<20:
            print (column,x[column].dtype,x[column].nunique(),x[column].unique().tolist())
        else:
            print (column,x[column].dtype,x[column].nunique())
    print ("******************************")
get_unique(df)

# =============================================================================
# Data Cleaning
# =============================================================================
#convert SeniorCitizen to object
df['SeniorCitizen']=df['SeniorCitizen'].replace({1:"Yes",0:"No"})

def replaceValue(x,oldValues,newValue):
    """Replace value under given condition
    Input: 
        x: DataFrame
        oldValues: List of String/Integer 
        newValue: String/Integer
    """
    for column in x:
        for value in oldValues:
            if x[column].unique().tolist().count(value)>0:
                x[column]=x[column].replace({value:newValue})
    return x 

#replace "No Phone Service" & "No Internet Service" with "No"
df=replaceValue(df,["No phone service","No internet service"],"No")
get_unique(df)
#Search if any missing value 
print ("Missing values per column:")
print (df.isnull().sum())
#delete rows with missing value 
df=df.dropna()
#delete the customerID column
df=df.drop('customerID',1)

# =============================================================================
# Exploratory Data Analysis
# =============================================================================
def plotGraphs(x,target):
    """Plot the relationship between parameters & target
    Input:
        x: DataFrame
        target: String (Column name of the target value)
    """
    for column in x:
        if x[column].dtype.name=="object" and column!=target:
            ct=pd.crosstab(x[column],x[target],normalize="index")  #normalize over each row 
            ct.plot.bar(stacked=True)
            plt.legend(title=target)
            plt.show()
        elif column!=target:
            print (column)
            sns.boxplot(x="Churn",y=column,data=x)
            plt.show()
plotGraphs(df,"Churn")   #plot barchart & boxplot
#check correlation between numerical variables:
sns.heatmap(df.corr())
plt.show()

#drop TotalCharges column
df=df.drop('TotalCharges',1)

# =============================================================================
# Data Preprocessing
# =============================================================================
def convert_lab(x,column,ValueRange,Label):
    """Convert numberical variable into categorical variables
    Input:
        x: DataFrame
        column: String  (column name of parameters)
        ValueRange: List of Integer (the Upper bound of each category) 
    """
    for i in range(len(Label)):
        if len(ValueRange)==len(Label) or i<=len(ValueRange)-1:
            if x[column]<=ValueRange[i]:
                return Label[i]
        else:
             return Label[i]
         
#Divide tenure& MonthlyCharges into groups
df["tenure_group"]=df.apply(lambda df:convert_lab(df,"tenure",[12,24,36,48,60,72],["tenure_0-1yr","tenure_1-2yr","tenure_2-3yr","tenure_3-4yr","tenure_4-5yr","tenure_5-6yr"]),axis=1)            
df["MonthlyCharges_group"]=df.apply(lambda df:convert_lab(df,"MonthlyCharges",[35,60,80],["c1","c2","c3","c4"]),axis=1)            
 #drop the original tenure,MonthlyCharges Column 
df=df.drop(["tenure","MonthlyCharges"],1)

#Set the target column and parameter columns: binary column/ multiclass column 
target_col=["Churn"]
all_cols=list(df)
bin_cols=df.nunique()[df.nunique()==2].keys().tolist()
multi_cols=[col for col in all_cols if col not in (bin_cols+target_col)]

le=LabelEncoder()
#Label encoding Binary columns
for i in bin_cols:
    df[i]=le.fit_transform(df[i])
#Duplicate columns for multi value columns
df=pd.get_dummies(data=df,columns=multi_cols)
#check Overall Correlation 
sns.heatmap(df.corr())
plt.show()

#split into features and result
X=df.drop(['Churn'],axis=1)
y=df['Churn']
#split into training & testnig set
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state =20000)

# =============================================================================
# Build Model
# =============================================================================

def FindOptimalCutOff(target,X,classifier):
    """
    Find the optimal probaiblity cutoff point for a classification parameters
    Input:
        target: List of actual result 
        predicted: List of predicted result from model
        X: feature DataFrame
    """
    df=pd.DataFrame() #dataFrame to record the result 
    classifier.fit(X,target) #fit the model
    df['y_predict_prob']=classifier.predict_proba(X)[:,1] #get the predicted prob of training data
    fpr,tpr,threshold=roc_curve(target,df['y_predict_prob'])
    i=np.arange(len(tpr))
    roc=pd.DataFrame({'tf':pd.Series(tpr-(1-fpr),index=i),'threshold':pd.Series(threshold,index=i)})
    roc_t=roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    return list(roc_t['threshold'])[0]

def getConfusionMatrix(target,predicted):
    """
    Confusion Matrix and Statistics: Accuracy, sensitivity , specificty, AUC
    Input:
        target: List of actual result 
        predicted: List of predicted result from model
    """
    cm=confusion_matrix(target,predicted)
    print ("Confusion Matrix : \n",cm)
    accuracy=accuracy_score(target,predicted)
    print ('Accuracy: {:.2f}'.format(accuracy))
    sensitivity=cm[0,0]/float(cm[0,0]+cm[0,1])
    print ('Sensitivity: {:.2f}'.format(sensitivity))
    specificity=cm[1,1]/float(cm[1,0]+cm[1,1])
    print ('Specificity: {:.2f}'.format(specificity))
    fpr,tpr,thresholds=roc_curve(target,predicted)
    Auc_value=auc(fpr,tpr)
    print ('Area Under Curve: {:.2f}'.format(Auc_value))
    
#[Model1: Only RandomForest]
rfr=RandomForestClassifier(n_estimators=500,random_state=1)
rfr.fit(X_train,y_train) #fit the model with training data
threshold_rf=FindOptimalCutOff(y_train,X_train,rfr) #get the threshold 
print ("Threshold: ",threshold_rf)
y_predict=rfr.predict(X_test) #predict the testing data
rf_o=pd.DataFrame() #dataFrame to record the result 
rf_o['y_predict_prob']=rfr.predict_proba(X_test)[:,1] #get the predicted prob of testing data
rf_o['y_predict']=rf_o['y_predict_prob'].map(lambda x:1 if x>threshold_rf else 0) #get the classification
getConfusionMatrix(y_test,rf_o['y_predict']) #get the confusion matrix result 

#[Model2: PCA+RandomForest]
#Principal Components [PCA]
pca=PCA()
pca.fit(X_train) #find the principal components
ex_var_ratio=pca.explained_variance_ratio_ # the amount of variance that each PC explains
ex_var_ratio_cum=np.cumsum(np.round(ex_var_ratio,decimals=4)*100)#Cumulative Variance explains
#print the cumulative variance with different no. of PC
for i in range(len(ex_var_ratio_cum)):
    print (i+1,round(ex_var_ratio_cum[i],2))
plt.plot(ex_var_ratio_cum)
plt.show()
pca=PCA(n_components=25) #Select 25 variables which variance ~98.32
pca.fit(X_train) #find the principal components
X_train_pca=pca.fit_transform(X_train) 
print (sum(pca.explained_variance_ratio_)) #Total variance
#training data
rfr.fit(X_train_pca,y_train) #fit the model with training data
threshold_pca=FindOptimalCutOff(y_train,X_train_pca,rfr) #get the threshold
print ("Threshold: ",threshold_pca)
#testing data
X_test_pca=pca.fit_transform(X_test) 
rf_pca=pd.DataFrame() #dataFrame to record the result 
rf_pca['y_predict_prob_pca']=rfr.predict_proba(X_test_pca)[:,1] #get the predicted prob of testing data
rf_pca['y_predict_pca']=rf_pca['y_predict_prob_pca'].map(lambda x:1 if x>threshold_pca else 0) #get the classification
getConfusionMatrix(y_test,rf_pca['y_predict_pca']) #get the confusion matrix result 

#[Model 3: Logistic Regression + Forward Stepwise + VIF]
def forward_stepwise(X,y,inital_list=[],threshold_in=0.01,verbose=True):
    """
    Perform a forward feature selection based on p-value from statsmodes.api.Logit
    INPUT: 
        X: DataFrame of features
        y: DataFrame of targetValue
        inital_list: inital list to start with
        threshold_in: if the p-value smaller than the threshold, include the feature to the model
        verbose: print the config
    RETURN:
        included:columns name of feature that should be included in the final model
    """
    included=list(inital_list)
    printMsg=[]
    changed=True
    while changed:
        changed=False
        excluded=list(set(X.columns)-set(included)) #currently excluded feature list 
        new_pval=pd.Series(index=excluded) #create the p_value list for currently excluded feature
        #fit the Logistics Regression Model with currently includede feature and record the p-value
        for new_column in excluded:
            model=sm.Logit(y,sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column]=model.pvalues[new_column]
        #get the minimum p-value between all currently excluded feature
        best_pval=new_pval.min()
        #add the feature to included list if the p_value is smaller than the threshold
        if best_pval<threshold_in:
            best_feature=new_pval.argmin()
            included.append(best_feature)
            changed=True
            if verbose:
                printMsg.append('Add {:30} with p-value {:.6}'.format(best_feature,best_pval))
                print ('Add {:30} with p-value {:.6}'.format(best_feature,best_pval))
                
    return included,printMsg
            
def calculate_VIF(X,threshold=5):
    """
    calculate the VIF for each features
    drop all the feature larger than threshold
    INPUT:
        X: Dataframe (all features)
    RETURN:
        Dataframe with some features removed
    """
    dropped=True #whether drop any feature
    variables=np.arange(X.shape[1])
    cols=X.columns
    while dropped:
        dropped=False
        vif=[variance_inflation_factor(X.values,i) for i in variables]
        max_vif=max(vif)
        if max_vif>threshold:
            maxloc=vif.index(max_vif)
            print ('dropping \' '+ X[cols[variables]].columns[maxloc]+'\' at index: '+str(maxloc))
            variables=np.delete(variables,maxloc)
            dropped=True
    
    print ('Remaining variables: ')
    print (X.columns[variables])
    included=X.columns[variables]
    return included

def print_VIF(X):
    """
    print the vif table 
    INPUT:
        X: DataFrame (features)
    """
    vif=pd.DataFrame()
    vif["VIF Factor"]=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
    vif["features"]=X.columns #feature names
    print (vif.round(2))

def getLogitResult(X,y):
    """
    print the Logit Model summary
    INPUT:
        X: DataFrame (features)
        y: target 
    """
    logit_model=sm.Logit(y,sm.add_constant(X))
    sm_result=logit_model.fit()
    print (sm_result.summary2())
    return sm_result

def RemoveHighPval(X,y,threshold=0.05):
    """
    Remove the feature with p-value higher than 0.05
    INPUT: 
        X: DataFrame of features
        y: DataFrame of targetValue
        threshold: if the p-value higher than the threshold, excluded the feature to the model
    RETURN:
        included:columns name of feature that should be included in the final model
    """
    changed=True
    included=list(set(X.columns)) #currently included feature list 
    while changed:
        changed=False
        new_pval=pd.Series(index=included) #create the p_value list for currently included feature
        #fit the Logistics Regression Model with currently includede feature and record the p-value
        for column in included:
            model=sm.Logit(y,sm.add_constant(pd.DataFrame(X[included]))).fit()
            new_pval[column]=model.pvalues[column]
        #get the maximum p-value between all currently excluded feature
        worst_pval=new_pval.max()
        print (model.summary2())
        print (new_pval)
        #add the feature to included list if the p_value is smaller than the threshold
        if worst_pval>threshold:
            worst_feature=new_pval.argmax()
            included.remove(worst_feature)
            changed=True
                
    return included

#Feature selection by Forward Stepwise and VIF
included,msg=forward_stepwise(X_train,y_train)
X_train_FS=X_train[included]
sm_result=getLogitResult(X_train_FS,y_train)
included=calculate_VIF(X_train_FS)
X_train_FS=X_train_FS[included]
sm_result=getLogitResult(X_train_FS,y_train)

#Training Data
classifier=LogisticRegression()
classifier.fit(X_train_FS,y_train) #fit the model with training data
threshold=FindOptimalCutOff(y_train,X_train_FS,classifier) #get the optimal threshold from training data
print ("Threshold: ",threshold)
#classifier.coef_  #coefficient of each features
#classifier.intercept_ #value of the constants
#testing data
X_test_LRFS=X_test[included]
LR_FS=pd.DataFrame() #dataFrame to record the result 
LR_FS['y_predict_prob_LR']=classifier.predict_proba(X_test_LRFS)[:,1] #get the predicted prob of testing data
LR_FS['y_predict_LR']=LR_FS['y_predict_prob_LR'].map(lambda x:1 if x>threshold else 0) #get the classification
getConfusionMatrix(y_test,LR_FS['y_predict_LR']) #get the confusion matrix result 


#[Model 4: Logistic Regression + Backward Stepwise + VIF]
def backward_stepwise(X,y,inital_list=[],threshold_out=0.05,verbose=True):
    """
    Perform a backward feature selection based on p-value from statsmodes.api.Logit
    INPUT: 
        X: DataFrame of features
        y: DataFrame of targetValue
        inital_list: inital list to start with
        threshold_out: if the p-value larger than the threshold, exclude the feature to the model
        verbose: print the config
    RETURN:
        included:columns name of feature that should be included in the final model
    """
    included=list(set(X.columns)) #currently included feature list 
    printMsg=[]
    changed=True
    while changed:
        changed=False
        model=sm.Logit(y,sm.add_constant(pd.DataFrame(X[included]))).fit() #fit the Logistics Regression Model with currently includede feature 
        new_pval=model.pvalues.iloc[1:] #p_value of all coefs except intercept
        #get the maximumu p-value between all currently included feature
        worst_pval=new_pval.max()
        #add the feature to included list if the p_value is smaller than the threshold
        if worst_pval>threshold_out:
            changed=True
            worst_feature=new_pval.argmax()
            included.remove(worst_feature)
            if verbose:
                  printMsg.append('Drop {:30} with p-value {:.6}'.format(worst_feature,worst_pval))
                  print ('Drop {:30} with p-value {:.6}'.format(worst_feature,worst_pval))
    return included,printMsg

#Feature selection by Backward Stepwise and VIF
included,msg=backward_stepwise(X_train,y_train)
X_train_bs=X_train[included]
sm_result=getLogitResult(X_train_bs,y_train)
included=calculate_VIF(X_train_bs)
X_train_bs=X_train_bs[included]
sm_result=getLogitResult(X_train_bs,y_train)
included=RemoveHighPval(X_train_bs,y_train)
X_train_bs=X_train_bs[included]
sm_result=getLogitResult(X_train_bs,y_train)

#Training Data
classifier=LogisticRegression()
classifier.fit(X_train_bs,y_train) #fit the model with training data
threshold_BS=FindOptimalCutOff(y_train,X_train_bs,classifier) #get the optimal threshold from training data
print ("Threshold: ",threshold_BS)
#classifier.coef_  #coefficient of each features
#classifier.intercept_ #value of the constants
#testing data
X_test_LRBS=X_test[included]
LR_BS=pd.DataFrame() #dataFrame to record the result 
LR_BS['y_predict_prob_LR']=classifier.predict_proba(X_test_LRBS)[:,1] #get the predicted prob of testing data
LR_BS['y_predict_LR']=LR_BS['y_predict_prob_LR'].map(lambda x:1 if x>threshold_BS else 0) #get the classification
getConfusionMatrix(y_test,LR_BS['y_predict_LR']) #get the confusion matrix result 
