# -*- coding: utf-8 -*-
import jieba			#msg separation
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix,roc_curve,auc
import langconv #Convert traditional chinese to simpliest chinese
# =============================================================================
# Parameters
# =============================================================================
#Data
train_n="normal_train.txt" #Normal msg for training 
train_s="spam_train.txt"   #Spam msg for training
test_n="normal_test.txt" #Normal msg for testing
test_s="spam_test.txt" #Spam msg for testing

#Output
Words_n="wnormal.txt" #keywords and freq extracted from normal
Words_s="wspam.txt"  #keywords and freq extracted from spam

##initialize variables
nwordDic={}	#keyword dictionary for normal msg
swordDic={}	#keyword dictionary for spam msg


def load_database(DBnames,DBvar):
    """load data from database, create a empty database if there is no exisiting database
    Input:
        DBnames: List of String
        DBvar  : List of dictionary, to store the DB value 
    """
    count=0
    for name in DBnames:
        #create a empty databaseif there is no exisiting database
        if not os.path.isfile(name):
            dbfile=open(name,'w+')
            dbfile.close()
            print "empty database for "+name+ " was created"
        #load data from existing database
        else:
            with open(name) as dbfile:
                for line in dbfile:
                    line = line.rstrip().decode('utf-8')
                    listedline = line.split(' = ') # split around tab sign
                    DBvar[count][listedline[0]]= int(listedline[1])
        count+=1
    
    return DBvar

def get_words(msg):
    """
    split a message into a set of keywords
    Input:
        message: String
    Return:
        words_set:list of keywords
    """
    #convert msg into simplified chinese
    langconv.Converter('zh-hans').convert(msg.decode('utf-8'))	
    msg = msg.replace('\n','')
    #ignore upper and lower case
    msg = msg.lower()										
    #separate into a list of keywords using jieba
    wordslist_org= list(jieba.cut(msg, cut_all=False))
    words_set = set(wordslist_org)
    words_set.discard(' ')
    words_set = filter(None, words_set)
    
    return words_set


def train(filename,dicttosave):
    """
    get the frequency of each words
    Input:
        filename: String, file record each message
        dicttosave: dictionary, to record the frequency
    """
    with open(filename) as reader:
         count=0
         for line in reader:
             count+=1
             words_set=get_words(line) #get keywords
             
             #update the keyword dictionary
             for value in words_set:
                 if value in dicttosave:
                     dicttosave[value]+=1
                 else:
                    dicttosave[value]=1
    return count

def exportCount(filename,dicttoexport):
    """
    output the frequency of each words to a text file
    Input:
        filename: String, export data to this file
        dicttoexport: dictionary, where the frequency stored in
    """
    output=open(filename,'w+')
    for keys in dicttoexport:
        output.write(keys.encode("UTF-8")+" = ")
        output.write(str(dicttoexport[keys]))
        output.write("\n")
    output.close()
    
    
def predict_prob(filename):
    """
    predict the spam probability of each message in the given file
    Input:
        filename: String
    """
    #create log file to store the information of each message
    logfile=open("log_"+filename,'w+')
    #store the spam probability of each message
    message_prob=[]
    
    #read each msg
    with open(filename) as reader:
        count=0
        for line in reader:
            spam_prob,p_dic=predict_1msg(line)
            message_prob.append(spam_prob)
            
            #export log to txt file
            logfile.write("probability = "+ str(spam_prob)+"\n"+line)
            output_string=""
            for keys in p_dic:
                output_string+=keys + ":" + str(p_dic[keys]) +", "
            logfile.write(output_string.encode("UTF-8")+"\n \n")	
            count+=1
    logfile.close()
    return message_prob,count

def predict_1msg(message):
    """
    predict one single message spam probability
    Input: 
        message: String
    Return:
        total_prob: the spam probability of the input message
        p_dic: the partial probability of each keywords
    """
    #split into keywords
    words_set=get_words(message)
    #initialize keyword probability dictionary
    p_dic={}
    for values in words_set:
        #calculate p(spam|word) for each keyword 
        p_dic[values]=cal_partial_prob(values)
    
    total_prob=cal_total_prob(p_dic)
    
    return total_prob,p_dic
    
def cal_partial_prob(word):
    """
    calculate and return the partial probaiblity p(spam|word)
    Input:
        word: String
        
    Avoid extreme value:
    #range: 0.01<=p(spam|word)<=0.99
    #p(spam|word) = 0.4 if it the does not exist in spam and normal database
	#p(spam|word) = 0.99 if it only exist in spam database
	#p(spam|word) = 0.01 if it only exist in normal database
    """    
    #word appear in both database
    if word in swordDic and word in nwordDic:
        spam_freq= swordDic[word] #frequency in spam message
        norm_freq= nwordDic[word] #frequency in normal message
        spam_prob=spam_freq/float(spam_freq+norm_freq)
    #word only exist in spam database
    elif word in swordDic and word not in nwordDic :
        spam_freq= swordDic[word] #frequency in spam message
        spam_prob=0.99 if spam_freq<=2 else 0.4
    
    #word only exist in normal database
    elif word not in swordDic and word in nwordDic :
        norm_freq= nwordDic[word] #frequency in normal message
        spam_prob=0.01 if norm_freq<=2 else 0.4
        
     #word does not exist in both database
    elif not(word in swordDic and word in nwordDic):
        spam_prob=0.4

    #ensure the probability range within [0.01,0.99]
    spam_prob=min(0.99,max(spam_prob,0.01))
    
    return spam_prob
        
def cal_total_prob(p_dic):
    """
    calculate the total spam probaility
    Final Spam Prob=product(p_dic[i])/ (product(p_dic[i])+product(1-p_dic[i]))
    Input:
        p_dic: dictionary, keyword probability
        n: number of keywords
    """    
    #turn dictionary into list, remains the first n highest term
    p_dic_list=sorted(p_dic.values(), reverse=True)
    one_minus_dic_list=[1-value for value in p_dic_list]
    
    #calculation the Final Spam Prob
    FinalProb=float(np.product(p_dic_list))/(np.product(p_dic_list)+np.product(one_minus_dic_list))
    
    return FinalProb

def findOptimalCutoff(target,prob):
    """
    Find the optimal probaiblity cutoff point for a classification parameters
    Input:
        target: List of actual result 
        predicted: List of predicted result from model
    """
    fpr,tpr,threshold=roc_curve(target,prob)
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
    
#load data/ get database ready        
nwordDic,swordDic=load_database([Words_n,Words_s],[nwordDic,swordDic])

# =============================================================================
# Config
# =============================================================================
#Train with 
if len(nwordDic)==0 and len(swordDic)==0:
    TrainNewData="Y"
else:
    print "Current No. of Normal Data: "+str(len(nwordDic))
    print "Current No. of Spam Data "+str(len(swordDic))
    TrainNewData=raw_input('Train New Data? (Y/N) ') or "N"

    

# =============================================================================
# Train data
# =============================================================================

if TrainNewData=="Y":
    #Normal Message
    nnmsg=train(train_n,nwordDic)
    exportCount(Words_n,nwordDic)
    #spam message
    nsmsg=train(train_s,swordDic)
    exportCount(Words_s,swordDic)
    print "number of normal msg trained: " + str(nnmsg)
    print "number of spam msg trained: " + str(nsmsg)

#get the probability of training data
prob_n,count_n=predict_prob(train_n)
prob_s,count_s=predict_prob(train_s)
train_type=[0]*count_n+[1]*count_s
train_spam_prob=prob_n+prob_s
#get the optimal threshold 
threshold=findOptimalCutoff(train_type,train_spam_prob)

# =============================================================================
# Predict with testing data
# =============================================================================
#predict normal data
prob_n,count_n=predict_prob(test_n)
#predict spam data
prob_s,count_s=predict_prob(test_s)
print "number of normal msg tested: " + str(count_n)
print "number of spam msg tested: " + str(count_s)
#combine normal and spam data 
spam_type=[0]*count_n+[1]*count_s
spam_prob=prob_n+prob_s
spam_predict=[1 if prob>threshold else 0 for prob in spam_prob]
getConfusionMatrix(spam_type,spam_predict)
