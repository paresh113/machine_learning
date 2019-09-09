# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 14:29:02 2019

@author: paresh
"""
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm 

    
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced


#import dataset

dataset = pd.read_csv('E:\kushboo_apu\Churn_Modelling.csv')
X= dataset.iloc[:,:-1].values
y= dataset.iloc[:,13].values


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_0 = LabelEncoder()
X[:, 2] = labelencoder_X_0.fit_transform(X[:, 2])
labelencoder_X_1 = LabelEncoder()
X[:, 4] = labelencoder_X_1.fit_transform(X[:, 4])
labelencoder_X_2 = LabelEncoder()
X[:, 5] = labelencoder_X_2.fit_transform(X[:, 5])

#onehotencoder = OneHotEncoder(categorical_features = "all")
#X = onehotencoder.fit_transform(X).toarray()




#spliting dataset into testing and training dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=1/5,random_state = 0 )

##############    applying decission tree classifier   #####################
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)

# create confuson matrix for DT 
from sklearn.metrics import confusion_matrix
cm_dt = confusion_matrix(y_test, y_pred.round())
print('confuson matrix for Decission Tree :\n',cm_dt)

#precision and recall and support for DT 
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

#find out accuracy  for DT
from sklearn.metrics import accuracy_score
score_dt =  accuracy_score(y_test, y_pred.round())
print("Accuracy score of decission tree:", score_dt)

########################    precision-recall curve and f1 for applying DT before balancing  ##########################
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from matplotlib import pyplot


model = DecisionTreeClassifier()
model.fit(X_train,y_train)
# predict probabilities
probs = model.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# predict class values
yhat = model.predict(X_test)
# calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, probs)
# calculate F1 score
f1 = f1_score(y_test, yhat)
# calculate precision-recall AUC
auc = auc(recall, precision)
# calculate average precision score
ap = average_precision_score(y_test, probs)
print('before balancing  : f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))
# plot no skill
plt.figure(figsize=(7,7))
pyplot.plot([0, 1], [0.5, 0.5], linestyle='--')
# plot the precision-recall curve for the model
pyplot.plot(recall, precision, marker='.')
# show the plot
plt.xlabel('recall')
plt.ylabel('precision ')
pyplot.show()

############################    ROC CURVE  for applying DT before balancing ###########################

from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot


# fit a model before balancing 

model.fit(X_train,y_train)
# predict probabilities
probs = model.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
auc = roc_auc_score(y_test, probs)
print(' before balanacing AUC: %.3f' % auc)
plt.figure(figsize=(7,7))
# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
# plot no skill
pyplot.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
pyplot.plot(fpr, tpr, marker='.')
# show the plot
plt.xlabel('false positive rate ')
plt.ylabel('true positive rate ')
pyplot.show()


#############################     logistic regrasion implementation ##################################
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_lr = confusion_matrix(y_test, y_pred.round())
print('confuson matrix for gnb:\n',cm_lr)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

from sklearn.metrics import accuracy_score
score_lr =  accuracy_score(y_test, y_pred.round())
print("Accuracy score of logistic regresion :", score_lr)

########################    precision-recall curve and f1 for applying LR before balancing  ##########################
logreg = LogisticRegression()
model = logreg
model.fit(X_train,y_train)
# predict probabilities
probs = model.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# predict class values
yhat = model.predict(X_test)
# calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, probs)
# calculate F1 score
f1 = f1_score(y_test, yhat)
# calculate precision-recall AUC
#auc = auc(recall, precision)
# calculate average precision score
ap = average_precision_score(y_test, probs)
print('before balancing  : f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))
# plot no skill
plt.figure(figsize=(7,7))
pyplot.plot([0, 1], [0.5, 0.5], linestyle='--')
# plot the precision-recall curve for the model
pyplot.plot(recall, precision, marker='.')
# show the plot
plt.xlabel('recall')
plt.ylabel('precision ')
pyplot.show()

############################    ROC CURVE  for applying LR before balancing ###########################

from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot


# fit a model before balancing 

model.fit(X_train,y_train)
# predict probabilities
probs = model.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
auc = roc_auc_score(y_test, probs)
print(' before balanacing AUC: %.3f' % auc)
plt.figure(figsize=(7,7))
# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
# plot no skill
pyplot.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
pyplot.plot(fpr, tpr, marker='.')
# show the plot
plt.xlabel('false positive rate ')
plt.ylabel('true positive rate ')
pyplot.show()


##################################   applying SVM classifier ################################
from sklearn import svm 
clf = svm.SVC(gamma='scale')
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

# create confuson matrix for SVM
from sklearn.metrics import confusion_matrix
cm_svm = confusion_matrix(y_test, y_pred.round())
print('confuson matrix for Decission Tree :\n',cm_svm)

#precision and recall and support for SVM
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

#find out accuracy  for SVM
from sklearn.metrics import accuracy_score
score_svm = accuracy_score(y_test, y_pred.round())
print("Accuracy score of SVM:", score_svm)


######################## precision-recall curve and f1 for applying SVM before balancing  ##########################
logreg = KNeighborsClassifier()
model = logreg
model.fit(X_train,y_train)
# predict probabilities
probs = model.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# predict class values
yhat = model.predict(X_test)
# calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, probs)
# calculate F1 score
f1 = f1_score(y_test, yhat)
# calculate precision-recall AUC
#auc = auc(recall, precision)
# calculate average precision score
ap = average_precision_score(y_test, probs)
print('before balancing  : f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))
# plot no skill
plt.figure(figsize=(7,7))
pyplot.plot([0, 1], [0.5, 0.5], linestyle='--')
# plot the precision-recall curve for the model
pyplot.plot(recall, precision, marker='.')
# show the plot
plt.xlabel('recall')
plt.ylabel('precision ')
pyplot.show()


############################    ROC CURVE  for applying SVM before balancing ###########################
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot


# fit a model before balancing 
model.fit(X_train,y_train)
# predict probabilities
probs = model.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
auc = roc_auc_score(y_test, probs)
print(' before balanacing AUC: %.3f' % auc)
plt.figure(figsize=(7,7))
# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
# plot no skill
pyplot.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
pyplot.plot(fpr, tpr, marker='.')
# show the plot
plt.xlabel('false positive rate ')
plt.ylabel('true positive rate ')
pyplot.show()



#spliting dataset into testing and training dataset(using Kfold CV)
from sklearn.model_selection import StratifiedKFold
k=5
kf = StratifiedKFold(n_splits=k)

def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


scores_dt = []
scores_lr = []
scores_svm = []
scores_gnb = []


for train_index, test_index in kf.split(X,y):
    #print index
    #print('TRAIN: ',train_index,'TEST: ', test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    #print('X_TRAIN: \n',X_train,'\nX_TEST: \n',X_test)
    scores_dt.append(get_score(DecisionTreeClassifier(), X_train, X_test, y_train, y_test))
    scores_lr.append(get_score(LogisticRegression(), X_train, X_test, y_train, y_test))
    scores_svm.append(get_score(svm.SVC(gamma='scale'), X_train, X_test, y_train, y_test))
    scores_gnb.append(get_score(GaussianNB(), X_train, X_test, y_train, y_test))


scores_dt
scores_lr
scores_svm 
scores_gnb 

sm1 = sum(scores_dt[0:len(scores_dt)])
dt_before_= sm1/k
print('Decision Tree before balancing =',dt_before_)
sm2 = sum(scores_lr[0:len(scores_lr)])
lr_before_= sm2/k
print('LogisticRegression before balancing =',lr_before_)
sm3 = sum(scores_svm[0:len(scores_svm)])
svm_before_ = sm3/k
print('SVM before balancing=',svm_before_)
s4 = sum(scores_gnb[0:len(scores_gnb)])
gnb_before_ = s4/k
print('GaussianNB before balancing =',gnb_before_)



######################## applying RandomOverSampler for balancing #####################

from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(X, y)
from collections import Counter
X = X_resampled 
y = y_resampled

#spliting dataset into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=1/5,random_state = 0 )
print(X.shape, y.shape)



#################################    applying decission tree after balancing ###########################
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)

# create confuson matrix for DT after balancing
from sklearn.metrics import confusion_matrix
cm_dt = confusion_matrix(y_test, y_pred.round())
print('confuson matrix for Decission Tree :\n',cm_dt)

#precision and recall and support for DT after balancing
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

#find out accuracy  for DT after balancing

from sklearn.metrics import accuracy_score
score_dt_ =  accuracy_score(y_test, y_pred.round())
print("Accuracy score of decission tree after : ", score_dt_)

################### precision recall curve of DT  after balancing  ###################
model = DecisionTreeClassifier()
model.fit(X_train,y_train)
# predict probabilities
probs = model.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# predict class values
yhat = model.predict(X_test)
# calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, probs)
# calculate F1 score
f1 = f1_score(y_test, yhat)
# calculate precision-recall AUC
#auc = auc(recall, precision)
# calculate average precision score
ap = average_precision_score(y_test, probs)
print('before balancing  : f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))
# plot no skill
plt.figure(figsize=(7,7))
pyplot.plot([0, 1], [0.5, 0.5], linestyle='--')
# plot the precision-recall curve for the model
pyplot.plot(recall, precision, marker='.')
# show the plot
plt.xlabel('recall')
plt.ylabel('precision ')
pyplot.show()

############################    ROC CURVE  for applying DT After balancing ###########################
model.fit(X_train,y_train)
# predict probabilities
probs = model.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
auc = roc_auc_score(y_test, probs)
print(' before balanacing AUC: %.3f' % auc)
plt.figure(figsize=(7,7))
# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
# plot no skill
pyplot.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
pyplot.plot(fpr, tpr, marker='.')
# show the plot
plt.xlabel('false positive rate ')
plt.ylabel('true positive rate ')
pyplot.show()

##############################    logistic regrasion implementation ################################
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_lr_after = confusion_matrix(y_test, y_pred.round())
print('confuson matrix for gnb:\n',cm_lr_after)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

from sklearn.metrics import accuracy_score
score_lr_ =  accuracy_score(y_test, y_pred.round())
print("Accuracy score of logistic regresion AFTER :", score_lr_)

########################    precision-recall curve and f1 for applying LR after balancing  ##########################
logreg = KNeighborsClassifier()
model = logreg
model.fit(X_train,y_train)
# predict probabilities
probs = model.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# predict class values
yhat = model.predict(X_test)
# calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, probs)
# calculate F1 score
f1 = f1_score(y_test, yhat)
# calculate precision-recall AUC
#auc = auc(recall, precision)
# calculate average precision score
ap = average_precision_score(y_test, probs)
print('before balancing  : f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))
# plot no skill
plt.figure(figsize=(7,7))
pyplot.plot([0, 1], [0.5, 0.5], linestyle='--')
# plot the precision-recall curve for the model
pyplot.plot(recall, precision, marker='.')
# show the plot
plt.xlabel('recall')
plt.ylabel('precision ')
pyplot.show()

############################    ROC CURVE  for applying LR after balancing ###########################
# fit a model after balancing 
model.fit(X_train,y_train)
# predict probabilities
probs = model.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
auc = roc_auc_score(y_test, probs)
print(' before balanacing AUC: %.3f' % auc)
plt.figure(figsize=(7,7))
# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
# plot no skill
pyplot.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
pyplot.plot(fpr, tpr, marker='.')
# show the plot
plt.xlabel('false positive rate ')
plt.ylabel('true positive rate ')
pyplot.show()

##############################################  applying SVM  #################################
from sklearn import metrics
clf = svm.SVC(gamma='scale')
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

# create confuson matrix for SVM
from sklearn.metrics import confusion_matrix
cm_svm = confusion_matrix(y_test, y_pred.round())
print('confuson matrix for Decission Tree :\n',cm_svm)

#precision and recall and support for SVM
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

#find out accuracy  for SVM
from sklearn.metrics import accuracy_score
score_svm_ = accuracy_score(y_test, y_pred.round())
print("Accuracy score of SVM after balancing :", score_svm_)




#########################     spliting dataset into testing and training dataset(using Kfold Cross validation after)
from sklearn.model_selection import StratifiedKFold
k=5
kf = StratifiedKFold(n_splits=k)

def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


scores_dt = []
scores_lr = []
scores_svm = []
scores_gnb = []


for train_index, test_index in kf.split(X,y):
    #print index
    #print('TRAIN: ',train_index,'TEST: ', test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    #print('X_TRAIN: \n',X_train,'\nX_TEST: \n',X_test)
    scores_dt.append(get_score(DecisionTreeClassifier(), X_train, X_test, y_train, y_test))
    scores_lr.append(get_score(LogisticRegression(), X_train, X_test, y_train, y_test))
    scores_svm.append(get_score(svm.SVC(gamma='scale'), X_train, X_test, y_train, y_test))
    scores_gnb.append(get_score(GaussianNB(), X_train, X_test, y_train, y_test))


scores_dt
scores_lr
scores_svm 
scores_gnb 

sm1 = sum(scores_dt[0:len(scores_dt)])
dt_after_= sm1/k
#print('Decision Tree after balancing =',dt_after_)
sm2 = sum(scores_lr[0:len(scores_lr)])
lr_after_= sm2/k
#print('LogisticRegression after balancing =',lr_after_)
sm3 = sum(scores_svm[0:len(scores_svm)])
svm_after_ = sm3/k
#print('SVM after balancing=',svm_after_)
s4 = sum(scores_gnb[0:len(scores_gnb)])
gnb_after_ = s4/k
#print('GaussianNB after balancing =',gnb_after_)

### Accuracy comparison using K-fold cross validation
print('\nAccuracy comparison using K-fold cross validation :')
print('Decision Tree before balancing =',dt_before_)
print('Decision Tree after balancing =',dt_after_)

print('LogisticRegression before balancing =',lr_before_)
print('LogisticRegression after balancing =',lr_after_)

print('SVM before balancing=',svm_before_)
print('SVM after balancing=',svm_after_)

print('GaussianNB before balancing =',gnb_before_)
print('GaussianNB after balancing =',gnb_after_)



###Accuracy comparison using train test split
print('\nAccuracy comparison using train test split : ')
print("Accuracy score of decission tree:", score_dt)
print("Accuracy score of decission tree after balancing : ", score_dt_)

print("Accuracy score of logistic regresion :", score_lr)
print("Accuracy score of logistic regresion after balancing:", score_lr_)

print("Accuracy score of SVM:", score_svm)
print("Accuracy score of SVM after balancing:", score_svm_)


plt.figure(figsize=(8,8))
plt.bar(1,score_dt,label ='Decision Tree before balancing', width = 0.15)
plt.bar(1.1,score_dt_,label ='Decision Tree after balancing', width = 0.15)
plt.bar(1.4,score_lr,label ='logistic regresion before balancing', width = 0.15)
plt.bar(1.5,score_lr_,label ='logistic regresion after balancing', width = 0.15)
plt.bar(1.8,score_svm,label ='SVM before balancing', width = 0.1)
plt.bar(1.9,score_svm_,label ='SVM after balancing', width = 0.1)
plt.xticks([1.0,4.0])
plt.yticks([0.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
plt.xlabel('Models')
plt.ylabel('Acuracy')
plt.title('comparison')
plt.legend(loc='lower right')
plt.show()









