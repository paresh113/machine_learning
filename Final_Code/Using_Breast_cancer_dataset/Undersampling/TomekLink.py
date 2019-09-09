import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import model_selection
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

#import dataset
dataset = pd.read_csv("E:\kushboo_apu\imbalance_dataset.csv")
X= dataset.iloc[:,0:10].values
y= dataset.iloc[:,10:11].values
dataset['Class (2 for benign, 4 for malignant)'].value_counts()

#print("Dataset shape = ",X.shape,y.shape)
#dataset.head(20)
#calculating missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN' , strategy = 'mean' , axis =0)
imputer = imputer.fit(X[:, 0:10])
X[:,0:10] = imputer.transform(X[:,0:10])

#spliting dataset into testing and training dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=1/5,random_state = 0 )

#applying decission tree 
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


#logistic regrasion implementation
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

#precision and recall and support for SVM

from sklearn.metrics import classification_report
print('precision and recall for gnb: \n',classification_report(y_test,y_pred))

#find out accuracy  for SVM

from sklearn.metrics import accuracy_score
score_svm = accuracy_score(y_test, y_pred.round())
print("Accuracy score of gnb:", score_svm)

#applying SVM
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
#print('Decision Tree before balancing =',dt_before_)
sm2 = sum(scores_lr[0:len(scores_lr)])
lr_before_= sm2/k
#print('LogisticRegression before balancing =',lr_before_)
sm3 = sum(scores_svm[0:len(scores_svm)])
svm_before_ = sm3/k
#print('SVM before balancing=',svm_before_)
s4 = sum(scores_gnb[0:len(scores_gnb)])
gnb_before_ = s4/k
#print('GaussianNB before balancing =',gnb_before_)



########
#### applying NearMiss for balancing 

from collections import Counter 
from sklearn.datasets import fetch_mldata 
from imblearn.under_sampling import CondensedNearestNeighbour


#### applying Tomek link in dataset

from collections import Counter

from imblearn.under_sampling import TomekLinks

#print('Original dataset shape %s' % Counter(y))

tl = TomekLinks()
X_res, y_res = tl.fit_resample(X, y)
print('Resampled dataset shape %s' % Counter(y_res))

X = X_res
y = y_res

#spliting dataset into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X_res, y_res, test_size=1/5,random_state = 0 )
print(X.shape, y.shape)


#applying decission tree after balancing
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

#logistic regrasion implementation
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


#applying SVM

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



plt.bar(1,score_dt,label ='Decision Tree before balancing', width = 0.15)
plt.bar(1.1,score_dt_,label ='Decision Tree after balancing', width = 0.15)
plt.bar(1.4,score_lr,label ='logistic regresion before balancing', width = 0.15)
plt.bar(1.5,score_lr_,label ='logistic regresion after balancing', width = 0.15)
plt.bar(1.8,score_svm,label ='SVM before balancing', width = 0.1)
plt.bar(1.9,score_svm_,label ='SVM after balancing', width = 0.1)
plt.bar(1.31,0)
plt.xticks([1.0,4.0])
plt.yticks([0.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
plt.xlabel('Models')
plt.ylabel('Acuracy')
plt.title('comparison')
plt.legend(loc='lower right')
plt.show()

