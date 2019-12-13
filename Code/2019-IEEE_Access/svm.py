
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn import metrics
from sklearn import datasets
#linux
#path = r"/home/mirsahib/Desktop/Project-Andromeda/Dataset/1st_Level_Feature_Extracted/mega.csv"

path = r"C:\Users\Mir Sahib\Desktop\Project-Andromeda\Dataset\2nd_Level_Feature_Extracted\combined_csv.csv"

df = pd.read_csv(path)

col = df.shape[1]

X = df.values[:,0:col-1]
Y = df.values[:,-1]
scaler = StandardScaler()
#X = scaler.fit_transform(X)
fold = 10
"""## SVM Linear with penalty parameter = 1"""

kfold = KFold(fold, True, 1)
acc_score = []
count=1
for train_index, test_index in kfold.split(df):
  X_train, X_test, y_train, y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
  #SVM
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.fit_transform(X_test)
  svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 
  svm_predictions = svm_model_linear.predict(X_test) 
  accuracy = svm_model_linear.score(X_test, y_test)
  acc_score.append(accuracy*100)
  print('Accuracy in fold '+str(count)+' is '+str(accuracy*100))
  count = count+1

print("Average Accuracy: "+str(sum(acc_score)/10))

kfold = KFold(fold, True, 1)
acc_score = []
count=1
for train_index, test_index in kfold.split(df):
  X_train, X_test, y_train, y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
  #SVM
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.fit_transform(X_test)
  svm_model_poly = SVC(kernel = 'poly',gamma='auto').fit(X_train, y_train) 
  svm_predictions = svm_model_poly.predict(X_test) 
  accuracy = svm_model_linear.score(X_test, y_test)
  acc_score.append(accuracy*100)
  print('Accuracy in fold '+str(count)+' is '+str(accuracy*100))
  count = count+1

print("Average Accuracy: "+str(sum(acc_score)/10))

"""# SVM RBF"""

kfold = KFold(fold, True, 1)
acc_score = []
count=1
for train_index, test_index in kfold.split(df):
  X_train, X_test, y_train, y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
  #SVM
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.fit_transform(X_test)
  svm_model_rbf = SVC(kernel = 'rbf',gamma='auto').fit(X_train, y_train) 
  svm_predictions = svm_model_rbf.predict(X_test) 
  accuracy = svm_model_linear.score(X_test, y_test)
  acc_score.append(accuracy*100)
  print('Accuracy in fold '+str(count)+' is '+str(accuracy*100))
  count = count+1

print("Average Accuracy: "+str(sum(acc_score)/10))

