import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix




path = r"C:\Users\Mir Sahib\Desktop/Project-Andromeda/Dataset/demo/combined_csv.csv"

df = pd.read_csv(path)
col = df.shape[1]

X = df.values[:,0:col-1]
Y = df.values[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20)
model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)

y_predict = model.predict(X_test)

print(accuracy_score(y_test, y_predict))

print(confusion_matrix(y_test, y_predict))
    
