#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 09:42:08 2023

@author: giftypokuaa
"""
#%%======== importing Packages========
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from keras import models, layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import metrics, linear_model
from sklearn.metrics import confusion_matrix,classification_report,ConfusionMatrixDisplay 
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC 
#from tkinter import filedialog
import tkinter as tk
import matplotlib.pyplot as plt
root = tk.Tk()
root.withdraw()
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import time
#%%================ Set Up the Directory, Read the Data ==================

root.attributes("-topmost", True)
npath=r"/Users/giftypokuaa/Desktop/PROJECT"
dataset=pd.read_csv(npath+"/Data/heart.csv")

dataset['sex','cp','fbs','restecg','exang'] = dataset['sex','cp','fbs','restecg','exang'].astype(object)

#%% ========Shape of dataset ====
print(dataset.shape)

#The shape of the data shows that, there are 297 rows and 14 features.

#%%=====Correlation Matrix for only the continous variables=====


#Numeric_Data=dataset.select_dtypes(exclude=["sex","cp","fbs","restecg","exang"])
Numeric_Data=dataset.select_dtypes(exclude=['object'])

Numeric_Data.corr()
#%%
## putting the outcome variable(condition) on the first column
cols=list(dataset)
cols.insert(0,cols.pop(cols.index('condition')))
dataset= dataset.loc[:,cols]


X = dataset.drop('condition', axis=1)  # removing outcome variable "condition from data

X=pd.get_dummies(X, columns=["sex","cp","fbs","restecg","exang"]) # dummy for categorical covariate variables

y = dataset.iloc[:,0]# last column 13 will the output variable 

Descriptive=dataset.describe() #descriptive statistics on data
print(Descriptive)
#Descriptive.to_csv(main_dir+ '/Results/Descriptive_Statistics.csv') # saving descriptive statistics as csv file



#%%=======Creating Enpty Lists============
Rf_accuracy = []
Lr_accuracy = []
Sv_accuracy=[]

#%% =================Logistic Regression=========
##===================== at 90-10=====================

for i in range(20): 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    logreg = LogisticRegression(random_state=0)
    logreg.fit(X_train, y_train)# fitting the model with  training data
    L_pred = logreg.predict(X_test)# making prediction

cml=confusion_matrix(y_test, L_pred) # confusion matrice


la=accuracy_score(y_test,L_pred)# accuaracy

la





#%%


#======RANDOM FOREST Figure===========
clf=DecisionTreeClassifier(max_depth=4)# spicifying the classifier

model=clf.fit(X,y)# fitting the model with the entire dataset to get tree presentation

#%%
#===Figure presentation of the tree
fig=plt.figure(figsize=(40,35))
#_=tree.plot_tree(clf,feature_names=X,
#                class_names=y,
#                 filled=(True))
tree.plot_tree(model,
               filled=(True))
fig.savefig(npath + "/Results/decision4_tree.png")

#%%  RANDOM FOREST MODEL===
#=========70:20 split============


#100 iteration for Random forest and Multiple linear regression models
for i in range(20): 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    clf= RandomForestClassifier(criterion="gini",n_estimators=200, random_state=0)  
    clf.fit(X_train, y_train)  
    RF_pred = clf.predict(X_test)  
    
clf.feature_importances_ # to check for important features

cm3=confusion_matrix(y_test, RF_pred) # confusion matrice


Ra=accuracy_score(y_test,RF_pred)# accuaracy

Ra     
   
    

#%%
# 90- 10 split

#100 iteration for Random forest and Multiple linear regression models
for i in range(20): 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,random_state=0)
    
    clf= RandomForestClassifier(criterion="gini",random_state=0)  
    clf.fit(X_train, y_train)  
    RF_pred = clf.predict(X_test)  
    


clf.feature_importances_ # to check for important features

cm1=confusion_matrix(y_test, RF_pred) # confusion matrice
cm1

Ra1=accuracy_score(y_test,RF_pred)# accuaracy

Ra1

#%%====SVM==========

for i in range(20): 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,random_state=0)
    
    svclassifier = SVC(kernel='linear') # specifying linear function  for the model
    svclassifier.fit(X_train, y_train)  # fitting SVC model with traing data 
s_pred = svclassifier.predict(X_test)  # predicting with test data

cms=confusion_matrix(y_test,s_pred)  # confusion matrix output
print(cms)
print(classification_report(y_test,s_pred))  


sa=accuracy_score(y_test,s_pred)# accuaracy
sa


#%%======= Creating side by side Boxplot for the 3 models's accuracy=========
Rf_accuracy.append(Ra1)

Lr_accuracy.append(la)

Sv_accuracy.append(sa)

#Dataframe of Random forest and regression errors
Result_accuracy=pd.DataFrame({'Random_Forest':Rf_accuracy,"Logistic_Regression":Lr_accuracy,'SVM':Sv_accuracy}) 
 # Saving the datafram with errors as a csv file   

#Plotting a side-by-side boxplot of the errors from the two models
fig, ax = plt.subplots()

ax.boxplot(Result_accuracy)
# labeling the block
ax.set_title('Side by Side Boxplot of Accuracy for different Models')
ax.set_xlabel('Predictive Models')
ax.set_ylabel('Model Accuracy')
xticklabels=['Random Forest','Logistic Regression','SVM']

ax.set_xticklabels(xticklabels)
# 
ax.yaxis.grid(True)
# 
plt.savefig(npath+'/Results/Side_by_Side.png')

#Description of error dataframe
Res=Result_accuracy.describe()
Res.to_csv(npath+'/Results/Accuracy_description.csv')




#%%=========== Neutral Network which is dropped from analysis due to very low accuracy===

#%%===== Main Loop To try all Optimizers and all loss functions =======================================
NN_Accuracies = []
NN_Times=[]
NN_Epochs=[]
NN_Batch_Size=[]
  #================ Setting up the Kerasn Model =====================================================
model = Sequential() # is a type of sequencial NN (simple )
model.add(Dense(80, input_dim=13, activation='relu')) # add a layer that is 12 nodes; 
model.add(Dense(40, activation='relu')) # input =8 means we have 8 x variables
model.add(Dense(20, activation='relu')) 
model.add(Dense(10, activation='relu')) 
model.add(Dense(1, activation='sigmoid')) # output function, sigmoid convert output into 1 or 0 
  # compile the keras model
model.compile(loss='squared_hinge', optimizer='adam', metrics=['accuracy']) 


for epochs in range (0,150,20):
 for batch_size in range (6,15,2):
  start_time = time.time()
  model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
  total_time = time.time() - start_time
  #=============== Predict New Values on the test set ================================================
  #NN_pred = model.predict_classes(X_test)
  predict_x=model.predict(X_test) 
  classes_x=np.argmax(predict_x,axis=1)
  #========
  NN_Accuracies.append(accuracy_score(y_test,classes_x))
  NN_Times.append(total_time)
  NN_Epochs.append(epochs)
  NN_Batch_Size.append(batch_size)
  #=============== Fit the model on the dataset ======================================================
start_time = time.time()
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
total_time = time.time() - start_time
  #=============== Predict New Values on the test set ================================================
  #NN_pred = model.predict(X_test)
predict_x=model.predict(X_test) 
classes_x=np.argmax(predict_x,axis=1)
  #========

#NN_Optimiser.append(optim)
#NN_Loses.append(lo)
#%%======== Convert to dataframe and save results =====================================================
cols=np.asarray([NN_Accuracies,NN_Times,NN_Epochs,NN_Batch_Size])
cols=cols.transpose()
df1=pd.DataFrame(cols,columns=["NN_Accuracies","NN_Times","NN_Epochs","NN_Batch_Size"])
df1=df1.sort_values(["NN_Accuracies","NN_Times"],ascending = (False, True) )
df1.to_csv (npath+'//Results//Result_Table.csv', index = False, header=True)

f=confusion_matrix(y_test,classes_x)  # confusion matrix output
print(f)
#print(classification_report(y_test,classes_x))  





