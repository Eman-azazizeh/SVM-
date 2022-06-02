#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
from builtins import range


# In[2]:


"""
SECTION 1 : Load and setup data for training

"""
import pandas as pd
import numpy as np


# In[3]:


# Data sets
IRIS_TRAINING = "train.txt"
IRIS_TEST = "test.txt"
train_data = np.genfromtxt(IRIS_TRAINING, skip_header=1, 
    dtype=float, delimiter=';')
test_data = np.genfromtxt(IRIS_TEST, skip_header=1, 
    dtype=float, delimiter=';')


# In[4]:


#split x and y (feature and target)
xtrain = train_data[:,:4000]
ytrain = train_data[:,4001]
print(ytrain)


# In[5]:


import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1457)


# In[6]:


# import SVC classifier
from sklearn.svm import SVC
# import metrics to compute accuracy
from sklearn.metrics import accuracy_score


# # Run SVM with default hyperparameters
# 

# In[7]:


# instantiate classifier with default hyperparameters
svc=SVC() 


# In[8]:


# fit classifier to training set
svc.fit(xtrain,ytrain)


# In[9]:


#split x and y (feature and target)
xtest = test_data[:,:4000]
ytest = test_data[:,4001]


# In[10]:


# make predictions on test set
y_pred=svc.predict(xtest)


# In[11]:


# compute and print accuracy score
print('Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(ytest, y_pred)))


# # Run SVM with rbf kernel and C=100.0

# In[14]:


# instantiate classifier with rbf kernel and C=100
svc=SVC(C=100.0)
# fit classifier to training set
svc.fit(xtrain,ytrain)
# make predictions on test set
y_pred=svc.predict(xtest)
# compute and print accuracy score
print('Model accuracy score with rbf kernel and C=100.0 : {0:0.4f}'. format(accuracy_score(ytest, y_pred)))


# # Run SVM with rbf kernel and C=1000.0
# 
# 

# In[36]:


# instantiate classifier with rbf kernel and C=1000
svc=SVC(C=1000.0) 


# fit classifier to training set
svc.fit(xtrain,ytrain)


# make predictions on test set
y_pred=svc.predict(xtest)
# compute and print accuracy score
print('Model accuracy score with rbf kernel and C=1000.0 : {0:0.4f}'. format(accuracy_score(ytest, y_pred)))


# #  Run SVM with linear kernel

# # Run SVM with linear kernel and C=1.0

# In[38]:


# instantiate classifier with linear kernel and C=1.0
linear_svc=SVC(kernel='linear', C=1.0) 


# fit classifier to training set
linear_svc.fit(xtrain,ytrain)


# make predictions on test set
y_pred_test=linear_svc.predict(xtest)


# compute and print accuracy score
print('Model accuracy score with linear kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(ytest, y_pred_test)))


# # Run SVM with linear kernel and C=100.0
# 
# 

# In[37]:


# instantiate classifier with linear kernel and C=100.0
linear_svc100=SVC(kernel='linear', C=100.0) 


# fit classifier to training set
linear_svc100.fit(xtrain, ytrain)


# make predictions on test set
y_pred=linear_svc100.predict(xtest)


# compute and print accuracy score
print('Model accuracy score with linear kernel and C=100.0 : {0:0.4f}'. format(accuracy_score(ytest, y_pred)))


# # Run SVM with linear kernel and C=1000.0
# 
# 

# In[18]:


# instantiate classifier with linear kernel and C=1000.0
linear_svc1000=SVC(kernel='linear', C=1000.0) 


# fit classifier to training set
linear_svc1000.fit(xtrain, ytrain)


# make predictions on test set
y_pred=linear_svc1000.predict(xtest)


# compute and print accuracy score
print('Model accuracy score with linear kernel and C=1000.0 : {0:0.4f}'. format(accuracy_score(ytest, y_pred)))


# # Compare the train-set and test-set accuracy

# In[24]:


from sklearn.svm import LinearSVC
#Compare the train-set and test-set accuracy
y_pred_train = linear_svc1000.predict(xtrain)

y_pred_train


# In[25]:


print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(ytrain, y_pred_train)))


# In[27]:


print('Training set score: {:.4f}'.format(linear_svc1000.score(xtrain, ytrain)))

print('Test set score: {:.4f}'.format(linear_svc1000.score(xtest, ytest)))


# # # instantiate classifier with polynomial kernel and C=1.0
# 

# In[29]:


#14. Run SVM with polynomial kernel
#Run SVM with polynomial kernel and C=1.0
# instantiate classifier with polynomial kernel and C=1.0
poly_svc=SVC(kernel='poly', C=1.0) 


# fit classifier to training set
poly_svc.fit(xtrain,ytrain)


# make predictions on test set
y_pred=poly_svc.predict(xtest)


# compute and print accuracy score
print('Model accuracy score with polynomial kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(ytest, y_pred)))


# # Run SVM with polynomial kernel and C=100.0
# 
# 

# In[43]:


# instantiate classifier with polynomial kernel and C=100.0
poly_svc100=SVC(kernel='poly', C=100.0) 


# fit classifier to training set
poly_svc100.fit(xtrain, ytrain)


# make predictions on test set
y_pred=poly_svc100.predict(xtest)


# compute and print accuracy score
print('Model accuracy score with polynomial kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(ytest, y_pred)))


# # Run SVM with sigmoid kernel

# # Run SVM with sigmoid kernel and C=1.0

# In[32]:


# instantiate classifier with sigmoid kernel and C=1.0
sigmoid_svc=SVC(kernel='sigmoid', C=1.0) 


# fit classifier to training set
sigmoid_svc.fit(xtrain,ytrain)


# make predictions on test set
y_pred=sigmoid_svc.predict(xtest)


# compute and print accuracy score
print('Model accuracy score with sigmoid kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(ytest, y_pred)))


# # Run SVM with sigmoid kernel and C=100.0

# In[34]:


# instantiate classifier with sigmoid kernel and C=100.0
sigmoid_svc100=SVC(kernel='sigmoid', C=100.0) 


# fit classifier to training set
sigmoid_svc100.fit(xtrain,ytrain)


# make predictions on test set
y_pred=sigmoid_svc100.predict(xtest)


# compute and print accuracy score
print('Model accuracy score with sigmoid kernel and C=100.0 : {0:0.4f}'. format(accuracy_score(ytest, y_pred)))


# # Run SVM with sigmoid kernel and C=1000.0

# In[44]:


# instantiate classifier with sigmoid kernel and C=100.0
sigmoid_svc1000=SVC(kernel='sigmoid', C=1000.0) 


# fit classifier to training set
sigmoid_svc1000.fit(xtrain,ytrain)


# make predictions on test set
y_pred=sigmoid_svc1000.predict(xtest)


# compute and print accuracy score
print('Model accuracy score with sigmoid kernel and C=1000.0 : {0:0.4f}'. format(accuracy_score(ytest, y_pred)))


# In[ ]:




