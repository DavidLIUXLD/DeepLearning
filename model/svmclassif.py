#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# In[2]:


datasets = [np.loadtxt("../Data/tictac_single.txt"),             np.loadtxt("../Data/tictac_final.txt")]


# In[3]:


kernels = ['sigmoid', 'rbf', 'poly', 'linear', 'sigmoid']
kf = KFold(n_splits=10)
Cs = np.arange(1,10,1)
dataset_names = ['Single', 'Final']


# In[4]:


for ds_cnt, ds in enumerate(datasets):
    X = ds[:, :9]
    y = ds[:, 9:]
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
    
    CC = 1
    kk = ""
    acc_fold = []
    conf_matrix = []
    max_a2 = 0
    
    print(f'Dataset: {dataset_names[ds_cnt]}')
    for kernel in kernels:
        for C in Cs:
            clf = SVC(kernel=kernel, C=C)
            f = 1
            a2_val = 0
            for train_index, validation_index in kf.split(X_train):
                X_train2, X_validation = X_train[train_index,:], X_train[validation_index,:]
                y_train2, y_validation = y_train[train_index,:], y_train[validation_index,:]
                
                clf.fit(X_train2, y_train2.ravel())
                val_score = clf.score(X_validation, y_validation)
                a2_val += val_score
                
                acc_fold += [a2_val/10]
                
                if a2_val/10 > max_a2:
                    kk = kernel
                    CC = C
                    max_a2 = a2_val/10
                
                f += 1
    print(f'The chosen kernel function is {kk}\n'
          f'The chosen C value is {CC}\n'
          f'The final score is {np.round(max_a2*100,2)}%')
        


# In[8]:


ds = datasets[0]
X = ds[:, :9]
y = ds[:, 9:]
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
clf = SVC(kernel='rbf', C=9)
clf.fit (X_train, y_train.ravel())
score = clf.score(X_test, y_test)
print('Dataset: Single')
print(f'The accuracy score is {np.round(score * 100, 2)}%')


# In[7]:


ds = datasets[1]
X = ds[:, :9]
y = ds[:, 9:]
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
clf = SVC(kernel='rbf', C=4)
clf.fit (X_train, y_train.ravel())
score = clf.score(X_test, y_test)
print('Dataset: Final')
print(f'The accuracy score is {np.round(score * 100, 2)}%')


# In[ ]:




