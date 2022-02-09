#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn import metrics


# In[2]:


datasets = [np.loadtxt("../Data/tictac_single.txt"),             np.loadtxt("../Data/tictac_final.txt")]


# In[15]:


kf = KFold(n_splits=10)
k_vals = np.arange(2,40,1)

# names = ['k-NN Uniform,', 'k-NN Weighted']
names = ['k-NN Weighted']

max_a2 = 0
kk = 0
acc_fold = []
conf_matrix = []

for ds_cnt, ds in enumerate(datasets):    
    for k in k_vals:
        #     classifiers = [KNeighborsClassifier(k, weights = 'uniform'), \
#                    KNeighborsClassifier(k, weights = 'distance')]
        classifiers =[KNeighborsClassifier(k, weights = 'distance')]

        X = ds[:, :9]
        y = ds[:, 9:]
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

        f = 1
        a2_val = 0
        for train_index, validation_index in kf.split(X_train):
#             print(f'Fold {f}')
            X_train2, X_validation = X_train[train_index,:], X_train[validation_index,:]
            y_train2, y_validation = y_train[train_index,:], y_train[validation_index,:]

            for name, clf in zip(names,classifiers):
#                 print(f'Name {name}')
                clf.fit(X_train2, y_train2.ravel())
                y_val_predict = clf.predict(X_validation)
                val_score = clf.score(X_validation, y_validation)
#                 print(f'Validation Set Score: {val_score}')
                a2_val += val_score
            acc_fold += [a2_val/10]
            
            if a2_val/10 > max_a2:
                kk = k
                max_a2 = a2_val/10
            
            f += 1
    
    print(f'The  k value chosen is {kk}, with an average score of {max_a2*100}%')
    final_classifier = KNeighborsClassifier(kk, weights = 'distance')
    final_classifier.fit(X_train, y_train.ravel())
    y_pred = final_classifier.predict(X_test)
    score = final_classifier.score(X_test, y_test)
    print(f'The score for the test set is {score*100}%')
    conf_matrix += [confusion_matrix(y_test, y_pred)]


# In[17]:


print('Confusion matrix for the "Single" dataset:')
print(conf_matrix[0])
print('\nConfusion matrix for the "Final" dataset:')
print(conf_matrix[1])


# In[ ]:




