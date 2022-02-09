#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# In[2]:


datasets = [np.loadtxt("../Data/tictac_single.txt"),             np.loadtxt("../Data/tictac_final.txt")]


# In[3]:


kf = KFold(n_splits=10)

net_hidden_layers = (75, 50, 25)
net_learning_rate = 0.01
epochs = 250
datasets_names = ['Single', 'Final']


for ds_cnt, ds in enumerate(datasets):
    X = ds[:, :9]
    y = ds[:, 9:]
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
    
    net = MLPClassifier(activation='tanh',
                    n_iter_no_change=1000)
    
    net.set_params(hidden_layer_sizes=net_hidden_layers, 
                       learning_rate_init=net_learning_rate,
                       max_iter=epochs)
    f = 1
    avg_score = 0
    print(f'Dataset: {datasets_names[ds_cnt]}')
    for train_index, validation_index in kf.split(X_train):
        X_train2, X_validation = X_train[train_index,:], X_train[validation_index,:]
        y_train2, y_validation = y_train[train_index,:], y_train[validation_index,:]
        
        
        net.fit(X_train2, y_train2.ravel())
        y_train_pred = net.predict(X_train2)
        y_val_pred = net.predict(X_validation)
        acc_score = accuracy_score(y_train2, y_train_pred)
        test_score = accuracy_score(y_validation, y_val_pred)
        
        print('--------------------------------------')
        print(f'Fold {f}')
        print(f'Hidden Layer Architecture: {net_hidden_layers}')
        print(f'Learning Rate: {net_learning_rate}')
        print(f'Number of Epochs: {epochs}')
        print(f'Train Accuracy = {np.round(acc_score*100,2)}%')
        print(f'Test Accuracy = {np.round(test_score*100,2)}%')
        print('--------------------------------------')
        
        avg_score += test_score
        f += 1
        
    avg_score /= 10
    print(f'Average score for all the folds is {avg_score}')
    
    final_net = MLPClassifier(activation='tanh',
                              n_iter_no_change=1000)
    
    final_net.set_params(hidden_layer_sizes=net_hidden_layers, 
                         learning_rate_init=net_learning_rate,
                         max_iter=epochs)
    
    final_net.fit(X_train, y_train.ravel())
    final_pred = final_net.predict(X_test)
    final_acc = accuracy_score(y_test, final_pred)
    print(f'Final Accuracy score for the MLP: {np.round(final_acc*100, 2)}')
    


# In[ ]:




