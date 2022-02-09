import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
data_multi = np.loadtxt('Data\\tictac_multi.txt')
#print(data_multi)
#print(np.shape(data_multi))
#print('\n')
names = ['k-NN Weighted']
max_a2 = 0
kk = 0
acc_fold = []
knnReg = KNeighborsRegressor(algorithm='kd_tree',n_neighbors=3, weights='distance')
kFold = KFold(n_splits=10, shuffle=True)
k_vals = np.arange(2,40,1)
for k in k_vals :
        X = data_multi[:,:9]
        Y = data_multi[:,9:]
        X = StandardScaler().fit_transform(X)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.3)
        regs = [KNeighborsRegressor(k, weights = 'distance')]
        f = 1
        a2_val = 0
        for train_index, validation_index in kFold.split(X_train):
            x_train2, x_val = X_train[train_index], X_train[validation_index]
            y_train2, y_val = Y_train[train_index,:], Y_train[validation_index]

            for name, reg in zip(names,regs):
#                 print(f'Name {name}')
                reg.fit(x_train2, y_train2)
                y_val_predict = np.round(reg.predict(x_val))
                val_score = accuracy_score(y_val,y_val_predict)
#                 print(f'Validation Set Score: {val_score}')
                a2_val += val_score
            acc_fold += [a2_val/10]
            if a2_val/10 > max_a2:
                kk = k
                max_a2 = a2_val/10
            
            f += 1
print(f'The  k value chosen is {kk}, with an average score of {max_a2*100}%')
final_reg = KNeighborsRegressor(kk, weights = 'distance')
final_reg.fit(X_train, Y_train)
Y_pred = np.round(final_reg.predict(X_test))
score = accuracy_score(Y_pred, Y_test)
print(f'The score for the test set is {score*100}%') 
