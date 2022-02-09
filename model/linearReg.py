import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

def normalEq(x, y):
    b = np.ones((x.shape[0], 1))
    #print(b.shape)
    x_new = []
    for i in range(0,x.shape[1]):
        temp = np.reshape(x[:,i], (x.shape[0], 1))
        #print(temp.shape)
        if i == 0 :
            x_new = np.append(b, temp, axis = 1)
        else:
            x_new = np.append(x_new, temp, axis = 1)
    #print(x_new.shape)
    x_new_t = np.transpose(x_new)
    theta = np.linalg.inv(x_new_t.dot(x_new)).dot(x_new_t.dot(y))
    #print(theta.shape)
    return theta
acc_scores = []
data_multi = np.loadtxt('p1\\Data\\tictac_multi.txt')
X = data_multi[:,:9]
Y = data_multi[:,9:]
kFold = KFold(n_splits = 10, shuffle=True)
outputs = []
for train_index, test_index in kFold.split(X) :
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    param = normalEq(x_train, y_train)
    b = param[0,:]
    w = param[1:,:]
    y_predict = x_test.dot(w) + b
    #print(y_predict.shape, len(y_predict))
    for i in range(0, len(y_predict)) :
        arr = y_predict[i,:]
        mi = np.where(arr == np.amax(arr))
        y_predict[i, mi] = 1
    y_predict = np.round(y_predict)
    acc = accuracy_score(y_test, y_predict)
    #print(acc)
    acc_scores.append(acc)
print(acc_scores)
print(np.mean(acc_scores))

