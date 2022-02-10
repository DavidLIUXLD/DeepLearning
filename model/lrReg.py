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
    theta = np.linalg.inv(x_new_t.dot(x_new)).dot(x_new_t.dot(y.ravel()))
    #print(theta.shape)
    return theta
acc_scores = []
data_multi = np.loadtxt('Data\\tictac_multi.txt')
X = data_multi[:,:9]
Y = data_multi[:,9:]
kFold = KFold(n_splits = 10, shuffle=True)
outputs = []
for train_index, test_index in kFold.split(X) :
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    params = []
    y_predicts = []
    for i in range(0,9):
        param = normalEq(x_train, y_train[:,i]).ravel()#643*9
        params.append(param)
        b = param[:-1]
        w = param[-1]
        y_predict = x_test.dot(w) + b#YT
        y_predicts.append(y_predict)#YTC
    y_predicts = np.array(y_predicts)
    for i in range(0, len(y_predicts[0])) :
        arr = y_predicts[:,i]
        mi = np.where(arr == np.amax(arr))
        y_predict[i, mi] = 1
    y_predict = np.argmax(np.floor(y_predict), axis = 1)
    acc = accuracy_score(np.argmax(y_test, axis=1), y_predict)
    acc_scores.append(acc)
print(acc_scores)
print(np.mean(acc_scores))