import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from joblib import dump

data_multi = np.loadtxt('Data\\tictac_multi.txt')
X = data_multi[:,:9]
Y = data_multi[:,9:]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y);
kFold = KFold(n_splits=10, shuffle=True)
mlpReg = MLPRegressor(activation='relu', hidden_layer_sizes=(100,80,80,80,80,80,50,), max_iter=100000, learning_rate_init=0.001)
acc_scores = []
for train_index, val_index in kFold.split(X_train):
    x_train, x_val = X_train[train_index], X_train[val_index]
    y_train, y_val = Y_train[train_index], Y_train[val_index]
    mlpReg.fit(x_train,y_train)
    y_val_predict = np.round(np.abs(mlpReg.predict(x_val)));
    accu = accuracy_score(np.argmax(y_val, axis=1),np.argmax(y_val_predict, axis=1))
    acc = mlpReg.score(x_val, y_val)
    acc_scores.append(acc)
print("mlp regressor accuracy: ", np.mean(acc_scores))
y_predict = mlpReg.predict(X_test);
fa = accuracy_score(np.argmax(Y_test, axis=1),np.argmax(y_predict, axis=1))
print("final accuracy: ", fa)

dump(mlpReg, 'mlpRegModel.joblib')


