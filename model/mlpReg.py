import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, cross_val_score
from joblib import dump

data_multi = np.loadtxt('p1\\Data\\tictac_multi.txt')
X = data_multi[:,:9]
Y = data_multi[:,9:]
kFold = KFold(n_splits=10, shuffle=True)
mlpReg = MLPRegressor(activation='relu', learning_rate='constant', max_iter = 400, hidden_layer_sizes=(300, ), alpha=0.001)
cross_val_SC = cross_val_score(mlpReg, X, Y, cv = kFold)
print(cross_val_SC)
print(np.mean(cross_val_SC))
dump(mlpReg, 'mlpRegModel.joblib')


