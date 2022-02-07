import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, cross_val_score

data_multi = np.loadtxt('p1\\Data\\tictac_multi.txt')
X = data_multi[:,:9]
Y = data_multi[:,9:]
kFold = KFold(n_splits=10, shuffle=True)
mlpReg = MLPRegressor()
cross_val_SC = cross_val_score(mlpReg, X, Y, cv = kFold)
print(cross_val_SC)
print(np.mean(cross_val_SC))


