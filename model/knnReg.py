import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

data_multi = np.loadtxt('Data\\tictac_multi.txt')
#print(data_multi)
#print(np.shape(data_multi))
#print('\n')
X = data_multi[:,:9]
Y = data_multi[:,9:]
knnReg = KNeighborsRegressor()
KFold = KFold(n_splits=10, shuffle=True)
cross_val_SC = cross_val_score(knnReg, X, Y, cv = KFold)
print(cross_val_SC)
print(np.mean(cross_val_SC))
