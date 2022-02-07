import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

data_multi = np.loadtxt('p1\\Data\\tictac_multi.txt')
#print(data_multi)
#print(np.shape(data_multi))
#print('\n')
X = data_multi[:,:9]
Y = data_multi[:,9:]
knnReg = KNeighborsRegressor()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
knnReg.fit(X_train, Y_train)
cross_val_SC = cross_val_score(knnReg, X, Y, cv = 10)
print(cross_val_SC)
print(np.mean(cross_val_SC))
