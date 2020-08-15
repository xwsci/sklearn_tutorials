import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

#Generate data
x_train=15*np.random.rand(100,1)
noise=1*(0.5-np.random.rand(x_train.shape[0]))
y_train=np.sin(x_train).ravel()+noise
x_test=np.arange(-1,16,0.1).reshape(-1,1)
#Predict by Tree vs Bagging
y_tree_pred=DecisionTreeRegressor().fit(x_train,y_train).predict(x_test)
y_bagging_pred=BaggingRegressor().fit(x_train,y_train).predict(x_test)

plt.scatter(x_train,y_train,s=30,c='b',label='real data')
plt.plot(x_test,y_tree_pred,c='cyan',label='Tree, valence=%f'%np.var(y_tree_pred))
plt.plot(x_test,y_bagging_pred,label='Bagging, valence=%f'%np.var(y_bagging_pred))
plt.legend()
plt.show()




