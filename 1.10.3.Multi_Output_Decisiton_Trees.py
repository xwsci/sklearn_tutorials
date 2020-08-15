import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor,plot_tree

#Generate datasets x.shape=(100,1) y.shape=(100,2)
rng=np.random.RandomState(1)
x=np.sort(200*rng.rand(100,1)-100)
y=np.array([np.pi*np.sin(x).ravel(),np.pi*np.cos(x).ravel()]).T
y[::5,:]+=(0.5-rng.rand(20,2))
#Fit the model using DecisionTreeClassifier
regr_1=DecisionTreeRegressor(max_depth=1)
regr_2=DecisionTreeRegressor(max_depth=5)
regr_3=DecisionTreeRegressor(max_depth=10)
regr_1.fit(x,y)
regr_2.fit(x,y)
regr_3.fit(x,y)
#Predict
x_test=np.arange(-100,100,0.01).reshape(-1,1)
y_1=regr_1.predict(x_test)
y_2=regr_2.predict(x_test)
y_3=regr_3.predict(x_test)
#Plot
plt.figure(figsize=(15,10))
plt.subplot(121)
plt.scatter(y[:,0],y[:,1],c='navy',s=20,edgecolors='k',label='data')
plt.plot(y_1[:,0],y_1[:,1],c='cornflowerblue',label='Max_depth=1')
plt.plot(y_2[:,0],y_2[:,1],c='red',label='Max_depth=5')
plt.plot(y_3[:,0],y_3[:,1],c='orange',label='Max_depth=10')
plt.xlabel('Target 1');plt.ylabel('Target 2');plt.axis('tight')
plt.legend()
#Plot tree
plt.subplot(122)
plot_tree(regr_2,filled=True)
plt.show()

