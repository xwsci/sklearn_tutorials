import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSCanonical, PLSRegression,CCA

#Datasets
l1=np.random.normal(size=50)
l2=np.random.normal(size=50)
latents=np.array([l1,l1,l2,l2]).T
x=latents+np.random.normal(size=200).reshape((50,4))
y=latents+np.random.normal(size=200).reshape((50,4))
x_train=x[0:25];x_test=x[25:]
y_train=y[0:25];y_test=y[25:]
#Canonical (symmetric) PLS
#Transform data
plsca=PLSCanonical(n_components=2)
plsca.fit(x_train,y_train)
x_train_r,y_train_r=plsca.transform(x_train,y_train)
x_test_r,y_test_r=plsca.transform(x_test,y_test)
#Scatter plot of scores
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.scatter(x_train_r[:,0],y_train_r[:,0],label='Train',marker='o',c='b',s=25)
plt.scatter(x_test_r[:,0],y_test_r[:,0],label='Test',marker='o',c='r',s=25)
plt.xlabel('x scores');plt.ylabel('y scores')
plt.title('Comp. 1: x vs y (test corr = %.2f)'%np.corrcoef(x_test_r[:,0],y_test_r[:,0])[0,1])
plt.legend()

plt.subplot(122)
plt.scatter(x_train_r[:,0],x_train_r[:,1],label='Train',marker='o',c='b',s=25)
plt.scatter(x_test_r[:,0],x_test_r[:,1],label='Test',marker='o',c='r',s=25)
plt.xlabel('x comp. 1');plt.ylabel('x comp. 2')
plt.title('Comp. 1 x vs Comp. 2 x (test corr = %.2f)'%np.corrcoef(x_test_r[:,0],x_test_r[:,1])[0,1])
plt.legend()


plt.show()
