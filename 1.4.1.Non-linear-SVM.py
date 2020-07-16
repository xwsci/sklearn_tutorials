import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

xx,yy=np.meshgrid(np.linspace(-3,3,500),np.linspace(-3,3,500))
np.random.seed(0)
x=np.random.randn(300,2)
y=np.logical_xor(x[:,0]>0,x[:,1]>0)

#Fit the model
clf1=svm.NuSVC(gamma='auto')
clf1.fit(x,y)
clf2=svm.SVC(kernel='sigmoid',C=1)
clf2.fit(x,y)
clf3=svm.LinearSVC()
clf3.fit(x,y)

#Plot the non-linear decision function for each datapoint on the grid
z1=clf1.decision_function(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
fig=plt.figure(figsize=(15,5))
plt.subplot(131)
plt.imshow(z1,interpolation='nearest',extent=(xx.min(),xx.max(),yy.min(),yy.max()),aspect='auto',origin='lower',cmap=plt.cm.PuOr_r)
contours=plt.contour(xx,yy,z1,levels=[0],linewidths=2,linestyles='dashed')
plt.scatter(x[:,0],x[:,1],s=30,c=y, cmap=plt.cm.Paired,edgecolors='k')
plt.xticks(())
plt.yticks(())
plt.axis([-3,3,-3,3])
plt.title("Non-linear SVC")

#Plot the non-linear decision function for each datapoint on the grid
z2=clf2.decision_function(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
plt.subplot(132)
plt.imshow(z2,interpolation='nearest',extent=(xx.min(),xx.max(),yy.min(),yy.max()),aspect='auto',origin='lower',cmap=plt.cm.PuOr_r)
contours=plt.contour(xx,yy,z2,levels=[0],linewidths=2,linestyles='dashed')
plt.scatter(x[:,0],x[:,1],s=30,c=y, cmap=plt.cm.Paired,edgecolors='k')
plt.xticks(())
plt.yticks(())
plt.axis([-3,3,-3,3])
plt.title("SVC,kernel=sigmoid")

#Plot the non-linear decision function for each datapoint on the grid
z3=clf3.decision_function(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
plt.subplot(133)
plt.imshow(z3,interpolation='nearest',extent=(xx.min(),xx.max(),yy.min(),yy.max()),aspect='auto',origin='lower',cmap=plt.cm.PuOr_r)
contours=plt.contour(xx,yy,z3,levels=[0],linewidths=2,linestyles='dashed')
plt.scatter(x[:,0],x[:,1],s=30,c=y, cmap=plt.cm.Paired,edgecolors='k')
plt.xticks(())
plt.yticks(())
plt.axis([-3,3,-3,3])
plt.title("Linear SVC")

plt.show()



