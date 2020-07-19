import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_blobs

#Create database, x.shape=(50,2),y.shape=(50,)
x,y=make_blobs(n_samples=50,centers=2,random_state=0,cluster_std=0.60)
#Fit the model using SGD
clf=SGDClassifier(loss='hinge',alpha=0.1,max_iter=500)
clf.fit(x,y)
#Plot
xx,yy=np.meshgrid(np.arange(x[:,0].min()-1, x[:,0].max()+1, 0.02),
                  np.arange(x[:,1].min()-1, x[:,0].max()+3, 0.02))
z=clf.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
plt.contourf(xx,yy,z,cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(x[:,0],x[:,1],c=y,cmap=plt.cm.coolwarm,s=20,edgecolors='k')
plt.title("SVC with customized kernel, score=%f"%clf.score(x,y))
#plt.xlim(4,8);plt.ylim(1,5)
plt.xlabel('x1');plt.ylabel('x2')#;plt.axis('tight')

plt.show()

