import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm,datasets

#Import iris dataset
iris=datasets.load_iris() #x=[attribute1, attribute2, attribute3, attribute4];y=[0,1,2]
x=iris.data[:,:2]
y=iris.target
#Create SVM models to fit the database
linearsvc=svm.LinearSVC(max_iter=10000)
svc_linear=svm.SVC(kernel='linear')
svc_rbf=svm.SVC(kernel='rbf',gamma='auto')
svc_poly=svm.SVC(kernel='poly',degree=3,gamma='auto')
linearsvc.fit(x,y)
svc_linear.fit(x,y)
svc_rbf.fit(x,y)
svc_poly.fit(x,y)
#Make predictions and plot
#Linear SVC
plt.figure(figsize=(10,10))
plt.subplot(221)
xx,yy=np.meshgrid(np.arange(x[:,0].min()-1, x[:,0].max()+1, 0.02),np.arange(x[:,1].min()-1, x[:,0].max()+1, 0.02))
z=linearsvc.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
plt.contourf(xx,yy,z,cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(x[:,0],x[:,1],c=y,cmap=plt.cm.coolwarm,s=20,edgecolors='k')
plt.title("Linear SVC")
plt.xlim(4,8);plt.ylim(1,5)
plt.xlabel('x1');plt.ylabel('x2')#;plt.axis('tight')

#SVC with kernel = linear
plt.subplot(222)
xx,yy=np.meshgrid(np.arange(x[:,0].min()-1, x[:,0].max()+1, 0.02),np.arange(x[:,1].min()-1, x[:,0].max()+1, 0.02))
z=svc_linear.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
plt.contourf(xx,yy,z,cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(x[:,0],x[:,1],c=y,cmap=plt.cm.coolwarm,s=20,edgecolors='k')
plt.title("SVC with linear kernel")
plt.xlim(4,8);plt.ylim(1,5)
plt.xlabel('x1');plt.ylabel('x2')#;plt.axis('tight')

#SVC with kernel = rbf
plt.subplot(223)
xx,yy=np.meshgrid(np.arange(x[:,0].min()-1, x[:,0].max()+1, 0.02),np.arange(x[:,1].min()-1, x[:,0].max()+1, 0.02))
z=svc_rbf.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
plt.contourf(xx,yy,z,cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(x[:,0],x[:,1],c=y,cmap=plt.cm.coolwarm,s=20,edgecolors='k')
plt.title("SVC with RBF kernel")
plt.xlim(4,8);plt.ylim(1,5)
plt.xlabel('x1');plt.ylabel('x2')#;plt.axis('tight')

#SVC with kernel = poly
plt.subplot(224)
xx,yy=np.meshgrid(np.arange(x[:,0].min()-1, x[:,0].max()+1, 0.02),np.arange(x[:,1].min()-1, x[:,0].max()+1, 0.02))
z=svc_poly.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
plt.contourf(xx,yy,z,cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(x[:,0],x[:,1],c=y,cmap=plt.cm.coolwarm,s=20,edgecolors='k')
plt.title("SVC with polynomial (degree=3) kernel")
plt.xlim(4,8);plt.ylim(1,5)
plt.xlabel('x1');plt.ylabel('x2')#;plt.axis('tight')

plt.show()
