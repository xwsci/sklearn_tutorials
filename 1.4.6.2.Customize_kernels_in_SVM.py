import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm,datasets

#Import iris dataset
iris=datasets.load_iris() #x=[attribute1, attribute2, attribute3, attribute4];y=[0,1,2]
x=iris.data[:,:2]
y=iris.target
#Define customized kernel function
def my_kernel(x,y):
    M=np.array([[2,0],[0,1]])
    return np.dot(np.dot(x,M),y.T)
#Create SVM model with customized kernel
clf=svm.SVC(kernel=my_kernel)
clf.fit(x,y)
#Make predictions and plot
#Linear SVC
plt.figure(figsize=(10,10))

xx,yy=np.meshgrid(np.arange(x[:,0].min()-1, x[:,0].max()+1, 0.02),
                  np.arange(x[:,1].min()-1, x[:,0].max()+1, 0.02))
z=clf.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
plt.contourf(xx,yy,z,cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(x[:,0],x[:,1],c=y,cmap=plt.cm.coolwarm,s=20,edgecolors='k')
plt.title("SVC with customized kernel, score=%f"%clf.score(x,y))
plt.xlim(4,8);plt.ylim(1,5)
plt.xlabel('x1');plt.ylabel('x2')#;plt.axis('tight')

plt.show()
