import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.datasets import load_iris

#Import iris dataset
iris=load_iris() #x=[attribute1, attribute2, attribute3, attribute4];y=[0,1,2]
x=iris.data[:,:2]
y=iris.target
#Fit the model using DecisionTreeClassifier
clf=DecisionTreeClassifier()
clf.fit(x,y)
#Plot
xx,yy=np.meshgrid(np.arange(x[:,0].min()-1, x[:,0].max()+1, 0.02),
                  np.arange(x[:,1].min()-1, x[:,0].max()+3, 0.02))
z=clf.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
plt.figure(figsize=(16,8))
plt.subplot(121)
plt.contourf(xx,yy,z,cmap=plt.cm.RdYlBu, alpha=0.8)
plt.scatter(x[:,0],x[:,1],c=y,cmap=plt.cm.RdYlBu,s=20,edgecolors='k')
plt.title("SVC with customized kernel, score=%f"%clf.score(x,y))
plt.xlim(4,8);plt.ylim(1,5)
plt.xlabel('x1');plt.ylabel('x2')#;plt.axis('tight')
#Plot tree
plt.subplot(122)
plot_tree(clf,filled=True)
plt.show()

