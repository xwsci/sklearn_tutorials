import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

#Import data
iris=datasets.load_iris()
x,y=iris.data[:,[0,2]],iris.target
#Training classifiers
clf1=DecisionTreeClassifier(max_depth=4)
clf2=KNeighborsClassifier(n_neighbors=7)
clf3=SVC(gamma=0.1,kernel='rbf',probability=True)
eclf=VotingClassifier(estimators=[('dt',clf1),('knn',clf2),('svc',clf3)],
                     voting='soft',weights=[2,1,2])
clf1.fit(x,y)
clf2.fit(x,y)
clf3.fit(x,y)
eclf.fit(x,y)
#Plotting decision regions
x_min,x_max=x[:,0].min()-1,x[:,0].max()+1
y_min,y_max=x[:,1].min()-1,x[:,1].max()+1
xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))

plt.figure(figsize=(10,10))
plt.subplot(221)
z=clf1.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
plt.contourf(xx,yy,z,alpha=0.4)
plt.scatter(x[:,0],x[:,1],c=y,s=20,edgecolor='k')
plt.title('DecisionTreeClassifier, Score=%f'
          %cross_val_score(clf1,x,y,scoring='accuracy',cv=5).mean())

plt.subplot(222)
z=clf2.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
plt.contourf(xx,yy,z,alpha=0.4)
plt.scatter(x[:,0],x[:,1],c=y,s=20,edgecolor='k')
plt.title('KNeighborsClassifier, Score=%f'
          %cross_val_score(clf2,x,y,scoring='accuracy',cv=5).mean())

plt.subplot(223)
z=clf3.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
plt.contourf(xx,yy,z,alpha=0.4)
plt.scatter(x[:,0],x[:,1],c=y,s=20,edgecolor='k')
plt.title('SVC, Score=%f'
          %cross_val_score(clf3,x,y,scoring='accuracy',cv=5).mean())

plt.subplot(224)
z=eclf.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
plt.contourf(xx,yy,z,alpha=0.4)
plt.scatter(x[:,0],x[:,1],c=y,s=20,edgecolor='k')
plt.title('VotingClassifier, Score=%f'
          %cross_val_score(eclf,x,y,scoring='accuracy',cv=5).mean())

plt.show()
