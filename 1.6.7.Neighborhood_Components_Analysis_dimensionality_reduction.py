import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NeighborhoodComponentsAnalysis,KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

#Load the faces datasets
x,y=datasets.load_digits(return_X_y=True) #x.shape=(1797,64)
#x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5,stratify=y,random_state=0)
dim=len(x) #dim=64
n_classes=len(np.unique(y)) #n_classes=10

#Reduce the dimension of x from 64 to 2 with NCA
nca=make_pipeline(StandardScaler(),NeighborhoodComponentsAnalysis(n_components=2,random_state=0))
nca.fit(x,y)
x_embedded=nca.transform(x) #print(x_embedded[:,0].min(),x_embedded[:,0].max())
#Train the classifier by KNeighborsClassifier using the dimensionality reduced x
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_embedded,y)
#Plot the dimensionality reduced x and y.
xx,yy=np.meshgrid(np.arange(x_embedded[:,0].min()-1, x_embedded[:,0].max()+50, 10),
                  np.arange(x_embedded[:,1].min()-1, x_embedded[:,0].max()+200, 10))
z=knn.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
plt.contourf(xx,yy,z,cmap=plt.cm.coolwarm, alpha=0.8) #cmap=plt.cm.coolwarm

plt.scatter(x_embedded[:,0],x_embedded[:,1],c=y,s=30,cmap=plt.cm.coolwarm)
plt.title('score=%f'%knn.score(nca.transform(x),y))
plt.show()


