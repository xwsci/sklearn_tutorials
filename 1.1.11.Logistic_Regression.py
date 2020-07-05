import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs #Generate isotropic Gaussian blobs for clustering
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error

#Make 3-class dtaset for classification
centers=[[-5,0],[0,1.5],[5,-1]]
x,y=make_blobs(n_samples=1000,centers=centers,random_state=0)
transformation=[[0.4,0.2],[-0.4,1.2]]
x=np.dot(x,transformation)

for multi_class in ('multinomial','ovr'):
    clf=LogisticRegression(solver='sag', max_iter=100, random_state=0,
                             multi_class=multi_class).fit(x, y)
    #Create a mesh to plot in
    h=0.02 #step size in the mesh
    x_min,x_max=x[:,0].min()-1,x[:,0].max()+1
    y_min,y_max=x[:,1].min()-1,x[:,1].max()+1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    #Plot the decision boundary
    z=clf.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)

    plt.contourf(xx,yy,z)#,cmap=plt.cm.Paired)
    plt.title("Decision surface of LogisticRegression (%s)" % multi_class)
    #Print the training scores
    plt.xlabel("training score : %.3f (%s)" % (clf.score(x, y), multi_class))
    plt.axis('tight')

    #Plot the training points
    for i,color in zip(clf.classes_,"bry"):
        idx=np.where(y==i)
        plt.scatter(x[idx,0],x[idx,1],c=color,edgecolor='black',s=20)

    plt.show()








    


