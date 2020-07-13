import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#Import iris dataset
iris=datasets.load_iris() #x=[attribute1, attribute2, attribute3, attribute4];y=[0,1,2]
x=iris.data
y=iris.target
#Reduce the number of features from 4 to 2
pca=PCA(n_components=2) #In principle, there a 4 features, here try to reduce the N_feature to 2
x_reduce=pca.fit(x).transform(x) #Reduce 4 features to 2.
lda=LinearDiscriminantAnalysis(n_components=2)
x_reduce2=lda.fit(x,y).transform(x)

fig=plt.figure(figsize=(10,8))
fig.add_subplot(221)
plt.scatter(x_reduce[y==0,0],x_reduce[y==0,1],color='navy',alpha=0.8,lw=2,label='iris.0')
plt.scatter(x_reduce[y==1,0],x_reduce[y==1,1],color='turquoise',alpha=0.8,lw=2,label='iris.1')
plt.scatter(x_reduce[y==2,0],x_reduce[y==2,1],color='darkorange',alpha=0.8,lw=2,label='iris.2')
plt.legend()
plt.xlabel("Reduced featre 1")
plt.ylabel("Reduced featre 2")
plt.title("Reduce feature by PCA")

fig.add_subplot(222)
plt.scatter(x_reduce2[y==0,0],x_reduce2[y==0,1],color='navy',alpha=0.8,lw=2,label='iris.0')
plt.scatter(x_reduce2[y==1,0],x_reduce2[y==1,1],color='turquoise',alpha=0.8,lw=2,label='iris.1')
plt.scatter(x_reduce2[y==2,0],x_reduce2[y==2,1],color='darkorange',alpha=0.8,lw=2,label='iris.2')
plt.legend()
plt.xlabel("Reduced featre 1")
plt.ylabel("Reduced featre 2")
plt.title("Reduce feature by LDA")

#Predict target
#y_pred_pca=pca.fit(x_reduce).components_
y_pred_lda=lda.fit(x_reduce2,y).predict(x_reduce2)

#How to usu an unsupervised method to perform predictions
fig.add_subplot(223) ##plt.subplot(3,3,9)
eqs = []
eqs.append((r"$How$"))
eqs.append((r"$Does$"))
eqs.append((r"$Unsurprivsed$"))
eqs.append((r"$Classifier$"))
eqs.append((r"$Work$"))
eqs.append((r"$?$"))

for i in range(20):
    index = np.random.randint(0,len(eqs))
    eq = eqs[index]
    size = np.random.uniform(12,32)
    x,y = np.random.uniform(0,1,2)
    alpha = np.random.uniform(0.25,.75)
    plt.text(x, y, eq, ha='center', va='center', color="#11557c", alpha=alpha,
             transform=plt.gca().transAxes, fontsize=size, clip_on=True)
plt.xticks([]), plt.yticks([])

#3D Plot reduced_x1, reduced_x2, y_pred
ax_lda = fig.add_subplot(224, projection='3d') #ax_lda=plt.axes(projection='3d') Plot a singe 3D picture
ax_lda.scatter3D(x_reduce2[y_pred_lda==0,0],x_reduce2[y_pred_lda==0,1],y_pred_lda[y_pred_lda==0],c='y',label='Setosa')
ax_lda.scatter3D(x_reduce2[y_pred_lda==1,0],x_reduce2[y_pred_lda==1,1],y_pred_lda[y_pred_lda==1],c='g',label='Versicolor')
ax_lda.scatter3D(x_reduce2[y_pred_lda==2,0],x_reduce2[y_pred_lda==2,1],y_pred_lda[y_pred_lda==2],c='b',label='Virginica')
ax_lda.set_xlabel('Reduced feature x1')
ax_lda.set_ylabel('Reduced feature x2')
ax_lda.set_zlabel('Target')
ax_lda.set_title('Reduce feature by LDA')
ax_lda.legend()
plt.show()


