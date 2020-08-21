import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.ensemble import ExtraTreesClassifier

#Load the faces dataset
data=fetch_olivetti_faces()
x,y=data.data,data.target
x=x[y<5]
y=y[y<5]
etc=ExtraTreesClassifier(n_estimators=1000,max_features=128,n_jobs=-1,random_state=0)
etc.fit(x,y)

importances=etc.feature_importances_.reshape(data.images[0].shape)

plt.imshow(importances,cmap=plt.cm.hot)
plt.title('Pixel importances with forests of trees')
plt.show()
