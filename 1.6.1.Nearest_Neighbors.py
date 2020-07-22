import numpy as np
from sklearn.neighbors import NearestNeighbors

x=np.array([0,1,2,3,4,5]).reshape(-1,1)
y=np.array([1500]).reshape(-1,1)
nbrs=NearestNeighbors(n_neighbors=1,algorithm='auto')
nbrs.fit(x)
print(nbrs.kneighbors(y))

#The output is: (array([[1495.]]), array([[5]], dtype=int64))




