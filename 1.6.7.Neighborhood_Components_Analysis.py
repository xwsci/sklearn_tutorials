import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from matplotlib import cm
from scipy.special import logsumexp

#Load the faces datasets
x,y=make_classification(n_samples=9,n_features=2,n_informative=2,
                    #Number of samples, total features, informative features,
                        n_redundant=0,n_classes=3,n_clusters_per_class=1,
                    #Number redundant features,classes, clusters per class
                        class_sep=1.0,random_state=0)
                    #The factor multiplying the hypercube size
plt.figure(figsize=(15,5))
plt.subplot(121)
for i in range(x.shape[0]):
    plt.text(x[i,0],x[i,1],str(i),va='center',ha='center')
    plt.scatter(x[i,0],x[i,1],s=300,c=cm.Set1(y[[i]]),alpha=0.4)
plt.title('Original points')
i=3 #Focus on point 3
for j, pt_j in enumerate(x): #enumerate(x): print the line number as j, x as pt_j
    if i!=j:
        line=([x[3][0],pt_j[0]],[x[3][1],pt_j[1]])
        plt.plot(*line,c=cm.Set1(y[j]),linewidth=1) #thickness[j])
plt.axis('equal')
         
plt.subplot(122)
nca=NeighborhoodComponentsAnalysis(max_iter=100,random_state=0)
nca=nca.fit(x,y)
x_embedded=nca.transform(x)
for i in range(len(x)):
    plt.text(x_embedded[i,0],x_embedded[i,1],str(i),va='center',ha='center')
    plt.scatter(x_embedded[i,0],x_embedded[i,1],s=300,c=cm.Set1(y[[i]]), alpha=0.4)
    plt.title('NCA embedding')
    plt.axis('equal')
i=3
for j, pt_j in enumerate(x_embedded): #enumerate(x): print the line number as j, x as pt_j
    if i!=j:
        line=([x_embedded[3][0],pt_j[0]],[x_embedded[3][1],pt_j[1]])
        plt.plot(*line,c=cm.Set1(y[j]),linewidth=1) #thickness[j])

plt.show()
