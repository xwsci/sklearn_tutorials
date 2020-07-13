import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from random import shuffle

#Import iris dataset
iris=datasets.load_iris() #x=[attribute1, attribute2, attribute3, attribute4];y=[0,1,2]
x=iris.data
y=iris.target

#Reduce the number of features from 4 to 2
lda1=LinearDiscriminantAnalysis(solver='lsqr',shrinkage=None)
lda2=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto')

#train model with different sizes of x and y
lda1_score=[]
lda2_score=[]
sample_number=[]
for N in range(5,100):
    test_size=(100-N)/100
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=test_size,random_state=1)
    lda1.fit(x_train,y_train)
    lda2.fit(x_train,y_train)
    lda1_score.append(lda1.score(x,y))
    lda2_score.append(lda2.score(x,y))
    sample_number.append(x_train.shape[0])

plt.plot(sample_number,lda1_score,label='Fit with no Shrinkage')
plt.plot(sample_number,lda2_score,label='Fit with auto Shrinkage')
plt.xlabel('Number of Samples')
plt.ylabel('Predict score')
plt.legend()
plt.xlim((5,60))
plt.ylim((0.74, 1.01))
plt.show()


