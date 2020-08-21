import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.ensemble import (RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#Parameters
n_classes=3
n_estimators=30
cmap=plt.cm.RdYlBu
plot_step_coarser =0.5  # step widths for coarse classifier guesses
RANDOM_SEED =13  # fix the seed on each iteration

#Load data
iris=load_iris()
models=[DecisionTreeClassifier(max_depth=None),
        RandomForestClassifier(n_estimators=n_estimators),
        ExtraTreesClassifier(n_estimators=n_estimators),
        AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),n_estimators=n_estimators)]

plot_idx=1;plt.figure(figsize=(16,8))
for pair in ([0,1],[0,2],[2,3]):
    for model in models:
        x=iris.data[:,pair]
        y=iris.target
        #Shuffle
        idx=np.arange(x.shape[0])
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(idx)
        x=x[idx];y=y[idx]
        #Standardize
        mean=x.mean(axis=0)
        std=x.std(axis=0)
        x=(x-mean)/std
        #Train
        model.fit(x,y)
        scores=model.score(x,y)
        #Plot
        model_title = str(type(model)).split('.')[-1][:-2][:-len('Classifier')]
        plt.subplot(3,4,plot_idx)
        if plot_idx <= len(models):
            plt.title(model_title,fontsize=9)

        #Plot decision boundary using a mesh
        x_min,x_max=x[:,0].min()-1,x[:,0].max()+1
        y_min,y_max=x[:,1].min()-1,x[:,1].max()+1
        xx,yy=np.meshgrid(np.arange(x_min,x_max,0.02),
                          np.arange(y_min,y_max,0.02))
        if isinstance(model,DecisionTreeClassifier):
            z=model.predict(np.c_[xx.ravel(),yy.ravel()])
            z=z.reshape(xx.shape)
            cs=plt.contourf(xx,yy,z,cmap=cmap)
        else:
            estimator_alpha=1.0/len(model.estimators_)
            for tree in model.estimators_:
                z=tree.predict(np.c_[xx.ravel(),yy.ravel()])
                z=z.reshape(xx.shape)
                cs=plt.contourf(xx,yy,z,alpha=estimator_alpha,cmap=cmap)
        xx_coarser, yy_coarser = np.meshgrid(np.arange(x_min, x_max, plot_step_coarser),
                                             np.arange(y_min, y_max, plot_step_coarser))
        z_points_coarser = model.predict(np.c_[xx_coarser.ravel(),
                                               yy_coarser.ravel()]).reshape(xx_coarser.shape)
        cs_points = plt.scatter(xx_coarser, yy_coarser, s=15,c=z_points_coarser,
                                cmap=cmap,edgecolors="none")

        plt.scatter(x[:,0],x[:,1],c=y,cmap=ListedColormap(['r','y','b']),edgecolor='k',s=20)
        plot_idx+=1

plt.suptitle("Classifiers on feature subsets of the Iris dataset",fontsize=12)
plt.axis('tight')
plt.show()



















        
