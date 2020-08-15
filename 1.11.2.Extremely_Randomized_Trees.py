import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

#Generate data
x,y=make_blobs(n_samples=10000,n_features=10,centers=100,random_state=0)
#Train data with DecisionTree, RandomForest and ExtraTrees Classifier
clf1=DecisionTreeClassifier(max_depth=None,min_samples_split=2,random_state=0)
clf2=RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2,random_state=0)
clf3=ExtraTreesClassifier(n_estimators=10, max_depth=None,min_samples_split=2,random_state=0)
#Plot the scores
score1=cross_val_score(clf1,x,y,cv=5).mean()
score2=cross_val_score(clf2,x,y,cv=5).mean()
score3=cross_val_score(clf3,x,y,cv=5).mean()
score_list=[score1,score2,score3]
plt.bar(range(len(score_list)),score_list,color='rgb',
        tick_label=['DecisionTree','RandomForest','ExtraTrees'],
        width=0.5)
plt.ylim(0.98,1.005)
plt.title('Score comparison')
plt.show()




