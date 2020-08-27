from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

#import data
iris=datasets.load_iris()
x,y=iris.data[:,:],iris.target

clf1=LogisticRegression()
clf2=RandomForestClassifier()
clf3=GaussianNB()
eclf=VotingClassifier(estimators=[('lr',clf1),('rf',clf2),('gnb',clf3)],
                     voting='hard')
for clf,label in zip([clf1,clf2,clf3,eclf],['lr','rf','gnb','voting']):
    scores=cross_val_score(clf,x,y,scoring='accuracy',cv=5)
    print('%s Score=%f (+/- %f)'%(label,scores.mean(),scores.std()))

#Output is:
#lr Score=0.973333 (+/- 0.024944)
#rf Score=0.960000 (+/- 0.024944)
#gnb Score=0.953333 (+/- 0.026667)
#voting Score=0.960000 (+/- 0.024944)
