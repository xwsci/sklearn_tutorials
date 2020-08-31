import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV

#Load data
x,y=load_diabetes(return_X_y=True)
#Train classifiers
reg1=GradientBoostingRegressor(random_state=1)
reg2=RandomForestRegressor(random_state=1)
reg3=LinearRegression()
ereg=StackingRegressor(estimators=[('gb',reg1),('rf',reg2),('lr',reg3)],
                       final_estimator=RidgeCV())#Default final_estimator=RidgeCV.
#Predict
pred1=reg1.fit(x,y).predict(x[:20])
pred2=reg2.fit(x,y).predict(x[:20])
pred3=reg3.fit(x,y).predict(x[:20])
pred4=ereg.fit(x,y).predict(x[:20])
print(ereg.final_estimator_)
#Plot
plt.plot(pred1,label='GradientBoostingRegressorm,score=%f'%reg1.score(x,y))
plt.plot(pred2,label='RandomForestRegressor,score=%f'%reg2.score(x,y))
plt.plot(pred3,label='LinearRegression,score=%f'%reg3.score(x,y))
plt.plot(pred4,label='StackingRegressor,score=%f'%ereg.score(x,y,sample_weight=None))
plt.plot(y[:20],label='Real data')

plt.legend()
plt.show()



