import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.linear_model import BayesianRidge,LinearRegression,Ridge
from sklearn.metrics import mean_squared_error

#Gererating simulated data with Gaussian weights
np.random.seed(0)
n_samples,n_features=100,100
x=np.random.randn(n_samples,n_features) #Define x
w=np.zeros(n_features)
relevant_features=np.random.randint(0,n_features,10) #Selected 10 numbers from [0,n_features)
for i in relevant_features:
    w[i]=stats.norm.rvs(loc=0,scale=1./np.sqrt(4))

noise=stats.norm.rvs(loc=0,scale=1./np.sqrt(4),size=n_samples)
y=np.dot(x,w)+noise #Define y
#Plot x and y
plt.plot(x,y)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
#Fit x and y using BayesianRidge and Ridge
clf=BayesianRidge(compute_score=True)
clf.fit(x,y)
ridge=Ridge(alpha=0.5)
ridge.fit(x,y)
#Compare the coefficients predicted by BayesianRidge and Ridge with the real coefficients
plt.plot(clf.coef_,color='green',label="Bayesian Ridge estimate",linewidth=1)
plt.plot(w,color='navy',label="Real coefficients",linewidth=1)
plt.plot(ridge.coef_,color='gold',label="Ridge estimate",linewidth=1)
plt.xlabel("Features")
plt.ylabel("Coefficients")
plt.legend()
plt.show()
#Use BayesianRidge and Ridge to make predictions
y_predict_Bayseian=clf.predict(x)
y_predict_Ridge=ridge.predict(x)
error_Bayseian=mean_squared_error(y,y_predict_Bayseian,squared=False)
error_Ridge=mean_squared_error(y,y_predict_Ridge,squared=False)
#y_mean, y_std = clf.predict(x, return_std=True) #Return the mean and standard deviation of posterior prediction
plt.scatter(y,y_predict_Bayseian,label="Bayseian Ridge estimate, RMSE=%f"
            % error_Bayseian)
plt.scatter(y,y_predict_Ridge,label="Ridge estimate, RMSE=%f"
            % error_Ridge)
plt.xlabel("Real y")
plt.ylabel("Predicted y")
plt.legend()
plt.show()
