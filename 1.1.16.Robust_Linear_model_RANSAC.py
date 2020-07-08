import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model,datasets

#Genearte inlier data
n_samples=1000
n_outliers=50
x,y,coef=datasets.make_regression(n_samples=n_samples,n_features=1,n_informative=1,noise=10,coef=True,random_state=0)
#Add outlier data
np.random.seed(0)
x[:n_outliers]=3+0.5*np.random.normal(size=(n_outliers,1))
y[:n_outliers]=-3+10*np.random.normal(size=n_outliers)

#Fit line using Ridge
ridge=linear_model.Ridge()
ridge.fit(x,y)

#Robustly fit linear model with RANSAC algorithm
ransac=linear_model.RANSACRegressor()
ransac.fit(x,y)
inlier_mask=ransac.inlier_mask_ #Classify which data is inlier (True) and which ones are outliers (False)
outlier_mask=np.logical_not(inlier_mask) #Classify which data is inlier (False) and which ones are outliers (True)

#Predict data of estimated models
x_test=np.arange(x.min(),x.max()).reshape(-1,1)
ridge_y=ridge.predict(x_test)
ransac_y=ransac.predict(x_test)
#Plot lines and inlier/outlier data
plt.scatter(x[inlier_mask],y[inlier_mask],color='yellowgreen',marker='.',label='Inliers')
plt.scatter(x[outlier_mask],y[outlier_mask],color='gold',marker='x',label='Outliers')
plt.plot(x_test,ridge_y,color='navy',linewidth=2,label='Ridge regressor')
plt.plot(x_test,ransac_y,color='red',linewidth=2,label='RANSAC regressor')
plt.legend()
plt.xlabel('Data x')
plt.ylabel('Data y')
plt.show()

