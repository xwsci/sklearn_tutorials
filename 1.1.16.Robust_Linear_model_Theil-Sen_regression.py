import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,TheilSenRegressor,RANSACRegressor

######## Task 1: Genearte inlier data: Outliers only in the y direction
np.random.seed(0)
n_samples=200
x=np.random.randn(n_samples)
w=3
c=2
noise=0.1*np.random.randn(n_samples)
y=w*x+c+noise #y=3x+2+noise
y[-20:]+=-20*x[-20:] #10% y outliers
#Fit with Linear, TheilSen and RANSAC regressors
line=LinearRegression()
ransac=RANSACRegressor()
theilsen=TheilSenRegressor()
line.fit(x.reshape(-1,1),y)
ransac.fit(x.reshape(-1,1),y)
theilsen.fit(x.reshape(-1,1),y)
#Make predictions
x_test=np.arange(x.min(),x.max()+1).reshape(-1,1)
y_pred_line=line.predict(x_test)
y_pred_ransac=ransac.predict(x_test)
y_pred_theilsen=theilsen.predict(x_test)
#Plot
plt.scatter(x,y,color='indigo',marker='x',s=20)
plt.plot(x_test,y_pred_line,label='Line regression')
plt.plot(x_test,y_pred_ransac,label='RANSAC regression')
plt.plot(x_test,y_pred_theilsen,label='TheilSen regression')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title("y outliers")
plt.axis('tight')
plt.show()

######## Task 2: Genearte inlier data: Outliers only in the x direction
np.random.seed(0)
n_samples=200
x=np.random.randn(n_samples)
w=3
c=2
noise=0.1*np.random.randn(n_samples)
y=w*x+c+noise #y=3x+2+noise
x[-20:]=9.9 #10% x outliers
y[-20:]+=22
#Fit with Linear, TheilSen and RANSAC regressors
line=LinearRegression()
ransac=RANSACRegressor()
theilsen=TheilSenRegressor()
line.fit(x.reshape(-1,1),y)
ransac.fit(x.reshape(-1,1),y)
theilsen.fit(x.reshape(-1,1),y)
#Make predictions
x_test=np.arange(x.min(),x.max()+1).reshape(-1,1)
y_pred_line=line.predict(x_test)
y_pred_ransac=ransac.predict(x_test)
y_pred_theilsen=theilsen.predict(x_test)
#Plot
plt.scatter(x,y,color='indigo',marker='x',s=20)
plt.plot(x_test,y_pred_line,label='Line regression')
plt.plot(x_test,y_pred_ransac,label='RANSAC regression')
plt.plot(x_test,y_pred_theilsen,label='TheilSen regression')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title("x outliers")
plt.axis('tight')
plt.show()












