import numpy as np
import pandas as pd
from pandas import DataFrame
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.svm import SVR
import joblib

os.chdir(r'C:\Users\82375\Google 云端硬盘\High-throughput\2020-3-3\ML\G')
data=pd.read_csv('700-0.3125-0.4375.csv')
data_predictnew=pd.read_csv('new-composition.csv') #header=None
#Load data using pandas, 
y=data.iloc[:,0]
x=data.iloc[:,1:15]
x_predictnew=data_predictnew.iloc[:,0:14]

#Split the data into 80% train and 20% test data sets, then normalize all data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
min_max_scaler = MinMaxScaler(feature_range=(0, 1))
x_train=min_max_scaler.fit_transform(x_train)
x_test=min_max_scaler.fit_transform(x_test)
#Support vector regression, train and predict
svr_rbf=SVR(kernel='rbf',C=100,gamma='auto',epsilon=0.1)
svr_linear=SVR(kernel='linear',C=100,gamma='auto')
svr_poly=SVR(kernel='poly',C=100,gamma='auto',degree=3,epsilon=0.1,coef0=1)
svr_rbf.fit(x_train,y_train)
svr_linear.fit(x_train,y_train)
svr_poly.fit(x_train,y_train)
svr_rbf_y_pred=svr_rbf.predict(x_test)
svr_linear_y_pred=svr_linear.predict(x_test)
svr_poly_y_pred=svr_poly.predict(x_test)

#Plot y_test vs. y_pred
fig=plt.figure(figsize=(15,5))   
plt.subplot(131) #RandomForestRegressor
plt.scatter(y_test,svr_rbf_y_pred)
plt.plot(y_test,y_test)
person1=np.corrcoef(svr_rbf_y_pred,y_test,rowvar=0)[0][1]
plt.title("SVR.RBF\nscore=%.3f,r=%f,MAE=%f" % (svr_rbf.score(x_test, y_test),person1,
                                          mean_absolute_error(y_test,svr_rbf_y_pred)))
plt.xlabel("Real y")
plt.ylabel("Predicted y")

plt.subplot(132) #ExtraTreesRegressor
plt.scatter(y_test,svr_linear_y_pred)
plt.plot(y_test,y_test)
person2=np.corrcoef(svr_linear_y_pred,y_test,rowvar=0)[0][1]
plt.title("SVR.Linear\nscore=%.3f,r=%f,MAE=%f" % (svr_linear.score(x_test, y_test),person2,
                                          mean_absolute_error(y_test,svr_linear_y_pred)))
plt.xlabel("Real y")
plt.ylabel("Predicted y")

plt.subplot(133) #GradientBoostingRegressor
plt.scatter(y_test,svr_poly_y_pred)
plt.plot(y_test,y_test)
person3=np.corrcoef(svr_poly_y_pred,y_test,rowvar=0)[0][1]
plt.title("SVR.Polynomial\nscore=%.3f,r=%f,MAE=%f" % (svr_poly.score(x_test, y_test),person3,
                                          mean_absolute_error(y_test,svr_poly_y_pred)))
plt.xlabel("Real y")
plt.ylabel("Predicted y")
plt.show()
