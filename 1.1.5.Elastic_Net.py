import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import linear_model

rng=np.random.RandomState(0) #Ensure that each time we can get the same random numbers

os.chdir(r'C:\Users\82375\Google 云端硬盘\High-throughput\2020-3-3\ML\test')
data=pd.read_csv('Ev-0.000-0.125.csv')

#Load data using pandas, split the data into 80% train and 20% test data sets
y=data.iloc[:,0]
x=data.iloc[:,1:15]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
#Normalize all data
min_max_scaler = MinMaxScaler(feature_range=(0, 1))
x_train=min_max_scaler.fit_transform(x_train)
x_test=min_max_scaler.fit_transform(x_test)
#Train the machine learning model: Enet vs. Lasso vs. Ridge
enet=linear_model.ElasticNet(alpha=0.0003,l1_ratio=0.5)
enet.fit(x_train,y_train)
lasso=linear_model.Lasso(alpha=0.0003)
lasso.fit(x_train,y_train)
ridge=linear_model.Ridge(alpha=33)
ridge.fit(x_train,y_train)
#Make predictions: Enet vs. Lasso vs. Ridge
y_pred_enet=enet.predict(x_test)
y_pred_lasso=lasso.predict(x_test)
y_pred_ridge=ridge.predict(x_test)
error_enet=mean_squared_error(y_test,y_pred_enet,squared=False)
error_lasso=mean_squared_error(y_test,y_pred_lasso,squared=False)
error_ridge=mean_squared_error(y_test,y_pred_ridge,squared=False)
r2_score_enet=r2_score(y_test,y_pred_enet)
r2_score_lasso=r2_score(y_test,y_pred_lasso)
r2_score_ridge=r2_score(y_test,y_pred_ridge)

#Plot real and predicted data, and output the RMSE: Enet vs. Lasso vs. Ridge
fig=plt.figure(figsize=(15,5))
#lasso
ax_lasso=fig.add_subplot(131)
ax_lasso.set_title(r'$\rho$=1,Lasso')
plt.text(-6,6,r'RMSE=%f,$R^2$=%f'
              % (error_lasso,r2_score_lasso),fontsize=12)
plt.xlabel('Real data')
plt.ylabel('Predicted data')
ax_lasso.scatter(y_test,y_pred_lasso,color='black',marker='*')
ax_lasso.plot(y_test,y_test,color='red',linewidth=3)
#enet
ax_enet=fig.add_subplot(132)
ax_enet.set_title(r'$\rho$=0.5,Elastic-Net')
plt.text(-6,6,'RMSE=%f,$R^2$=%f'
              % (error_enet,r2_score_enet),fontsize=12)
plt.xlabel('Real data')
plt.ylabel('Predicted data')
ax_enet.scatter(y_test,y_pred_enet,color='black',marker='*')
ax_enet.plot(y_test,y_test,color='red',linewidth=3)
#ridge
ax_ridge=fig.add_subplot(133)
ax_ridge.set_title(r'$\rho$=0,Ridge')
plt.text(-6,6,'RMSE=%f,$R^2$=%f'
              % (error_ridge,r2_score_ridge),fontsize=12)
plt.xlabel('Real data')
plt.ylabel('Predicted data')
ax_ridge.scatter(y_test,y_pred_ridge,color='black',marker='*')
ax_ridge.plot(y_test,y_test,color='red',linewidth=3)

plt.show()
#Output coefficients
plt.stem(data.columns.values[1:15],lasso.coef_,label=r'$\rho$=1,Lasso',markerfmt='o',linefmt='blue',use_line_collection=True)
plt.stem(data.columns.values[1:15],enet.coef_,label=r'$\rho$=0.5,Elastic-Net',markerfmt='x',linefmt='orange',use_line_collection=True)
plt.stem(data.columns.values[1:15],ridge.coef_,label=r'$\rho$=0,Ridge',markerfmt='D',linefmt='green',use_line_collection=True)
plt.title('Coefficients of each descriptor')
plt.legend()
plt.show()
