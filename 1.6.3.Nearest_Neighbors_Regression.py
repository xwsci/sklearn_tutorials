import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn import neighbors
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

os.chdir(r'C:\Users\82375\Google 云端硬盘\High-throughput\2020-3-3\ML\G')
data=pd.read_csv('400-0.375-0.5.csv')

#Load data using pandas, split the data into 80% train and 20% test data sets
y=data.iloc[:,0]
x=data.iloc[:,1:15]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
#Normalize all data
min_max_scaler = MinMaxScaler(feature_range=(0, 1))
x_train=min_max_scaler.fit_transform(x_train)
x_test=min_max_scaler.fit_transform(x_test)
#Taining model using SGD regressor
knn_uniform=neighbors.KNeighborsRegressor(5,weights='uniform') #,average=True)
knn_distance=neighbors.KNeighborsRegressor(5,weights='distance') #,average=True)
knn_uniform.fit(x_train,y_train)
knn_distance.fit(x_train,y_train)

#Make predictions and plot
y_pred=knn_uniform.predict(x_test)
error=mean_absolute_error(y_test,y_pred) #mean_squared_error(y_test,y_pred,squared=True)
fig=plt.figure(figsize=(10,5))
ax1=fig.add_subplot(121)
ax1.set_title('weights=\'uniform\',MAE=%f'%error) 
plt.xlabel('Real data')
plt.ylabel('Predicted data')
ax1.scatter(y_test,y_pred,color='black',marker='*')
ax1.plot(y_test,y_test,color='red',linewidth=3)

y_pred=knn_distance.predict(x_test)
error=mean_absolute_error(y_test,y_pred) #mean_squared_error(y_test,y_pred,squared=True)
ax1=fig.add_subplot(122)
ax1.set_title('weights=\'distance\',MAE=%f'%error) 
plt.xlabel('Real data')
plt.ylabel('Predicted data')
ax1.scatter(y_test,y_pred,color='black',marker='*')
ax1.plot(y_test,y_test,color='red',linewidth=3)

plt.show()
