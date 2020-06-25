import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

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
#Explore the optimized alpha that leads to the minium error 
errors=[]
list_a=np.arange(100)
for a in list_a:
    ridge=linear_model.Ridge(alpha=a) #fit_intercept=False
    ridge.fit(x_train,y_train)
    y_pred=ridge.predict(x_test)
    error=mean_squared_error(y_test,y_pred,squared=False)
    errors.append(error)
#Plot the alpha vs. error curve    
fig=plt.figure()    
ax1=fig.add_subplot(111)
plt.xlabel('alpha')
plt.ylabel('error')
ax1.scatter(list_a,errors,color='black',marker='*')
plt.show()
#Using the optimized alpha (33) to train the machine learning model
ridge=linear_model.Ridge(alpha=33)
ridge.fit(x_train,y_train)
#Make predictions
y_pred=ridge.predict(x_test)
error=mean_squared_error(y_test,y_pred,squared=False)
#Plot real and predicted data, and output the RMSE
fig=plt.figure()
ax1=fig.add_subplot(111)
ax1.set_title('RMSE=%f'
              % error) 
plt.xlabel('Real data')
plt.ylabel('Predicted data')
ax1.scatter(y_test,y_pred,color='black',marker='*')
ax1.plot(y_test,y_test,color='red',linewidth=3)
plt.show()
