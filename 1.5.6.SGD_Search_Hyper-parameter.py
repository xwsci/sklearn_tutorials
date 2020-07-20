import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
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
sgd=SGDRegressor(loss="squared_loss",penalty='l2') #,average=True)
parameters={'alpha':[0.00001,0.0001,0.001,0.01,0.1,1]} #Set test alpha values
sgdCV=GridSearchCV(sgd,parameters) #Using GridSearchCV to find out the optimized alpha
sgdCV.fit(x_train,y_train) #Make predictions

#Make predictions and plot
y_pred=sgdCV.predict(x_test)
error=mean_absolute_error(y_test,y_pred) #mean_squared_error(y_test,y_pred,squared=True)
plt.subplot(121)
plt.title('Loss=squared_loss,MAE=%f'%error) 
plt.xlabel('Real data')
plt.ylabel('Predicted data')
plt.scatter(y_test,y_pred,color='black',marker='*')
plt.plot(y_test,y_test,color='red',linewidth=3)

plt.subplot(122)
mae_list=[]
for alpha in np.linspace(0,1,100): #np.arange(0,1,0.001):
    sgd1=SGDRegressor(loss="squared_loss",penalty='l2',alpha=alpha)
    y_pred1=sgd1.fit(x_train,y_train).predict(x_test)
    mae=mean_absolute_error(y_test,y_pred1)
    mae_list.append(mae)
plt.plot(np.linspace(0,1,100),mae_list,label='alpha=%f'%alpha)
plt.xlabel('Alpha');plt.ylabel('MAE')
plt.show()
