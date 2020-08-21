import numpy as np
import pandas as pd
from pandas import DataFrame
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model

os.chdir(r'C:\Users\82375\Google 云端硬盘\High-throughput\2020-3-3\ML')
data=pd.read_csv('700-0.3125-0.4375.csv')
#Load data using pandas, 
y=data.iloc[:,0]
x=data.iloc[:,1:15]

#Split the data into 80% train and 20% test data sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
min_max_scaler = MinMaxScaler(feature_range=(0, 1))
#Normalize all data
x_train=min_max_scaler.fit_transform(x_train)
x_test=min_max_scaler.fit_transform(x_test)

#Test different loss functions for GBR
fig=plt.figure(figsize=(10,10))
plt.subplot(221) #loss='ls'
for n_estimators in [100,200,300,400]:
    y_gbr1=GradientBoostingRegressor(n_estimators=n_estimators,learning_rate=0.1,max_depth=1,
                                   random_state=0,loss='ls').fit(x_train,y_train).predict(x_test)
    person1=np.corrcoef(y_gbr1,y_test,rowvar=0)[0][1]
    mae1=mean_absolute_error(y_gbr1,y_test)
    plt.scatter(y_test,y_gbr1,label='n_estimator=%f,Person=%f,MAE=%f'%(n_estimators,person1,mae1))
    plt.plot(y_test,y_test);plt.legend()
    plt.xlabel("Real y");plt.ylabel("Predicted y");plt.title('Loss=ls')

plt.subplot(222) #loss='lad'
for n_estimators in [100,200,300,400]:
    y_gbr2=GradientBoostingRegressor(n_estimators=n_estimators,learning_rate=0.1,max_depth=1,
                                   random_state=0,loss='lad').fit(x_train,y_train).predict(x_test)
    person2=np.corrcoef(y_gbr2,y_test,rowvar=0)[0][1]
    mae2=mean_absolute_error(y_gbr2,y_test)
    plt.scatter(y_test,y_gbr2,label='n_estimator=%f,Person=%f,MAE=%f'%(n_estimators,person2,mae2))
    plt.plot(y_test,y_test);plt.legend()
    plt.xlabel("Real y");plt.ylabel("Predicted y");plt.title('Loss=lad')
    
plt.subplot(223) #loss='huber'
for n_estimators in [100,200,300,400]:
    y_gbr3=GradientBoostingRegressor(n_estimators=n_estimators,learning_rate=0.1,max_depth=1,
                                   random_state=0,loss='huber').fit(x_train,y_train).predict(x_test)
    person3=np.corrcoef(y_gbr3,y_test,rowvar=0)[0][1]
    mae3=mean_absolute_error(y_gbr3,y_test)
    plt.scatter(y_test,y_gbr3,label='n_estimator=%f,Person=%f,MAE=%f'%(n_estimators,person3,mae3))
    plt.plot(y_test,y_test);plt.legend()
    plt.xlabel("Real y");plt.ylabel("Predicted y");plt.title('Loss=huber')    
    
plt.subplot(224) #loss='quantile'
for n_estimators in [100,200,300,400]:
    y_gbr4=GradientBoostingRegressor(n_estimators=n_estimators,learning_rate=0.1,max_depth=1,
                                   random_state=0,loss='quantile').fit(x_train,y_train).predict(x_test)
    person4=np.corrcoef(y_gbr4,y_test,rowvar=0)[0][1]
    mae4=mean_absolute_error(y_gbr4,y_test)
    plt.scatter(y_test,y_gbr4,label='n_estimator=%f,Person=%f,MAE=%f'%(n_estimators,person4,mae4))
    plt.plot(y_test,y_test);plt.legend()
    plt.xlabel("Real y");plt.ylabel("Predicted y");plt.title('Loss=quantile')

plt.show()
