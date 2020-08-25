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
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
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

#No missing value vs contain missing values
plt.figure(figsize=(10,5))
plt.subplot(121)
y_gbr1=HistGradientBoostingRegressor(max_iter=100,learning_rate=0.1,max_depth=1,
                               random_state=0).fit(x_train,y_train).predict(x_test)
person1=np.corrcoef(y_gbr1,y_test,rowvar=0)[0][1]
mae1=mean_absolute_error(y_gbr1,y_test)
plt.scatter(y_test,y_gbr1,label='Person=%f,MAE=%f'%(person1,mae1))
plt.plot(y_test,y_test);plt.legend()
plt.xlabel("Real y");plt.ylabel("Predicted y");plt.title('No missing value')
print(x_test)

plt.subplot(122)
x_test_missvalue=x_test
for i in [np.arange(0,350,1)]:
    x_test_missvalue[i,0]=np.nan
    x_test_missvalue[i+1,1]=np.nan
    x_test_missvalue[i+2,2]=np.nan
#    x_test_missvalue[i+3,3]=np.nan
    x_test_missvalue[i+4,4]=np.nan
    x_test_missvalue[i+5,5]=np.nan
#    x_test_missvalue[i+6,6]=np.nan
    x_test_missvalue[i+7,7]=np.nan
    x_test_missvalue[i+8,8]=np.nan
#    x_test_missvalue[i+9,9]=np.nan
    x_test_missvalue[i+10,10]=np.nan
    x_test_missvalue[i+11,11]=np.nan
    x_test_missvalue[i+12,12]=np.nan
#    x_test_missvalue[i+13,13]=np.nan
    
print(x_test_missvalue)
y_gbr2=HistGradientBoostingRegressor(max_iter=100,learning_rate=0.1,max_depth=1,
                               random_state=0).fit(x_train,y_train).predict(x_test_missvalue)
person2=np.corrcoef(y_gbr2,y_test,rowvar=0)[0][1]
mae2=mean_absolute_error(y_gbr2,y_test)
plt.scatter(y_test,y_gbr2,label='Person=%f,MAE=%f'%(person2,mae2))
plt.plot(y_test,y_test);plt.legend()
plt.xlabel("Real y");plt.ylabel("Predicted y");plt.title('Contain missing values')

plt.show()
