import numpy as np
import pandas as pd
from pandas import DataFrame
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance

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

#Train model and count the feature importances
y_gbr=GradientBoostingRegressor(n_estimators=200,learning_rate=0.1,max_depth=1,
                               random_state=0,loss='ls').fit(x_train,y_train)
feature_importance=y_gbr.feature_importances_
#Plot
plt.bar(np.arange(0,14),feature_importance,align='center')
plt.xticks(np.arange(0,14),np.array(data.columns)[1:15])
plt.title('Feature Importance')
plt.show()
