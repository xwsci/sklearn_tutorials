import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV

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
#Estimate the number of non-zero coefficients giving the least error over the cross-validation folds.
N_nonzero_coef=OrthogonalMatchingPursuitCV().fit(x_train,y_train).n_nonzero_coefs_
#Using the optimized number of non-zero coefficients to train the machine learning model
omp=OrthogonalMatchingPursuit(n_nonzero_coefs=N_nonzero_coef)
omp.fit(x_train,y_train)
coef=omp.coef_  #the coefficient of each feature
idx_r,=coef.nonzero() #the order number of features that has non-zero coefficient
#Plot features with non-zero co-efficients vs. non-zero coefficients
plt.stem(x.columns.values[idx_r],coef[idx_r]) #plot all feature vs. non-zero coeffcients
#plt.stem(x.columns.values,coef) #plot all feature vs. coeffcients
plt.title("Non-zero cofficients of each feature")
plt.ylabel('Non-zero cofficients')
plt.show()
#Make predictions
y_pred=omp.predict(x_test)
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
