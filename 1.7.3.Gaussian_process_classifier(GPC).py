import numpy as np
import pandas as pd
from pandas import DataFrame
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import NeighborhoodComponentsAnalysis,KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

os.chdir(r'C:\Users\82375\Google 云端硬盘\High-throughput\2020-3-3\ML\brownmillerite-vs-perovskite')
data=pd.read_csv('brownmillerite-perovskite_2003_3.csv')
#Load data using pandas, 
y=data.iloc[:,4]
x=data.iloc[:,5:19]
#Reduce the dimension of x from 14 to 2
nca=make_pipeline(StandardScaler(),NeighborhoodComponentsAnalysis(n_components=2,random_state=0))
nca.fit(x,y)
x_reduced=nca.transform(x)
#Split the data into 80% train and 20% test data sets and then Normalize all data
x_train,x_test,y_train,y_test=train_test_split(x_reduced,y,test_size=0.2,random_state=0)
min_max_scaler = MinMaxScaler(feature_range=(0, 1))
x_train=min_max_scaler.fit_transform(x_train)
x_test=min_max_scaler.fit_transform(x_test)
#Define Classifier
gpc=GaussianProcessClassifier()
gpc.fit(x_train,y_train)
y_pred=gpc.predict(x_test)
#Plot
xx,yy=np.meshgrid(np.arange(x_test[:,0].min()-1000, x_test[:,0].max()+1000, 50),
                  np.arange(x_test[:,1].min()-1000, x_test[:,0].max()+1000, 50))
z=gpc.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
plt.contourf(xx,yy,z,cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(x_reduced[:,0],x_reduced[:,1],c=y,cmap=plt.cm.coolwarm,s=20,edgecolors='k')
plt.title("GPC, score=%f"%gpc.score(x_test,y_test))
plt.xlabel('x1');plt.ylabel('x2')#;plt.axis('tight')#plt.xlim(4,8);plt.ylim(1,5)
plt.show()
