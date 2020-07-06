import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from scipy import stats
from sklearn.metrics import mean_squared_error

os.chdir(r'C:\Users\82375\Google 云端硬盘\High-throughput\2020-3-3\ML\test')
data=pd.read_csv('Ev-0.000-0.125.csv')

#Gererating simulated data with Gaussian weights
np.random.seed(0)
n_samples,n_features=100,100
x=np.random.randn(n_samples,n_features) #Define x
w=np.zeros(n_features)
relevant_features=np.random.randint(0,n_features,10) #Selected 10 numbers from [0,n_features)
for i in relevant_features:
    w[i]=stats.norm.rvs(loc=0,scale=1./np.sqrt(4))

noise=stats.norm.rvs(loc=0,scale=1./np.sqrt(4),size=n_samples)
y=abs(np.dot(x,w)+noise) #Define y

###Load data using pandas, split the data into 80% train and 20% test data sets
##y=data.iloc[:,0]
##x=data.iloc[:,1:15]
##x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
###Normalize all data
##min_max_scaler = MinMaxScaler(feature_range=(0, 1))
##x_train=min_max_scaler.fit_transform(x_train)
##x_test=min_max_scaler.fit_transform(x_test)
#Train model using Poisson, Tweedie and Gamma regressor
sgd=linear_model.SGDRegressor().fit(x,y)
#perceptron=linear_model.Perceptron().fit(x,y)
pa=linear_model.PassiveAggressiveRegressor().fit(x,y)
#Make predictions
y_pred_sgd=sgd.predict(x)
#y_pred_perceptron=perceptron.predict(x)
y_pred_pa=pa.predict(x)
#Plot predicted data
fig=plt.figure(figsize=(10,5))   
plt.subplot(121) #sgd regression
plt.scatter(y,y_pred_sgd)
plt.plot(y,y)
plt.title("training score : %.3f (%s)" % (sgd.score(x, y),'SGD'))
plt.xlabel("Real y")
plt.ylabel("Predicted y")

plt.subplot(122) #pa regression
plt.scatter(y,y_pred_pa)
plt.plot(y,y)
plt.title("training score : %.3f (%s)" % (pa.score(x, y),'PA'))
plt.xlabel("Real y")
plt.ylabel("Predicted y")
plt.show()
