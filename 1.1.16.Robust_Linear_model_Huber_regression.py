import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import HuberRegressor,Ridge

#Generate data
rng=np.random.RandomState(0)
x,y=make_regression(n_samples=20,n_features=1,random_state=0,noise=4.0,bias=100.0)

#Add four strong outliers to the dataset
x_outliers=rng.normal(0,0.5,size=(4,1))
y_outliers=rng.normal(0,2.0,size=4)
x_outliers[:2,:]+=x.max()+x.mean()/4.
x_outliers[2:,:]+=x.min()-x.mean()/4.
y_outliers[:2]+=y.min()-y.mean()/4.
y_outliers[2:]+=y.max()+y.mean()/4.
x=np.vstack((x,x_outliers))
y=np.concatenate((y,y_outliers),axis=0)
#Fit the huber regressor
epsilon_list=[1.35,1.5,1.75,1.9] #Epsilon controls the number of outliers
for epsilon in epsilon_list:
    huber=HuberRegressor(alpha=0.0,epsilon=epsilon) #alpha is the penalty parameter
    huber.fit(x,y)
    x_test=np.arange(x.min(),x.max()+1).reshape(-1,1)
    y_pred=huber.predict(x_test) #make predictions
    plt.plot(x_test,y_pred,label="Epsilon=%2f" % epsilon)

plt.title("Comparison of differen epsilon in Huber regression")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc=0)
plt.scatter(x,y)
plt.show()
