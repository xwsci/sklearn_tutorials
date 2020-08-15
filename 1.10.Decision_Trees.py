import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

#Create a random dataset
rng=np.random.RandomState(0)
x=np.sort(5*rng.rand(80,1)) #.reshape(1,80)
noise=1*(0.5-rng.rand(80))
y=np.sin(x).ravel()+noise
#Fit regression model
regr_1=DecisionTreeRegressor(max_depth=1)
regr_2=DecisionTreeRegressor(max_depth=5)
regr_3=DecisionTreeRegressor(max_depth=10)
regr_1.fit(x,y)
regr_2.fit(x,y)
regr_3.fit(x,y)
#predict
x_test=np.arange(0.0,5.0,0.01)[:,np.newaxis]
y_1=regr_1.predict(x_test)
y_2=regr_2.predict(x_test)
y_3=regr_3.predict(x_test)
#Plot
plt.scatter(x,y,label='Real Data')
plt.plot(x_test,y_1,label='Max depth = 1')
plt.plot(x_test,y_2,label='Max depth = 5')
plt.plot(x_test,y_3,label='Max depth = 10')
plt.xlabel("x");plt.ylabel("y")
plt.title("Decision Tree Regression");plt.legend()
plt.show()
