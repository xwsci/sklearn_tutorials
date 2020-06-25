import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

#Load data
X,y=datasets.load_diabetes(return_X_y=True)
X=X[:,np.newaxis,2]
#Define train and test sets
X_train=X[:-20]
X_test=X[-20:]
y_train=y[:-20]
y_test=y[-20:]
#Train the model using the Least Squares methods
regr=linear_model.LinearRegression()
regr.fit(X_train,y_train)
#Make predictions
y_pred=regr.predict(X_test)

#print coefficients
print('Coefficients:%.f'
      % float(regr.coef_))
print('Coefficient of determination:%.2f'
       % mean_squared_error(y_test,y_pred))
print('Coefficient of determination:%.2f'
      % r2_score(y_test,y_pred))

#Plot outputs
fig=plt.figure()
ax1=fig.add_subplot(111)
#ax1.set_title('test')
plt.xlabel('Real data')
plt.ylabel('Predicted data')
ax1.scatter(y_test,y_pred,color='black',marker='*')
ax1.plot(y_test,y_test,color='red',linewidth=3)
#plt.legend('AB')
plt.show()

