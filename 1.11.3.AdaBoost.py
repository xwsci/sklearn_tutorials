import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import zero_one_loss
from sklearn.ensemble import AdaBoostClassifier

#Datasets
n_estimators=400
learning_rate=1
x,y=datasets.make_hastie_10_2(n_samples=12000,random_state=1)
x_test,y_test=x[2000:],y[2000:]
x_train,y_train=x[:2000],y[:2000]

#Define several Classifier
dt_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
dt_stump.fit(x_train, y_train)
dt_stump_err = 1.0 - dt_stump.score(x_test, y_test)

dt = DecisionTreeClassifier(max_depth=9, min_samples_leaf=1)
dt.fit(x_train, y_train)
dt_err = 1.0 - dt.score(x_test, y_test)

ada_discrete = AdaBoostClassifier(base_estimator=dt_stump,
                                  learning_rate=learning_rate,
                                  n_estimators=n_estimators,
                                  algorithm="SAMME")
ada_discrete.fit(x_train, y_train)
ada_discrete_err = np.zeros((n_estimators,))
for i, y_pred in enumerate(ada_discrete.staged_predict(x_test)):
    ada_discrete_err[i] = zero_one_loss(y_pred, y_test)
ada_discrete_err_train = np.zeros((n_estimators,))
for i, y_pred in enumerate(ada_discrete.staged_predict(x_train)):
    ada_discrete_err_train[i] = zero_one_loss(y_pred, y_train)

ada_real = AdaBoostClassifier(
    base_estimator=dt_stump,
    learning_rate=learning_rate,
    n_estimators=n_estimators,
    algorithm="SAMME.R")
ada_real.fit(x_train, y_train)
ada_real_err = np.zeros((n_estimators,))
for i, y_pred in enumerate(ada_real.staged_predict(x_test)):
    ada_real_err[i] = zero_one_loss(y_pred, y_test)
ada_real_err_train = np.zeros((n_estimators,))
for i, y_pred in enumerate(ada_real.staged_predict(x_train)):
    ada_real_err_train[i] = zero_one_loss(y_pred, y_train)

#Plot
plt.plot([1, n_estimators], [dt_stump_err] * 2, 'k-',
        label='Decision Stump Error')
plt.plot([1, n_estimators], [dt_err] * 2, 'k--',
        label='Decision Tree Error')
plt.plot(np.arange(n_estimators)+1,ada_discrete_err,
         label='Discrete AdaBoost test error',color='red')
plt.plot(np.arange(n_estimators)+1,ada_discrete_err_train,
         label='Discrete AdaBoost train error',color='blue')
plt.plot(np.arange(n_estimators)+1,ada_real_err,
         label='Real AdaBoost test error',color='orange')
plt.plot(np.arange(n_estimators)+1,ada_real_err_train,
         label='Real AdaBoost train error',color='green')
plt.legend(loc='upper right',fancybox=None).get_frame().set_alpha(0.7)
plt.show()

