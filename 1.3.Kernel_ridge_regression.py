import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.kernel_ridge import KernelRidge

rng=np.random.RandomState(0)
################ Task 1: Examine the prediction scores of SVR and KR vs. sample sizes
#Generate sample data
x=5*rng.rand(10000,1)
y=np.sin(x).ravel()
noise=3*(0.5-rng.rand(int(x.shape[0]/5)))
y[::5]=y[::5]+noise
x_plot=np.linspace(0,5,10*x.shape[0]).reshape(10*x.shape[0],1)
#Fit regression models. Here GridSearchCV is used to adjust hyper-parameters
train_size=100
svr=GridSearchCV(SVR(kernel='rbf',gamma=0.1),
                 param_grid={'C':[1e0,1e1,1e2,1e3],'gamma':np.logspace(-2,2,5)})
kr=GridSearchCV(KernelRidge(kernel='rbf',gamma=0.1),
                param_grid={'alpha':[1e0,0.1,1e-2,1e-3],'gamma':np.logspace(-2,2,5)})
t0=time.time();svr.fit(x[:train_size],y[:train_size]);svr_fit_time=time.time()-t0
t0=time.time();kr.fit(x[:train_size],y[:train_size]);kr_fit_time=time.time()-t0
t0=time.time();y_pred_svr=svr.predict(x_plot);svr_pred_time=time.time()-t0
t0=time.time();y_pred_kr=kr.predict(x_plot);kr_pred_time=time.time()-t0
#Plot predicted results
#sv_ind = svr.best_estimator_.support_
#plt.scatter(x[sv_ind],y[sv_ind],c='r',s=50,label='SVR support vectors',zorder=2,edgecolors=(0,0,0))
plt.scatter(x[:train_size],y[:train_size],c='k',label='data',zorder=1,edgecolors=(0,0,0))
plt.plot(x_plot,y_pred_svr,c='r',label='SVR (fit:%.3fs, predict:%.3fs)'% (svr_fit_time,svr_pred_time))
plt.plot(x_plot,y_pred_kr,c='g',label='KR (fit:%.3fs, predict:%.3fs)'% (kr_fit_time,kr_pred_time))
plt.xlabel('data');plt.ylabel('target');plt.title('SVR vs. KR,Train size=%d' % train_size);plt.legend();plt.show()

################ Task 2: Examine the time consumption of SVR and KR vs. sample sizes
time_svr_train=[]
time_kr_train=[]
time_svr_pred=[]
time_kr_pred=[]
for train_size in np.linspace(10,500,10):
    t0=time.time();svr.fit(x[:int(train_size)],y[:int(train_size)]);svr_fit_time=time.time()-t0
    t0=time.time();kr.fit(x[:int(train_size)],y[:int(train_size)]);kr_fit_time=time.time()-t0
    t0=time.time();y_pred_svr=svr.predict(x_plot);svr_pred_time=time.time()-t0
    t0=time.time();y_pred_kr=kr.predict(x_plot);kr_pred_time=time.time()-t0
    time_svr_train.append(svr_fit_time)
    time_kr_train.append(svr_pred_time)
    time_svr_pred.append(kr_fit_time)
    time_kr_pred.append(kr_pred_time)
plt.plot(np.linspace(100,10000,10),time_svr_train, 'o-',color='r',label='SVR Train')
plt.plot(np.linspace(100,10000,10),time_kr_train, 'o-',color='b',label='KR Train')
plt.plot(np.linspace(100,10000,10),time_svr_pred, 'o--',color='r',label='SVR Predict')
plt.plot(np.linspace(100,10000,10),time_kr_pred, 'o--',color='b',label='KR Predict')
plt.xlabel('Train set size')
plt.ylabel('Time consumption')
plt.legend()
plt.show()




















