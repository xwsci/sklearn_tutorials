import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

#Generate data
train_size=50
rng=np.random.RandomState(0)
x=rng.uniform(0,5,100).reshape(100,1)
y=np.array(x[:,0]>2.5,dtype=int) #True=1,Falso=0
#Define GPC with and without hyperparameter optimization
gp_fix=GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0),optimizer=None)
gp_opt=GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0),optimizer='fmin_l_bfgs_b')
gp_fix.fit(x[:train_size],y[:train_size])
gp_opt.fit(x[:train_size],y[:train_size])
x_pred=np.linspace(0,5,100)
y_pred_fix=gp_fix.predict_proba(x_pred.reshape(100,1))[:, 1]
y_pred_opt=gp_opt.predict_proba(x_pred.reshape(100,1))[:, 1]

#Plot predicted probability of GPC
plt.figure(figsize=(15,5))
plt.subplot(121)
plt.scatter(x[:train_size,0],y[:train_size],c='k',label='Train data')
plt.scatter(x[train_size:,0],y[train_size:],c='g',label='Test data')
plt.plot(x_pred,y_pred_fix,'r',label='Initial kernel:%s'%gp_fix.kernel_)
plt.plot(x_pred,y_pred_opt,'b',label='Optimized kernel:%s'%gp_opt.kernel_)
plt.xlabel('Feature');plt.ylabel('Class 1 probability')
plt.legend(loc='best')
#Plot LML landscape
plt.subplot(122)
theta0=np.logspace(-2,6,30)
theta1=np.logspace(-1,1,29)
Theta0,Theta1=np.meshgrid(theta0,theta1)
LML=np.array([[gp_opt.log_marginal_likelihood(np.log([Theta0[i, j], Theta1[i, j]]))
        for i in range(Theta0.shape[0])] for j in range(Theta0.shape[1])]).T
plt.plot(np.exp(gp_fix.kernel_.theta)[0], np.exp(gp_fix.kernel_.theta)[1],'ro', zorder=10)
plt.plot(np.exp(gp_opt.kernel_.theta)[0], np.exp(gp_opt.kernel_.theta)[1],'bo', zorder=10)
plt.pcolor(Theta0, Theta1, LML)
plt.xscale("log")
plt.yscale("log")
plt.colorbar()
plt.xlabel("Magnitude")
plt.ylabel("Length-scale")
plt.title("Log-marginal-likelihood")
plt.show()
