import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel,ExpSineSquared

#Define random data x and y=0.5sin(3x)+noise
x=15*np.random.rand(100,1)
noise=3*(0.5-np.random.rand(x.shape[0]))
y=np.sin(x).ravel()+noise
#Define kernel regression and test hyperparameters
param_grid = {"alpha": [1e0, 1e-1, 1e-2, 1e-3],
              "kernel": [ExpSineSquared(l, p)
                         for l in np.logspace(-2, 2, 10)
                         for p in np.logspace(0, 2, 10)]}
kr=GridSearchCV(KernelRidge(),param_grid=param_grid)
kr.fit(x,y)
#Define GP regression
gp_kernel=ExpSineSquared(1.0,5.0,periodicity_bounds=(1e-2,1e1))+WhiteKernel(1e-1)
gpr=GaussianProcessRegressor(kernel=gp_kernel)
gpr.fit(x,y)
#Plot
x_plot=np.linspace(0,20,1000)[:,None]
y_kr=kr.predict(x_plot)
y_gpr,y_std=gpr.predict(x_plot,return_std=True)

plt.figure(figsize=(10,5))
plt.scatter(x,y,c='k',label='Real data')
plt.plot(x_plot,np.sin(x_plot),color='navy',lw=2,label='True fitting')
plt.plot(x_plot,y_kr,color='turquoise',lw=2,label='KRR %s'%kr.best_params_)
plt.plot(x_plot,y_gpr,color='darkorange',lw=2,label='GPR %s'%gpr.kernel_)
plt.fill_between(x_plot[:,0],y_gpr-y_std,y_gpr+y_std,color='darkorange',alpha=0.2)
plt.xlabel('data')
plt.ylabel('target')
#plt.xlim(0,20);plt.ylim(-4,4)
plt.title('GPR vs. KR')
plt.legend(loc="best",scatterpoints=1)
plt.show()
