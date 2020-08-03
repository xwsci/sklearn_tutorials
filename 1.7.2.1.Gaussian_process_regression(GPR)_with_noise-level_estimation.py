import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,WhiteKernel

#Define random data x and y=0.5sin(3x)+noise
x=np.random.RandomState(0).uniform(0,5,20)[:,np.newaxis]
noise=np.random.RandomState(0).normal(0,0.5,x.shape[0])
y=0.5*np.sin(3*x[:,0])+noise
#Define two kernels, kernel1 has higher noise level while kernel2 has lower noise level
kernel1=1.0*RBF(length_scale=100.0,length_scale_bounds=(1e-2,1e3))+WhiteKernel(noise_level=1,noise_level_bounds=(1e-10,1e+1))
kernel2=1.0*RBF(length_scale=1.0,length_scale_bounds=(1e-2,1e3))+WhiteKernel(noise_level=1e-5,noise_level_bounds=(1e-10,1e+1))
#Prediction with different kerlels
plt.subplot(121)
gp1=GaussianProcessRegressor(kernel=kernel1,alpha=0.0).fit(x,y)
x_=np.linspace(0,5,100)
y_mean,y_cov=gp1.predict(x_[:,np.newaxis],return_cov=True)
plt.plot(x_,y_mean,'k',lw=3,zorder=9,label='Prediction')
plt.fill_between(x_,y_mean-np.sqrt(np.diag(y_cov)),y_mean+np.sqrt(np.diag(y_cov)),alpha=0.5,color='k',label='Error')
plt.plot(x_,0.5*np.sin(3*x_),'r',lw=3,zorder=9,label='Real')
plt.scatter(x[:,0],y,c='r',s=50,zorder=10,edgecolors=(0,0,0))
plt.legend();plt.tight_layout()
plt.subplot(122)
gp2=GaussianProcessRegressor(kernel=kernel2,alpha=0.0).fit(x,y)
x_=np.linspace(0,5,100)
y_mean,y_cov=gp2.predict(x_[:,np.newaxis],return_cov=True)
plt.plot(x_,y_mean,'k',lw=3,zorder=9,label='Prediction')
plt.fill_between(x_,y_mean-np.sqrt(np.diag(y_cov)),y_mean+np.sqrt(np.diag(y_cov)),alpha=0.5,color='k',label='Error')
plt.plot(x_,0.5*np.sin(3*x_),'r',lw=3,zorder=9,label='Real')
plt.scatter(x[:,0],y,c='r',s=50,zorder=10,edgecolors=(0,0,0))
plt.legend();plt.tight_layout()
plt.show()
