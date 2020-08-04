import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct

xx,yy=np.meshgrid(np.linspace(-3,3,500),np.linspace(-3,3,500))
np.random.seed(0)
x=np.random.randn(200,2)
y=np.logical_xor(x[:,0]>0,x[:,1]>0)

#Fit the model
clf1=GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0))
clf1.fit(x,y)
clf2=GaussianProcessClassifier(kernel=1.0 * DotProduct(sigma_0=1.0)**2)
clf2.fit(x,y)

#Plot the non-linear decision function for each datapoint on the grid
z1=clf1.predict_proba(np.vstack((xx.ravel(), yy.ravel())).T)[:, 1].reshape(xx.shape)
fig=plt.figure(figsize=(15,5))
plt.subplot(121)
plt.imshow(z1,interpolation='nearest',extent=(xx.min(),xx.max(),yy.min(),yy.max()),aspect='auto',origin='lower',cmap=plt.cm.PuOr_r)
contours=plt.contour(xx,yy,z1,levels=[0.5],linewidths=2,linestyles='dashed')
plt.scatter(x[:,0],x[:,1],s=30,c=y, cmap=plt.cm.Paired,edgecolors='k')
plt.xticks(())
plt.yticks(())
plt.axis([-3,3,-3,3])
plt.title("RBF %s\n Log-Marginal-Likelihood:%.3f"%(clf1.kernel_,
          clf1.log_marginal_likelihood(clf1.kernel_.theta)),fontsize=12)

#Plot the non-linear decision function for each datapoint on the grid
z2=clf2.predict_proba(np.vstack((xx.ravel(), yy.ravel())).T)[:, 1].reshape(xx.shape)
plt.subplot(122)
plt.imshow(z2,interpolation='nearest',extent=(xx.min(),xx.max(),yy.min(),yy.max()),aspect='auto',origin='lower',cmap=plt.cm.PuOr_r)
contours=plt.contour(xx,yy,z2,levels=[0.5],linewidths=2,linestyles='dashed')
plt.scatter(x[:,0],x[:,1],s=30,c=y, cmap=plt.cm.Paired,edgecolors='k')
plt.xticks(())
plt.yticks(())
plt.axis([-3,3,-3,3])
plt.title("DotProduct %s\n Log-Marginal-Likelihood:%.3f"%(clf2.kernel_,
          clf2.log_marginal_likelihood(clf2.kernel_.theta)),fontsize=12)
plt.show()



