import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

#Import iris dataset
iris=datasets.load_iris() #x=[attribute1, attribute2, attribute3, attribute4];y=[0,1,2]
x=iris.data[:,:2]
y=iris.target
#Fit the model using isotropic and anisotropic RBF
gpc_rbf_isotropic=GaussianProcessClassifier(kernel=1.0*RBF([1.0])).fit(x, y)
gpc_rbf_anisotropic=GaussianProcessClassifier(kernel=1.0*RBF([1.0,1.0])).fit(x, y)

#Plot
xx,yy=np.meshgrid(np.arange(x[:,0].min()-1,x[:,0].max()+1,0.02),
                  np.arange(x[:,1].min()-1,x[:,0].max()+1,0.02))
plt.subplot(121)                                              
z=gpc_rbf_isotropic.predict_proba(np.c_[xx.ravel(),yy.ravel()]).reshape((xx.shape[0], xx.shape[1], 3))
plt.imshow(z,extent=(x[:,0].min()-1,x[:,0].max()+1,x[:,1].min()-1,x[:,0].max()+1),origin="lower")
plt.scatter(x[:,0],x[:,1],c=np.array(['r','g','b'])[y],edgecolors=(0,0,0))
plt.title("%s, LML: %.3f"%('Isotropic RBF',gpc_rbf_isotropic.log_marginal_likelihood(gpc_rbf_isotropic.kernel_.theta)))
plt.xlim(xx.min(),xx.max());plt.ylim(yy.min(),yy.max())
plt.xlabel('x1');plt.ylabel('x2')#;plt.axis('tight')

plt.subplot(122)
z=gpc_rbf_anisotropic.predict_proba(np.c_[xx.ravel(),yy.ravel()]).reshape((xx.shape[0], xx.shape[1], 3))
plt.imshow(z,extent=(x[:,0].min()-1,x[:,0].max()+1,x[:,1].min()-1,x[:,0].max()+1),origin="lower")
plt.scatter(x[:,0],x[:,1],c=np.array(['r','g','b'])[y],edgecolors=(0,0,0))
plt.title("%s, LML: %.3f"%('Isotropic RBF',gpc_rbf_anisotropic.log_marginal_likelihood(gpc_rbf_anisotropic.kernel_.theta)))
plt.xlim(xx.min(),xx.max());plt.ylim(yy.min(),yy.max())
plt.xlabel('x1');plt.ylabel('x2')#;plt.axis('tight')
plt.show()


