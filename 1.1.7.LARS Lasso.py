import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import datasets

x,y=datasets.load_diabetes(return_X_y=True)

print('Computing regularization path using the LARS ...')

a,b,coefs=linear_model.lars_path(x,y,method='lasso',verbose=True)

xx = np.sum(np.abs(coefs), axis=0) # = np.sum(np.abs(coefs.T), axis=1)
xx /= xx[-1] # xx= xx/xx.max

plt.figure(figsize=(12,5))
plt.subplot(121)
plt.imshow(np.abs(coefs),cmap='gray_r')
plt.colorbar()
plt.xlabel('iterations')
plt.ylabel('|Coef|')

plt.subplot(122)
plt.plot(xx,coefs.T)
ymin,ymax=plt.ylim()
plt.vlines(xx,ymin,ymax,linestyle='dashed')
plt.xlabel('|coef|/max|coef|')
plt.ylabel('Coefficients')
plt.title('Lasso Path')
plt.axis('tight')
plt.show()


