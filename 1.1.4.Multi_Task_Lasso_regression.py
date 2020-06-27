import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import MultiTaskLasso,Lasso

rng=np.random.RandomState(0) #Ensure that each time we can get the same random numbers

#Generate some 2D coefficients with sine waves with random frequency and phase
n_samples,n_features,n_tasks = 100,30,40
n_relevant_features=15  #half features are relevant to tasks, but another half are random numbers
coef=np.zeros((n_tasks,n_features))
times=np.linspace(0,2*np.pi,n_tasks) #split 0~2pi into n_tasks parts,interval=2pi/(n_tasks-1)

for k in range(n_relevant_features):
    coef[:,k]=np.sin((1.+rng.randn(1))*times+3*rng.randn(1))

x=rng.randn(n_samples,n_features)   #random 2D array with n_samples lines and n_features columns
Y=np.dot(x,coef.T)+rng.randn(n_samples,n_tasks) #2D array with n_samples lines and n_tasks columns

coef_lasso_=np.array([Lasso(alpha=0.5).fit(x,y).coef_ for y in Y.T])
coef_multi_task_lasso_=MultiTaskLasso(alpha=1.).fit(x,Y).coef_

#plot the positive and zero co-efficients of Lass and MultiTaskLass fitting
plt.figure(figsize=(8,5))
plt.suptitle('Coefficient non-zero location')

plt.subplot(121)
plt.imshow(abs(coef_lasso_),cmap='gray_r')
plt.colorbar()
plt.xlabel('Feature')
plt.ylabel('Time (or Task)')
plt.title('Lasso')

plt.subplot(122)
plt.imshow(abs(coef_multi_task_lasso_),cmap='gray_r')
plt.colorbar()
plt.xlabel('Feature')
plt.ylabel('Time (or Task)')
plt.title('MultiTaskLasso')
plt.show()

plt.subplot()
feature_to_plot=1 #plot the 1th feature
plt.plot(coef[:,feature_to_plot],color='green',label='Real coefficient') 
plt.plot(coef_lasso_[:,feature_to_plot],color='red',label='Lasso coefficient')
plt.plot(coef_multi_task_lasso_[:,feature_to_plot],color='blue',label='MultiTaskLasso coefficient')
plt.xlabel('Order number of tasks')
plt.ylabel('Co-efficient value of selected feature')
plt.legend()
plt.show()


