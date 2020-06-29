import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNetCV,MultiTaskElasticNetCV,ElasticNet,MultiTaskElasticNet
from sklearn.metrics import mean_squared_error

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

#Optimize the alpha and rho value using MultiTaskElasticNetCV
#Optimize rho
feature_to_plot=1 #plot the 1th feature
plt.plot(coef[:,feature_to_plot],color='green',label='Real coefficient')

rho_list=['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9']
for rho in rho_list:
    plt.plot(MultiTaskElasticNet(alpha=MultiTaskElasticNetCV(cv=20,l1_ratio=0.1).fit(x,Y).alpha_,l1_ratio=MultiTaskElasticNetCV(cv=20).fit(x,Y).l1_ratio_).fit(x,Y).coef_[:,feature_to_plot],color='blue',label='rho=%2f'
             % float(rho))
plt.xlabel('Order number of tasks')
plt.ylabel('Co-efficient value of selected feature')
plt.legend()
plt.show()
    
#It is found that the coefficient is not sensitive to rho, so here the default value rho=0.5 was chosen to optimize alpha
model_multi_task_enet=MultiTaskElasticNetCV(cv=20).fit(x,Y)
print(r'Optimized alpha=',model_multi_task_enet.alpha_,'rho=',model_multi_task_enet.l1_ratio_)

plt.semilogx(model_multi_task_enet.alphas_,model_multi_task_enet.mse_path_,':')
plt.plot(model_multi_task_enet.alphas_,model_multi_task_enet.mse_path_.mean(axis=-1),'k',
         label='Average across the folds',linewidth=5)
plt.axvline(model_multi_task_enet.alpha_,label='alpha:CV estimate')
plt.axis('tight')
plt.ylim(1,6)
plt.xlabel(r'$\alpha$')
plt.title('Mean square error on each fold: coordinate descent')
plt.legend()
plt.show()

#Train model using MultiTaskElasticNet with the alpha optimized by MultiTaskElasticNetCV
coef_enet=np.array([ElasticNet(alpha=model_multi_task_enet.alpha_,l1_ratio=model_multi_task_enet.l1_ratio_).fit(x,y).coef_ for y in Y.T])
coef_multi_task_enet=MultiTaskElasticNet(alpha=model_multi_task_enet.alpha_,l1_ratio=model_multi_task_enet.l1_ratio_).fit(x,Y).coef_

#plot the positive and zero co-efficients of Lass and MultiTaskLass fitting
plt.figure(figsize=(8,5))
plt.suptitle('Coefficient non-zero location')

plt.subplot(121)
plt.imshow(abs(coef_enet),cmap='gray_r')
plt.colorbar()
plt.xlabel('Feature')
plt.ylabel('Time (or Task)')
plt.title('ElasticNet')

plt.subplot(122)
plt.imshow(abs(coef_multi_task_enet),cmap='gray_r')
plt.colorbar()
plt.xlabel('Feature')
plt.ylabel('Time (or Task)')
plt.title('MultiTaskElasticNet')
plt.show()

plt.subplot()
feature_to_plot=1 #plot the 1th feature
plt.plot(coef[:,feature_to_plot],color='green',label='Real coefficient') 
plt.plot(coef_enet[:,feature_to_plot],color='red',label='ElasticNet coefficient')
plt.plot(coef_multi_task_enet[:,feature_to_plot],color='blue',label='MultiTaskElasticNet coefficient')
plt.xlabel('Order number of tasks')
plt.ylabel('Co-efficient value of selected feature')
plt.legend()
plt.show()


