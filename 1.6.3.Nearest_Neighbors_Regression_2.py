import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.utils.validation import check_random_state
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge

#Load the faces datasets
data,targets=fetch_olivetti_faces(return_X_y=True)
train=data[targets<30]
test=data[targets>=30]
n_faces = 5
rng = check_random_state(4)
face_ids = rng.randint(test.shape[0], size=(n_faces, ))
test = test[face_ids, :]

n_pixels = data.shape[1]
# Upper half of the faces
x_train = train[:, :(n_pixels + 1) // 2]
# Lower half of the faces
y_train = train[:, n_pixels // 2:]
x_test = test[:, :(n_pixels + 1) // 2]
y_test = test[:, n_pixels // 2:]
#Fit model
knr_y_pred=KNeighborsRegressor().fit(x_train,y_train).predict(x_test)
ridge_y_pred=Ridge().fit(x_train,y_train).predict(x_test)
image_shape=(64,64)

true_face=np.hstack((x_test[1],y_test[1]))
knr_face=np.hstack((x_test[1],knr_y_pred[1]))
ridge_face=np.hstack((x_test[1],ridge_y_pred[1]))
plt.axis("off")
plt.subplot(131);plt.title('True face')
plt.imshow(true_face.reshape(image_shape),cmap=plt.cm.gray,interpolation="nearest")
plt.subplot(132);plt.title('KneighborsRegressor')
plt.imshow(knr_face.reshape(image_shape),cmap=plt.cm.gray,interpolation="nearest")
plt.subplot(133);plt.title('RidgeRegressor')
plt.imshow(ridge_face.reshape(image_shape),cmap=plt.cm.gray,interpolation="nearest")

plt.show()
