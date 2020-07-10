import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

#Generate data
x=np.linspace(0,10,100)
noise=np.random.randn(100)
y=x*np.sin(x)+0.5*noise
#Fitting with Ridge vs. PolynomialFeatures+Ridge
ridge=Ridge()
ridge.fit(x.reshape(-1,1),y)
y_pred_ridge=ridge.predict(x.reshape(-1,1))

for degree in [3,4,5]:
    poly_ridge=make_pipeline(PolynomialFeatures(degree),Ridge())
    poly_ridge.fit(x.reshape(-1,1),y)
    y_pred_poly_ridge=poly_ridge.predict(x.reshape(-1,1))
    plt.plot(x,y_pred_poly_ridge,label='Poly_Ridge, degree=%f' % degree)

    ####### Another way to invoke PolynomialFeatures
    #poly=PolynomialFeatures(degree=2)
    #z=poly.fit_transform(x.reshape(-1,1))
    #y_pred_poly=Ridge().fit(z,y).predict(z)

plt.scatter(x,y)
plt.plot(x,y-0.5*noise,label='Ground truth')
plt.plot(x,y_pred_ridge,label='Ridge')
plt.legend(loc=0)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
