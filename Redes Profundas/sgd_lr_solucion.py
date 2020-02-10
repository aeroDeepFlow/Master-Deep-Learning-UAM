"""
Regresion Lineal por el metodo de Stochastic Gradient Descent
@author: Victor García Anaya
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score


class SGDLinearRegression(BaseEstimator, RegressorMixin):

    def __init__(self, max_iter=1000, eta=0.01):
        self.max_iter = max_iter
        self.eta = eta

    def fit(self, X, y):
        if len(X.shape) == 1: X = X.reshape((-1, 1)) # checking dimensions
        m = X.shape[0] # lenght of first dimension
        
        #initializing weigths and storing in beta property
        self.beta_ = np.random.rand(X.shape[1] + 1, 1) 
        
        #including interceptor into model X matrix
        Xe = np.c_[np.ones((m, 1)), X] 
        
        for _ in range(self.max_iter):
            index = np.random.randint(m) #chosing randon index
            xn = Xe[index:index+1]
            yn = y[index:index+1]
            
            # computing gradient
            grads = 2 * xn.T.dot(xn.dot(self.beta_) - yn)
            
            self.beta_ = self.beta_ - self.eta * grads
        
    def predict(self, X):
        if len(X.shape) == 1: X = X.reshape((-1, 1)) # checking dims
        m = X.shape[0] # len first dim
        Xe = np.c_[np.ones((m, 1)), X] #including interceptor into model matrix
        
        return Xe @ self.beta_

            
            
    def score(self, X, y):
        preds = self.predict(X)
        return r2_score(y, preds)


from sklearn.datasets import load_boston
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


X, y = load_boston(return_X_y=True)

# testing SGDLinearRegressor Class
sgd_lr = Pipeline([('stds', StandardScaler()), ('sgd_lr', SGDLinearRegression())])
sgd_lr.fit(X, y)
print("R2 SGDLR: " + str(sgd_lr.score(X, y)))

# testing scikit SGDRegressorRegressor Class
sgd = Pipeline([('stds', StandardScaler()), ('sgd', SGDRegressor())])
sgd.fit(X, y)
print("R2 SGDR: " + str(sgd.score(X, y)))
