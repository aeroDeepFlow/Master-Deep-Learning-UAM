"""
Multilayer Perceptron Regressor
@author: Victor García Anaya
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score


class CustomMLPRegressor(BaseEstimator, RegressorMixin):
    ####################################################################################################################
    # This class is designed to mimic a multilayer Perceptron Regressor for, theoretically, any number of hidden layer.
    # The Relu Activation function is used at every hidden layer and it is training in the so-called Mini-Batch method
    # employing a set of 64 samples.  
    #
    # Class Attributes:
    # hidden_layer_sizes: number of hidden layer sizes. A tuple with the number of neurons per layer. eg (10, 2, 3)
    # max_iter: maximum number of the iteration loop when training the neural network
    # eta: constant Learning Rate to be used in gradient descent algorithm to update weights and bias  
    #
    # Class Methods:
    #__init__: it initializes CustomMLPRegressor object using Class Attributes 
    #__Relu: it computes Relu activation function taking X as input variable and returns Relu(X)
    #__Relu_backward: perform dZ_l = dA_l * f'(z_l). f' Relu derivative
    #__forward: it performs forward computation step. It takes X as input and returns a n-list containig the following:
    #           * [......, (A_prev_l, W_l, b_l), Z_l), ......, ((A_prev_L-1, W_L, b_L), Z_L)]
    #           * A_prev_l: activation of previous layer l
    #           * W_l: weight matrix in layer l
    #           * b_l: bias matrix in layer l          
    #__backprop: it performs backprop activation step. It takes X, y and the output returned by forward (backup).   
    # fit: overwritten inherit method producing the traning of neural network. It takes X, y arguments. 
    # predict: overwritten inherit method producing the prediction value of the trained neural network. It takes X samples
    #            to be predicted.
    # score: it computes R2 perfomance metric. It takes X and y samples to be predicted and evaluated using R2.
    ####################################################################################################################
    def __init__(self, hidden_layer_sizes=(10, ), max_iter=1000, eta=0.01):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.eta = eta
       
    @staticmethod
    def __Relu(X):
        return np.maximum(0, X)

    @staticmethod
    def __Relu_backward(dA, backup_layer_1):
        Z = backup_layer_1
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ   

    def __forward(self, X):
        A = X.T
       
        L = len(self.parameters_) // 2

        # list that stores backups per layer. To be used in __backprop
        backup = []
    
        for i in range(1, L + 1): 
            A_prev = A
            # computing Z per each layer
            Z = self.parameters_["W" + str(i)] @ A_prev + self.parameters_["b" + str(i)]
            # appying activation function to Z if layer is not the output layer
            A = CustomMLPRegressor.__Relu(Z) if i < L else Z 

            # list with [......, (A_(l-1), Wl, bl), Zl), .....]
            backup.append(( (A_prev, self.parameters_["W" + str(i)], self.parameters_["b" + str(i)]), Z )) 

        # cache values per layer. To be used ub __backprop
        return backup 

    def __backprop(self, y, backup):
        # defining a dictionary that stores gradients 
        grads = {}

        L = len(backup) 
        y = y.reshape(1, -1)

        # current cache. The output layer in this case 
        backup_layer = backup[-1] 
        m = backup_layer[0][0].shape[1]

        # initialization backprop dLoss/dAL
        dAL = backup_layer[1] - y 
        # in output layer this condition is fulfilled
        dZ = dAL 

        # output layer gradients
        grads["dW" + str(L)] = (dZ @ backup_layer[0][0].T) / m
        grads["db" + str(L)] = np.sum(dZ, axis=1, keepdims=True) / m
        dA = backup_layer[0][1].T @ dZ 
        
        # loop for computing hidden layers gradients
        for i in reversed(range(1, L)):
            # current cache
            backup_layer = backup[i - 1] 
            # Activation dA -> linear dZ in layer l
            dZ = CustomMLPRegressor.__Relu_backward(dA, backup_layer[1]) 

            m = backup_layer[0][0].shape[1]

            # hidden layer gradients 
            grads["dW" + str(i)] = (dZ @ backup_layer[0][0].T) / m
            grads["db" + str(i)] = np.sum(dZ, axis=1, keepdims=True) /m
            dA = backup_layer[0][1].T @ dZ

        # updating parameters
        L = len(self.parameters_) // 2

        # updating weights and bias using gradient descent algorithm
        for i in range(L):
            self.parameters_["W" + str(i + 1)] -= self.eta * grads["dW" + str(i + 1)]
            self.parameters_["b" + str(i + 1)] -= self.eta * grads["db" + str(i + 1)]

    def fit(self, X, y):
        # defining class attribute (parameters) that stores, in a dictionary, layer weight and bias 
        self.parameters_ = {}
        # definig class attribute (batch) that stores the batch length for training the net in mini-batch mode 
        self.batch_ = 64
        if len(X.shape) == 1: X = X.reshape((-1, 1))
        n = X.shape[0]
        n_inputs = X.shape[1] 

        layers = [n_inputs, *self.hidden_layer_sizes, 1]
        L = len(layers)

        # Initializing weights and bias
        for i in range(1, L):
            self.parameters_['W' + str(i)] = np.random.randn(layers[i], layers[i-1])
            self.parameters_['b' + str(i)] = np.zeros((layers[i], 1))
        
        # iteration loops for max_iter epochs
        for _ in range(self.max_iter):
            # iteration loops for mini-bath training samples
            for i in range(0, n, self.batch_):
                X_train = X[i:i + self.batch_,:]
                y_train = y[i:i + self.batch_]
                
                # applying forward and backpropagation steps
                backups = self.__forward(X_train)
                self.__backprop(y_train, backups)
        
    def predict(self, X):
        # predict method return the last element (ZL = Z_out) of the last component (last layer) in cache-list
        # returned by __forward class method
        return np.squeeze(self.__forward(X)[-1][-1])

    def score(self, X, y):
        preds = self.predict(X)
        return r2_score(y, preds)



from sklearn.datasets import load_boston
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


X, y = load_boston(return_X_y=True)

# testing CustomMLPRegressor Class
cmlp = Pipeline([('stds', StandardScaler()), ('cmlp', CustomMLPRegressor())])
cmlp.fit(X, y)
print("CMLPR R2: " + str(cmlp.score(X, y)))


#Testing scikit MLPregressor
from sklearn.neural_network import MLPRegressor

mlp = Pipeline([('stds', StandardScaler()), ('mlp', MLPRegressor(hidden_layer_sizes=(10,), max_iter = 5000))])
mlp.fit(X, y)
print("MLPR R2: " + str(mlp.score(X, y)))




