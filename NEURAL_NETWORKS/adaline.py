# Adaptive linear neuron 
# Weights are updated by minimising the cost function via gradient descent

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)



y = df.iloc[0:100,4].values # from row 0- 100 take the 4th column
y = np.where(y == 'Iris-setosa',-1,1) # where y = Iris setosa set it -1
X = df.iloc[0:100,[0,2]].values

plt.scatter(X[:50,0], X[:50,1],color='red',marker='o',label='setose')
plt.scatter(X[50:100,0], X[50:100,1],color='blue',marker='x',label='versicolor')
plt.xlabel('SEPAL LENGHT(CM)')
plt.ylabel('PETAL LENGHT(CM)')
plt.legend(loc='upper left')
plt.show()


class AdalineGD(object):
    """ ADaptive LInear Neuron classifier

    eta : float 
        Learning Rate between 0.0 and 1.0
    n_iter : int
        Passes over the training dataset
    random_state : 
        Random number genrator seed for random weight initialisation
    
    Attibutes:

    w_ : 1d-array
        weights after fitting
    cost_ : list
        Sum of squares cost functionvalue in each epoch
    """
    def __init__(self, eta=0.01, n_iter= 50, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.

        Parameters:

        X : array like, shape [n_samples, n_features]
        y : array like, shape [n_samples]
            Target values

        Returns:
            self object
        """
        # Initialising random weights
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale = 0.01, size = 1+X.shape[1]) 
        self.cost_ = [] # empty cost vector

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y-output)
            self.w_[1:] += self.eta*X.T.dot(errors)
            self.w_[0] += self.eta*errors.sum()
            cost = (errors**2).sum()/2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """ Caluculate net input """
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        return X

    def predict(self,X):
        return np.where(self,activation(self.net_input(X))>= 0.0,1,-1)
        # returns 1 if activation is > 0.0 or -1 is less than 0.0
    


if __name__ == '__main__':
    
    fig ,ax = plt.subplots(nrows=1,ncols=2,figsize=(10,4))
    ada1 = AdalineGD( eta = 0.01,n_iter = 10).fit(X, y)
    ada2 = AdalineGD( eta = 0.001 ,n_iter = 10).fit(X, y)
    ax[0].plot(range(1, len(ada1.cost_)+1), np.log10(ada1.cost_), marker = 'o')
    ax[1].plot(range(1, len(ada2.cost_)+1), np.log10(ada2.cost_), marker = 'o')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('log(sum-squered-errors)')
    ax[0].set_title('Adaline - Learning rate 0.01')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('log(sum-squered-errors)')
    ax[1].set_title('Adaline - Learning rate 0.001')
    plt.show()

