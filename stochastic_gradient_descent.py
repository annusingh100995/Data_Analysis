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

class AdalineSGD(object):
    """ ADAptive LInear Neuron classifier
    Parameter:

    eta : float
        Learning rate between 0.0 and 1.0
    n_iter : int
        Passes over the trainig dataset
    shuffle : bool , default = true
        Shuffels training data at every epoch to prevent cycles
    random_state : int 
        Random number generator seed for random weight initialization
    
    Arrtibutes:

    w_ : 1d array
        weights after fitting
    cost_ : list
        sum of sqaure cost functions averaged overall the training samples in each epoches
    """
    def __init__(self,eta=0.01, n_iter=10, shuffle=True, random_state = None):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.w_initilized = False
        self.random_state = random_state
    
    def fit(self,X,y):

        """
        Parameters:
        X: array like , shape [n_number, n_feature]
            Training vectors
        y : aray like n_samples
            target values
        
        Returns self
        """
        self._initialize_weights= (X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X,y = self._shuffle(X,y)
            cost = []
            for xi , target in zip(X,y):
                cost.append(self._update_weights(xi,target))
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self,X,y):
        """Fit trainng data withoout reintialisting the weight"""
        if not self.w_initiliazed:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi , target in zip(X,y):
                self._update_weights(xxi, target)
        else:
            self._update_weightd(X,y)
        return self

    def _shuffle(self,X,y):
        self.rgen = np.random.RandomState(self.random_state)
        r = self.rgen.permutation(len(y))
        return X[r], y[r]
    
    def _initialize_weight(self, m):
        """Initialize weights to small random number"""
        self.rgen = np.random.RandomState(self,random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1+m)
        self.w_initialized = True
    
    def _update_weights(self,xi,target):
        output = self.activation(self.net_input(xi))
        errors = (target-output)
        self.w_[1:] += self.eta*xi.dot(error)
        self.w_[0] += self.eta*error
        cost = 0.5*errro**2
        return cost

    def net_input(self,X):
        return np.dot(X, self.w_[1:]+self.w_[0])

    def activation(self, X):
        return X

    def predict(self,X):
        return np.where(self.activation(self.net_input(X)) >= 0.0,1,-1)


if __name__ == '__main__':
    ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
    X_std = np.copy(X)
    ada.fit(X_std,y)    
    plot_decision_regions(X_std, y, classfier=ada)
    plt.title('Adaline- Stochastic Gradient Descent')
    plt.xlabel('sepal length')
    plt.ylabel('petal length')
    plt.legend(loc='upper left')
    plt.show()
    plt.plot(range(1,len(ada.cost_)+1), ada.cost_, marker='0')
    plt.xlabel('Epochs')
    plt.ylabel('Average Cost')
    plt.show()
















