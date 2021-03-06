import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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



class Perceptron(object):
    """ Preceptron classifiers:
    Parameters:
    eta: float : Learning rate between 0.0 and 1.0
    n_iter" int passes over the training dataset
    random_state = int: random number generator seed for random weight

    w_ 1d array  weights after fitting
    errors_ list 
    number of missclassifications
    """
def __init__(self,eta=0.01, n_iter=50, random_state=1):
    self.eta = eta
    self.n_iter = n_iter
    self.random_state = random_state

def fit(self, X, y):
    """Fitting training dataset.

    Parameters:
    X:array like , shape [n_samples, n_features]
    Training vector where n_sample is the number of sample 
    and the n_feature is the number of features

    y: array like shape = [n_samples]
    target values
    Returns: 
    self:object
    """
    rgen = np.random.RandomState(self.random_state)
    self.w_ = rgen.normal(loc=0.0,scale=0.01,size=1 + X.shape[1])
    self.errors_ = []

    for _ in range(self.n_iter):
        errors = 0
        for xi, target in zip(X,y):
            update = self.eta *(target - self.predict(xi))
            self.w_[1:] += update*xi
            self.w_[0] += update
            errors += int(update != 0.0)
        self.errors_.append(errors)
    return self

def net_input(self, X):
    """Calculate net input"""
    return np.dot(X, self.w_[1:]) + self.w_[0]

def predict(self,X):
    return np.where(self.net_input(X) >=0.0,1,-1)

    #ax[0].xlabel('Epoch')
    #ax[0].ylabel('log(sum-squered-errors)')
    #ax[0].set_title('Adaline - Learning rate 0.01')
    
    
    #ax[1].set_xlabel('Epoch')
    #ax[1].set_ylabel('log(sum-squered-errors)')
    #ax[1].set_title('Adaline - Learning rate 0.001')
   