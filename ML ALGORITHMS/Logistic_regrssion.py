import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

class LogisticRegressionGD(object):
    """ Logistic Regression Classifier using gradient descent.
    ets: float
        Learning rate between 0 na d1
    n_iter : int
        passes over the training dataset
    random_state : int
        random number generator seed for random weight initialization
    
    Artibutes:

    w_ : 1d array
        Weights after fitting.
    cost_ : list
        sum of squares of cost function values in the each epoch
    """

    def __init__(self,eta=0.05, n_iter=100.0, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self,X,y):
        """ Fit training data.
        Parameters
        X: array like shape [n_sample, n_feature]
            Training vectors, where n_sample is the number of samples
            and n_feature is the number of features
        y : array like  shape : n_samples

        Returns:
        self: object
        """

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01,size=1+X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y-output)
            self.w_[1:] += self.eta*X.T.dot(errors)
            sefl.w_[0] == self.eta*errors.sum()

            # compute the logistic cost noe

            cost = (-y.dot(np.log(output))) - ((1-y).dot(np.log(1 - output)))
            self.cost_.append(cost)
        return self

    def net_input(self,X):
        return np.dot(X,self.w_[1:])+ self.w_[0]

    def activation(self,X):
        return 1./(1.+np.exp(-np.clip(z,-250,250)))
        
    def predict(self,X):
        """return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0,1,0)
    def plot_decision_regions(X,y,classifier,test_idx=None, resolution=0.02):
        markers = ('s','x','o','^','v')
        colors = ('red','blue','pink','lightgreen','cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])

        # plot the decision surface
        x1_min ,x1_max = X[:,0].min()-1, X[:,0].max() + 1
        x2_min ,x2_max = X[:,0].min()-1, X[:,0].max() + 1
        # creating the meshgrid, starting from x1_min to x1_max with increment of resolution
        xx1 , xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),np.arange(x2_min,x2_max,resolution))

        Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1,xx2,Z,alpha=0.3,cmap = cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        for idx , c1 in enumerate(np.unique(y)):
            plt.scatter(x=X[y==c1,0], y=X[y == c1,1], alpha=0.8, c=colors[idx], marker=markers[idx], label=c1, edgecolor = 'black')


        if test_idx:
            X_test , y_test = X[test_idx, :], y[test_idx]

            plt.scatter(X_test[:,0], X_test[:,1] , c='',edgecolor='black', alpha=1.0,linewidth=1, marker='o', s=100,label='test_set')



iris = datasets.load_iris()
X = iris.data[:,[2,3]] # all rows and column 2 and 3
y = iris.target
print('Class lables:' , np.unique(y)) # unique returns the three unique class label sorted in iris target

from sklearn.model_selection import train_test_split
X_train , X_test, y_train, y_test = train_test_split(X, y,test_size= 0.3, random_state=1, stratify=y)

X_train_01_subset = X_train[(y_train == 0)| (y_train == 1)]
y_train_01_subset = y_train[(y_train == 0)| (y_train == 1)]

lrgd = LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1)
lrgd.fit(X_train_01_subset,y_train_01_subset)
plot_decision_regions(X=X_train_01_subset, y = y_train_01_subset, classifier=lrgd)

plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc='upper left')
plt.show()