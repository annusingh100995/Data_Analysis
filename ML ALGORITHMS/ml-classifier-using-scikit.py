from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris.data[:,[2,3]] # all rows and column 2 and 3
y = iris.target
print('Class lables:' , np.unique(y)) # unique returns the three unique class label sorted in iris target

from sklearn.model_selection import train_test_split
X_train , X_test, y_train, y_test = train_test_split(X, y,test_size= 0.3, random_state=1, stratify=y)
# function divides the data into training and test data, test = 30% data

#bincount counts the occurance of each index in a positive array, ie the nuber of times the index's number is repeted
print('Lable counts in y:', np.bincount(y))
print('Lable count y_train :' , np.bincount(y_train))
print('Lable count y_test : ', np.bincount(y_test))

#Standard scaler is used for feature scaling for optimal performance

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train) # fit calculates the mean and standard deviation for each feature dimension
X_train_std = sc.transform(X_train) # tansform , transforms the data
X_test_std = sc.transform(X_test)

# Importing the classifier
from sklearn.linear_model import Perceptron

ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std,y_train)

y_pred = ppn.predict(X_test_std)
print('Misclassified samples: %d' %(y_test != y_pred).sum())
# tell the number of misclassified samples, where the predited is not equal to the test values, and also sum up those instances

# calculates the acccuracy score of the prediction
from sklearn.metrics import accuracy_score 
print('Accuracy %.2f ' % accuracy_score(y_test, y_pred))

#plotting the classification

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

#Method to draw the graph of the classification

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


X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X = X_combined_std, y = y_combined, classifier=ppn, test_idx=range(105,150))
plt.xlabel('petal lenght')
plt.ylabel('petal width')
plt.legend(loc='upper left')
plt.show()