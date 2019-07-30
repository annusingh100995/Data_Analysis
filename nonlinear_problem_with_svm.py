import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import ListedColormap

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


np.random.seed(1)

"""Creating a simple dataset that is in the form of an XOR gate using the 
    logical_or function from NumPy
    100 Samples are assigned to class label 1 and
    100 Samples are assigned to class label -1 """

X_xor = np.random.randn(200,2) # Generate 200 random numbers , 2 is the numbre of columns
y_xor = np.logical_xor(X_xor[:,0] > 0 , X_xor[:,1]>0) # for y_xor selecting the numbers which are positive in both the rows
# this will retrun anarray of true or false

y_xor = np.where(y_xor,1,-1)# where false, y will be set to -1 and where true y is sset to 1

plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], c='b',marker='x', label='1')
# scatter plot, for all indexs in row 0 and 1 where y_xor is 1

plt.scatter(X_xor[y_xor == -1, 0], X_xor[y_xor == -1, 1], c='r',marker='s', label='-1')
# scatter plot, for all indexs in row 0 and 1 where y_xor is -1

plt.xlim([-3,3])
plt.ylim([-3,3])
plt.legend(loc='best')
#plt.show()

# Using SVM to classify the data

from sklearn.svm import SVC
svm = SVC(kernel='rbf', random_state=1,gamma=0.10, C=10.0)
svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor,y_xor,classifier=svm)
plt.legend(loc='upper left')
plt.title('C=10, gamma = 0.10')
plt.show()

svm = SVC(kernel='rbf', random_state=1,gamma=0.20, C=10.0)
svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor,y_xor,classifier=svm)
plt.legend(loc='upper left')
plt.title('C=10, gamma = 0.20')
plt.show()

svm = SVC(kernel='rbf', random_state=1,gamma=0.10, C=100.0)
svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor,y_xor,classifier=svm)
plt.legend(loc='upper left')
plt.title('C=100, gamma = 0.10')
plt.show()

""" gamma parameter is a cut off parameter for the Gaussian sphere
    If we increase the gamma, the influence/reach of the training sample increases,
    which leads to tighter and bumpier boundary"""